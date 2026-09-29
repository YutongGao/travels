"""Photo pipeline for the travels site.

    python tools/photos.py scan [SRC]     read every original, assign it to a trip day, cache analysis
    python tools/photos.py build [SRC]    pick photos per day, export web sizes, write data/photos.json
                                          and the review page .cache/report.html

Hand edits live in two files that the pipeline never overwrites:
    tools/overrides.json   per day: "only" (exact list), "add", "remove", "cover" — original file names
    data/captions.json     original file name -> {"zh": ..., "en": ...}

Originals are never modified. Everything intermediate goes to .cache/ (git-ignored).
"""
import html, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat, ExifTags
import pillow_heif

pillow_heif.register_heif_opener()

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
PREVIEW = CACHE / "preview"
SCAN = CACHE / "scan.json"
DEFAULT_SRC = ROOT / "Photos-1-001"

REPORT = CACHE / "report.html"
OVERRIDES = ROOT / "tools" / "overrides.json"
CAPTIONS = ROOT / "data" / "captions.json"
PHOTOS_JSON = ROOT / "data" / "photos.json"
PHOTOS_DIR = ROOT / "photos"

IMAGE_EXT = {".heic", ".jpg", ".jpeg", ".png"}
PREVIEW_EDGE = 480
DAY_STARTS_AT = 3  # photos before 03:00 local belong to the previous day

PER_DAY = 9          # fills the 3-column gallery
FULL_EDGE, FULL_Q = 1600, 80
THUMB_EDGE, THUMB_Q = 720, 72   # thumbs also fill the large first cell on the trip page
BURST_SECONDS, BURST_BITS = 10, 16   # same moment, similar frame
NEAR_SECONDS, NEAR_BITS = 300, 6     # a few minutes apart, nearly identical frame
BLUR_RATIO = 0.35    # sharpness below this fraction of the day's median = blurry

# Where the phone physically was, as a UTC timeline. Fixed offsets are exact here:
# none of these zones observe DST between November and January.
PST, JST, TRT = (timezone(timedelta(hours=h), n) for h, n in ((-8, "PST"), (9, "JST"), (3, "TRT")))
ZONES = [  # (in effect until this UTC instant, zone)
    (datetime(2024, 11, 23, 6, tzinfo=timezone.utc), PST),  # until landing at Narita, 11.23 evening JST
    (datetime(2024, 12, 16, 5, tzinfo=timezone.utc), JST),  # until the Haneda -> Istanbul flight, 12.16 afternoon
    (datetime.max.replace(tzinfo=timezone.utc), TRT),
]


def zone_at(utc):
    return next(z for until, z in ZONES if utc < until)


def day_index():
    """Map calendar date -> (trip id, day key). A day with n nights also covers the following n-1 dates."""
    trips = json.loads((ROOT / "data" / "trips.json").read_text(encoding="utf-8"))
    out = {}
    for t in trips:
        first_month = int(t["days"][0]["d"][:2])
        year0 = 2024
        for d in t["days"]:
            m, dd = map(int, d["d"].split("."))
            y = year0 + (1 if m < first_month else 0)
            start = date(y, m, dd)
            for k in range(d.get("n", 1)):
                out[start + timedelta(days=k)] = (t["id"], d["d"])
    return out


# ---------- per-file analysis (runs in worker processes) ----------

def dhash(gray):
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f"{bits:016x}"


def sharpness(gray):
    """Variance of the Laplacian at a fixed working size; higher = sharper."""
    g = gray.copy()
    g.thumbnail((1024, 1024))
    lap = g.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
    return round(ImageStat.Stat(lap).var[0], 1)


def analyse(path_str):
    p = Path(path_str)
    rec = {"name": p.name, "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
    try:
        im = Image.open(p)
        ex = im.getexif()
        sub = ex.get_ifd(ExifTags.IFD.Exif)
        rec["model"] = ex.get(272)
        rec["dto"] = sub.get(36867) or ex.get(306)
        rec["oto"] = sub.get(36881)
        im = ImageOps.exif_transpose(im)
        rec["w"], rec["h"] = im.size
        if not rec["model"]:
            return rec  # not a camera photo; no need to decode further
        gray = im.convert("L")
        rec["dhash"] = dhash(gray)
        rec["sharp"] = sharpness(gray)
        pv = im.convert("RGB")
        pv.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
        pv.save(PREVIEW / (p.stem + ".jpg"), quality=80)
    except Exception as e:  # keep going; report it
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# ---------- classification ----------

def classify(rec, days):
    """Fill in utc / local / zone / day, or a reason the file is excluded."""
    for k in ("utc", "local", "zone", "trip", "day", "reason", "note"):
        rec.pop(k, None)
    if rec.get("error"):
        rec["reason"] = "unreadable"
        return
    if not rec.get("model"):
        rec["reason"] = "screenshot" if rec["name"].lower().endswith(".png") else "not-camera"
        return
    if not rec.get("dto"):
        rec["reason"] = "no-time"
        return
    local = datetime.strptime(rec["dto"], "%Y:%m:%d %H:%M:%S")
    if rec.get("oto"):
        sign = 1 if rec["oto"][0] == "+" else -1
        hh, mm = map(int, rec["oto"][1:].split(":"))
        phone_tz = timezone(sign * timedelta(hours=hh, minutes=mm))
        utc = local.replace(tzinfo=phone_tz).astimezone(timezone.utc)
    else:
        # no offset recorded: assume the phone clock showed trip-local time
        guess = local.replace(tzinfo=JST if local < datetime(2024, 12, 16, 14) else TRT)
        utc = guess.astimezone(timezone.utc)
        rec["note"] = "no-offset"
    tz = zone_at(utc)
    loc = utc.astimezone(tz)
    if rec.get("oto") and loc.utcoffset() != utc.astimezone(phone_tz).utcoffset():
        rec["note"] = f"phone-offset {rec['oto']} vs {tz.tzname(None)}"
    rec["utc"] = utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["local"] = loc.strftime("%Y-%m-%d %H:%M:%S")
    rec["zone"] = tz.tzname(None)
    d = (loc - timedelta(hours=DAY_STARTS_AT)).date()
    if d not in days:
        rec["reason"] = "outside-trip"
        return
    rec["trip"], rec["day"] = days[d]


# ---------- commands ----------

def cmd_scan(src):
    src = Path(src)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    old = {r["name"]: r for r in json.loads(SCAN.read_text(encoding="utf-8"))} if SCAN.exists() else {}
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
    todo, recs = [], []
    for p in files:
        r = old.get(p.name)
        st = p.stat()
        if r and r["size"] == st.st_size and r["mtime"] == int(st.st_mtime) and "error" not in r:
            recs.append(r)
        else:
            todo.append(str(p))
    print(f"{len(files)} images, {len(recs)} cached, {len(todo)} to analyse")
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
            for i, r in enumerate(ex.map(analyse, todo, chunksize=4), 1):
                recs.append(r)
                if i % 100 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
    days = day_index()
    for r in recs:
        classify(r, days)
    recs.sort(key=lambda r: (r.get("utc") or "~", r["name"]))
    SCAN.write_text(json.dumps(recs, ensure_ascii=False, indent=0), encoding="utf-8")
    summarise(recs, days)


def summarise(recs, days):
    from collections import Counter
    excl = Counter(r["reason"] for r in recs if r.get("reason"))
    notes = Counter(r["note"] for r in recs if r.get("note"))
    per_day = Counter((r["trip"], r["day"]) for r in recs if not r.get("reason"))
    print("\nexcluded:", dict(excl))
    print("notes:", dict(notes))
    print(f"assigned to a day: {sum(per_day.values())}")
    seen = []
    for d in sorted(days):
        key = days[d]
        if key in seen:
            continue
        seen.append(key)
        print(f"  {key[0]:12} {key[1]}  {per_day.get(key, 0):4}")
    out = [r for r in recs if r.get("reason") == "outside-trip"]
    if out:
        print("outside trip range:", out[0]["local"], "...", out[-1]["local"])


# ---------- selection ----------

def ham(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def ts(r):
    return datetime.strptime(r["utc"], "%Y-%m-%dT%H:%M:%SZ")


def pick_day(cands, ov):
    """Return (picked names in display order, cover name, {name: why dropped})."""
    by = {r["name"]: r for r in cands}
    why = {}
    if "only" in ov:
        picked = [n for n in ov["only"] if n in by]
        for r in cands:
            if r["name"] not in picked:
                why[r["name"]] = "manual: not in list"
        return picked, ov.get("cover") or (picked[0] if picked else None), why

    rs = sorted(cands, key=ts)
    # 1. duplicates: keep the sharpest of each group
    groups = []
    for r in rs:
        g = next((g for g in groups[-6:] if any(
            (abs((ts(r) - ts(x)).total_seconds()) <= BURST_SECONDS and ham(r["dhash"], x["dhash"]) <= BURST_BITS) or
            (abs((ts(r) - ts(x)).total_seconds()) <= NEAR_SECONDS and ham(r["dhash"], x["dhash"]) <= NEAR_BITS)
            for x in g)), None)
        if g is not None:
            g.append(r)
        else:
            groups.append([r])
    kept = []
    for g in groups:
        best = max(g, key=lambda x: x["sharp"])
        kept.append(best)
        for x in g:
            if x is not best:
                why[x["name"]] = f"duplicate of {best['name']}"
    # 2. blur, relative to the day
    if kept:
        med = sorted(x["sharp"] for x in kept)[len(kept) // 2]
        sharp = [x for x in kept if x["sharp"] >= med * BLUR_RATIO]
        for x in kept:
            if x not in sharp:
                why[x["name"]] = f"blurry ({x['sharp']:.0f} vs day median {med:.0f})"
        kept = sharp
    # 3. spread across the day: equal-sized runs in time order, best of each run
    n = min(PER_DAY, len(kept))
    picked = []
    for i in range(n):
        run = kept[len(kept) * i // n: len(kept) * (i + 1) // n]
        best = max(run, key=lambda x: x["sharp"])
        picked.append(best)
        for x in run:
            if x is not best:
                why[x["name"]] = "not picked (another photo from the same stretch of the day)"
    # manual adjustments
    names = [x["name"] for x in picked]
    for nme in ov.get("remove", []):
        if nme in names:
            names.remove(nme)
            why[nme] = "manual: removed"
    for nme in ov.get("add", []):
        if nme in by and nme not in names:
            names.append(nme)
            why.pop(nme, None)
    names.sort(key=lambda n: ts(by[n]))
    cover = ov.get("cover")
    if cover not in names:
        land = [n for n in names if by[n]["w"] > by[n]["h"]] or names
        cover = max(land, key=lambda n: by[n]["sharp"]) if land else None
    return names, cover, why


# ---------- export ----------

def export_one(job):
    src, full, thumb = job
    im = ImageOps.exif_transpose(Image.open(src))
    icc = im.info.get("icc_profile")
    im = im.convert("RGB")
    for path, edge, q in ((full, FULL_EDGE, FULL_Q), (thumb, THUMB_EDGE, THUMB_Q)):
        if Path(path).exists():
            continue
        out = im.copy()
        out.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        # no exif= / xmp= passed, so no GPS, camera or time metadata is written; only the colour profile
        out.save(path, "WEBP", quality=q, method=6, icc_profile=icc)
    with Image.open(full) as f:
        return f.size


def cmd_build(src):
    src = Path(src)
    recs = json.loads(SCAN.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8")) if OVERRIDES.exists() else {}
    captions = json.loads(CAPTIONS.read_text(encoding="utf-8")) if CAPTIONS.exists() else {}
    trips = json.loads((ROOT / "data" / "trips.json").read_text(encoding="utf-8"))
    by_day = {}
    for r in recs:
        if not r.get("reason"):
            by_day.setdefault((r["trip"], r["day"]), []).append(r)

    plan, jobs = {}, []
    for t in trips:
        for d in t["days"]:
            key = (t["id"], d["d"])
            cands = by_day.get(key, [])
            if not cands:
                continue
            names, cover, why = pick_day(cands, overrides.get(f"{key[0]}/{key[1]}", {}))
            out = PHOTOS_DIR / key[0] / key[1]
            out.mkdir(parents=True, exist_ok=True)
            want = set()
            for n in names:
                stem = Path(n).stem
                want |= {stem + ".webp", stem + ".thumb.webp"}
                jobs.append((str(src / n), str(out / (stem + ".webp")), str(out / (stem + ".thumb.webp"))))
            for f in out.iterdir():  # drop files from earlier runs that are no longer picked
                if f.name not in want:
                    f.unlink()
            plan[key] = (names, cover, why, cands)

    print(f"exporting {len(jobs)} photos")
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        sizes = dict(zip((j[0] for j in jobs), ex.map(export_one, jobs)))

    data = {}
    for (tid, day), (names, cover, why, cands) in plan.items():
        if not names:  # every photo removed by hand: the site shows the day's map instead
            continue
        items = []
        for n in names:
            stem = Path(n).stem
            w, h = sizes[str(src / n)]
            c = captions.get(n, {})
            items.append({"o": n, "f": f"photos/{tid}/{day}/{stem}.webp", "t": f"photos/{tid}/{day}/{stem}.thumb.webp",
                          "w": w, "h": h, "zh": c.get("zh", ""), "en": c.get("en", "")})
        data.setdefault(tid, {})[day] = {"cover": names.index(cover) if cover in names else 0, "items": items}
    for tid in list(data):  # remove day folders that no longer have photos
        for dirp in (PHOTOS_DIR / tid).iterdir():
            if dirp.is_dir() and dirp.name not in data[tid] and not any(dirp.iterdir()):
                dirp.rmdir()

    lines = []
    for tid, days in data.items():
        dl = [f'  {json.dumps(day)}: {{"cover": {p["cover"]}, "items": [\n' +
              ",\n".join("    " + json.dumps(it, ensure_ascii=False) for it in p["items"]) + "\n  ]}"
              for day, p in days.items()]
        lines.append(f" {json.dumps(tid)}: {{\n" + ",\n".join(dl) + "\n }")
    PHOTOS_JSON.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")

    write_report(recs, plan, captions)
    total = sum(len(p[0]) for p in plan.values())
    size = sum(f.stat().st_size for f in PHOTOS_DIR.rglob("*.webp"))
    print(f"{total} photos on {len(plan)} days, {size / 1e6:.1f} MB in photos/")
    print(f"review: {REPORT}")


def write_report(recs, plan, captions):
    e = html.escape
    fig = lambda n, label, cls="": (f'<figure class="{cls}"><img loading="lazy" src="preview/{e(Path(n).stem)}.jpg">'
                                    f"<figcaption><b>{e(n)}</b> {label}</figcaption></figure>")
    by = {r["name"]: r for r in recs}
    parts = []
    for (tid, day), (names, cover, why, cands) in plan.items():
        picked = "".join(fig(n, e(by[n]["local"][11:16]) + (" ★ 封面" if n == cover else "") +
                             (f"<br>{e(captions[n]['zh'])}" if n in captions else ""), "pick") for n in names)
        dropped = "".join(fig(r["name"], e(r["local"][11:16] + " · " + why.get(r["name"], "")))
                          for r in sorted(cands, key=lambda r: r["utc"]) if r["name"] not in names)
        parts.append(f'<section><h2>{tid} · {day} <small>选中 {len(names)} / 共 {len(cands)}</small></h2>'
                     f'<div class="g">{picked}</div><details><summary>未选中的 {len(cands) - len(names)} 张</summary>'
                     f'<div class="g">{dropped}</div></details></section>')
    excl = {}
    for r in recs:
        if r.get("reason"):
            excl.setdefault(r["reason"], []).append(r["name"])
    parts.append("<section><h2>排除的文件</h2>" + "".join(
        f"<p><b>{e(k)}</b>（{len(v)}）：{e(', '.join(v))}</p>" for k, v in excl.items()) + "</section>")
    REPORT.write_text(f"""<!doctype html><meta charset="utf-8"><title>照片报告</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;background:#f4f4f1;color:#222}}
h2{{font-size:18px;margin:32px 0 8px}}h2 small{{color:#777;font-weight:400}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}}
figure{{margin:0}}img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:3px;display:block;opacity:.75}}
.pick img{{opacity:1;outline:3px solid #2b7}}figcaption{{font-size:12px;color:#555;margin-top:3px}}
summary{{cursor:pointer;margin:10px 0;color:#555}}p{{word-break:break-all}}</style>
<h1>照片报告</h1><p>绿框 = 选中。改动写进 tools/overrides.json，然后运行 python tools/photos.py build</p>
{''.join(parts)}""", encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("scan", "build"):
        sys.exit(__doc__)
    src = args[1] if len(args) > 1 else DEFAULT_SRC
    if args[0] == "scan":
        cmd_scan(src)
    else:
        cmd_build(src)
