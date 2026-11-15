r"""A wrong red line in someone else's map, found on this PC (roadmap 30.3 and 30.4).

bench/health_truth.py measures the library health check on the server's stages: every
ranked map with two red lines or more, graded intact and with one seeded line moved
5 / 15 / 30 ms. This is the same protocol (its ``shifted``, ``flags`` and ``summarise``)
on the local Songs folder, ranked maps told apart by the local osu!.db, so a detector
gets a first real-audio number without the server. Two detectors side by side:

- ``p19``: the health check's own (``grade_reference_timing``, health_truth.flags);
- ``phase``: each red line's median phase of its strong attacks against the map's own
  grid, flagged when it leaves the map's common phase by FLAG_MS and FLAG_SIGMAS of its
  standard error. Nothing is refitted: the map's grid is the hypothesis.

Attacks come once per audio from the Rust core and are cached in bench/.cache/line_audit
(gitignored), so another seed costs no analysis. Prints aggregates only: the folders a
private library holds are never written out.

    .venv/Scripts/python.exe bench/line_audit.py --songs "C:\osu!\Songs" --osu-db "C:\osu!\osu!.db"
    .venv/Scripts/python.exe bench/line_audit.py ... --seed 62        # another sample
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "python"), str(HERE)]
import overtone as ta
import overtone_library as lib
import health_truth as ht
import stages

SEED = 61

CACHE = HERE / ".cache" / "line_audit"

#: The phase detector's constants. The first run (2026-11-14, seed 61) used 7 ms, 3
#: sigmas and 8 inliers; the strictest rule of a sweep on that sample (9, 6, 16) is
#: kept, and its seed-62 sample is where it was checked (docs/timeline.md).
GRID_DIV = 2          # residuals against the half-beat grid
WINDOW_SHARE = 0.3    # an attack within 30 % of a half beat of a tick is an inlier...
WINDOW_MAX_MS = 40.0  # ...and never further than 40 ms
STRONG_SHARE = 0.4    # the strongest 60 % of a span's attacks
MIN_INLIERS = 16
FLAG_MS = 9.0         # off the map's common phase by more than this...
FLAG_SIGMAS = 6.0     # ...and by more than six of its own standard errors


def phase_flags(beatmap, times, weights, duration):
    """Each red line's median phase against the map's own common phase; a line off it by
    FLAG_MS and FLAG_SIGMAS standard errors is flagged. Nothing is refitted: the map's
    own grid is the hypothesis, the attacks the test."""
    objects = [o for o in beatmap["hitobjects"] if "time" in o]
    first = min(o["time"] for o in objects) / 1000.0 if objects else 0.0
    last = min(duration, max(o.get("end_time", o["time"]) for o in objects) / 1000.0) if objects else duration
    rows = ta._beatmap_red_rows(beatmap)
    t_ms = np.asarray(times) * 1000.0
    w = np.asarray(weights)
    keep = (t_ms >= first * 1000 - 100) & (t_ms <= last * 1000 + 100)
    t_ms, w = t_ms[keep], w[keep]
    per = []
    for n, (offset, bpm, _meter) in enumerate(rows):
        end = rows[n + 1][0] if n + 1 < len(rows) else last * 1000.0
        sub = 60000.0 / bpm / GRID_DIV
        inside = (t_ms >= offset - 0.5 * sub) & (t_ms < end - 0.5 * sub)
        st, sw = t_ms[inside], w[inside]
        if st.size:
            strong = sw >= np.quantile(sw, STRONG_SHARE)
            st = st[strong]
        r = st - (offset + np.round((st - offset) / sub) * sub)
        r = r[np.abs(r) <= min(WINDOW_SHARE * sub, WINDOW_MAX_MS)]
        if r.size < MIN_INLIERS:
            per.append(None)
            continue
        med = float(np.median(r))
        se = float(1.2533 * np.std(r) / np.sqrt(r.size))
        per.append((med, se, r.size))
    graded = [p for p in per if p is not None]
    if len(graded) < 2:
        return []
    meds = np.array([p[0] for p in graded])
    counts = np.array([p[2] for p in graded], float)
    order = np.argsort(meds)
    cum = np.cumsum(counts[order])
    common = float(meds[order][np.searchsorted(cum, 0.5 * cum[-1])])
    out = []
    for n, p in enumerate(per):
        if p is None:
            continue
        d = p[0] - common
        if abs(d) > FLAG_MS and abs(d) > FLAG_SIGMAS * p[1]:
            out.append({"index": n, "kind": "actionable"})
    return out


def sample(songs: Path, osu_db: Path, audios: int):
    ranked, _ = stages.ranked_maps(osu_db)
    folders = sorted(p for p in songs.iterdir() if p.is_dir())
    random.Random(SEED).shuffle(folders)
    out = []
    for folder in folders:
        maps = [p for p in sorted(folder.glob("*.osu")) if (folder.name.casefold(), p.name.casefold()) in ranked]
        if not maps:
            continue
        by_audio = {}
        for path in maps:
            try:
                beatmap = ta.read_osu_beatmap(path)
            except (OSError, ValueError):
                continue
            if len(ta._beatmap_red_rows(beatmap)) < 2:
                continue
            name = str(beatmap.get("general", {}).get("AudioFilename") or "").strip()
            audio = next((f for f in folder.iterdir() if f.name.lower() == name.lower()), None)
            if audio:
                by_audio.setdefault(str(audio), []).append({"path": path, "file": path.name})
        for audio, ms in by_audio.items():
            out.append({"folder": folder.name, "audio": audio, "maps": ms})
            if len(out) >= audios:
                return out
    return out


def attacks(audio: str):
    """The Rust core's attacks of one audio file, cached by its path."""
    f = CACHE / (hashlib.sha1(audio.encode()).hexdigest() + ".npz")
    if f.exists():
        z = np.load(f)
        return z["t"], z["w"], float(z["d"])
    t, w, d = lib._attacks(audio, "rust")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, t=t, w=w, d=d)
    return t, w, d


def grade(songs_list, detectors):
    rng = random.Random(SEED)
    rows = {name: [] for name in detectors}
    for k, song in enumerate(songs_list):
        try:
            times, weights, duration = attacks(song["audio"])
        except Exception as e:  # noqa: BLE001
            print(f"  an audio skipped: {type(e).__name__}", flush=True)
            continue
        seen = set()
        for m in song["maps"]:
            beatmap = ta.read_osu_beatmap(m["path"])
            reds = ta._beatmap_red_rows(beatmap)
            key = tuple(reds)
            if len(reds) < 2 or key in seen:
                continue
            seen.add(key)
            target = rng.randrange(len(reds))
            for name, detector in detectors.items():
                row = {"lines": len(reds), "target": target, "intact": detector(beatmap, times, weights, duration)}
                for s in ht.SHIFTS_MS:
                    row[f"shift_{s:g}"] = detector(ht.shifted(beatmap, target, s), times, weights, duration)
                rows[name].append(row)
        if k % 10 == 9:
            print(f"  {k + 1} audios", flush=True)
    return rows


def main(argv=None) -> int:
    global SEED
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--songs", required=True, help="the osu! Songs folder")
    parser.add_argument("--osu-db", required=True, help="osu!.db, to keep ranked, approved and loved maps")
    parser.add_argument("--audios", type=int, default=80)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", help="the summary as JSON (aggregates only)")
    args = parser.parse_args(argv)
    SEED = args.seed
    songs = sample(Path(args.songs), Path(args.osu_db), args.audios)
    print(f"{len(songs)} audios, {sum(len(s['maps']) for s in songs)} ranked maps with two red lines or more", flush=True)
    report = {}
    for name, rows in grade(songs, {"p19": ht.flags, "phase": phase_flags}).items():
        report[name] = ht.summarise(rows)
        a = report[name]["actionable"]
        p = a["precision_pool_15"]
        print(f"{name}: {report[name]['maps']} maps, {report[name]['lines']} lines; intact flagged {a['false_flag_rate_intact']:.1%}; "
              f"found 5/15/30 ms {a['recall_5']:.1%} / {a['recall_15']:.1%} / {a['recall_30']:.1%}; "
              f"precision on the 15 ms pool {'-' if p is None else format(p, '.1%')}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
