"""
Preflight for the Microburbs Hackathon, Task 4.

Run this TONIGHT on the exact machine you will use tomorrow. It checks every
dependency, exercises the real model path end to end on synthetic data, times
it, and verifies the things that disqualify entries (repo hygiene, video
compression, the 95 MB limit).

    python preflight.py

Exit code 0 = ready. Anything else = fix it tonight, not at 10:30 tomorrow.
No network and no API key needed.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def log(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    mark = {PASS: "  ok ", WARN: " warn", FAIL: "FAIL "}[status]
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


# ---------------------------------------------------------------- 1. runtime
def check_python() -> None:
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro} ({platform.system()} {platform.machine()})"
    if v < (3, 10):
        log(FAIL, "Python version", f"{s} — need 3.10+")
    else:
        log(PASS, "Python version", s)


def check_packages() -> bool:
    need = {
        "pandas": "pandas", "numpy": "numpy", "lightgbm": "lightgbm",
        "sklearn": "scikit-learn", "optuna": "optuna", "requests": "requests",
    }
    missing = []
    for mod, pkg in need.items():
        try:
            m = importlib.import_module(mod)
            log(PASS, f"import {mod}", getattr(m, "__version__", "?"))
        except Exception as e:
            missing.append(pkg)
            log(FAIL, f"import {mod}", f"{type(e).__name__}: {e}")
    if missing:
        print(f"\n    FIX:  pip install {' '.join(missing)}\n")
        return False
    return True


def check_resources() -> None:
    cpus = os.cpu_count() or 1
    log(PASS if cpus >= 2 else WARN, "CPU cores", str(cpus))

    free_gb = shutil.disk_usage(".").free / 1e9
    log(PASS if free_gb > 3 else FAIL, "Free disk", f"{free_gb:.1f} GB (need ~2 GB)")

    try:  # best effort, not on every platform
        import ctypes
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
            gb = kb / 1e6
        else:
            gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        log(PASS if gb >= 7 else WARN, "RAM", f"{gb:.1f} GB" + ("" if gb >= 7 else " — close other apps tomorrow"))
    except Exception:
        log(WARN, "RAM", "could not detect")


# ------------------------------------------------------------------ 2. clock
def check_clock() -> None:
    now = datetime.now(ZoneInfo("Australia/Adelaide"))
    syd = now.astimezone(ZoneInfo("Australia/Sydney"))
    delta = (syd.utcoffset() - now.utcoffset()).total_seconds() / 60
    log(PASS, "Local time", now.strftime("%a %d %b %H:%M %Z"))
    log(WARN if delta else PASS, "Sydney offset",
        f"Sydney is {int(delta)} min ahead — key 10:30 your time, uploads close 14:30 your time")


# -------------------------------------------------------------- 3. the model
def check_pipeline() -> None:
    """Exercise the real training path on synthetic data, and time it."""
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 40_000
    df = pd.DataFrame({
        "suburb": pd.Categorical(rng.choice([f"S{i}" for i in range(33)], n)),
        "bedrooms": rng.integers(0, 4, n),
        "bathrooms": rng.integers(1, 3, n),
        "car_space": rng.integers(0, 2, n).astype(float),
        "floor_level": rng.integers(1, 30, n).astype(float),
        "lat": -33.88 + rng.normal(0, .02, n),
    })
    df["rent"] = (420 + 230 * df.bedrooms + 95 * df.car_space
                  + 4 * df.floor_level + rng.normal(0, 85, n))

    feats = ["suburb", "bedrooms", "bathrooms", "car_space", "floor_level", "lat"]
    tr, va = df.iloc[:34_000], df.iloc[34_000:]

    # the exact objective the real pipeline uses
    t0 = time.perf_counter()
    params = dict(objective="regression_l1", metric="l1", learning_rate=.05,
                  num_leaves=63, verbosity=-1, n_jobs=-1)
    dtr = lgb.Dataset(tr[feats], tr.rent, categorical_feature=["suburb"])
    dva = lgb.Dataset(va[feats], va.rent, reference=dtr)
    model = lgb.train(params, dtr, 600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    fit_s = time.perf_counter() - t0

    err = np.abs(va.rent.to_numpy() - model.predict(va[feats]))
    med = float(np.median(err))
    log(PASS if med < 120 else WARN, "LightGBM L1 trains",
        f"{fit_s:.1f}s, median abs error ${med:.0f} on synthetic")

    if fit_s > 90:
        log(WARN, "Training speed", f"{fit_s:.0f}s is slow — cut Optuna trials tomorrow")

    # the constraint that broke the first draft: monotone + L1 is rejected
    print("       (next line prints a LightGBM 'Fatal' — that is the expected result)")
    try:
        lgb.train({**params, "monotone_constraints": [0, 0, 0, 1, 0, 0]}, dtr, 5)
        log(WARN, "Monotone + L1", "accepted here — pipeline handles it either way")
    except Exception:
        log(PASS, "Monotone + L1 rejected", "expected; premium is enforced post-hoc")

    # Optuna, one trial, so the import path is proven
    t0 = time.perf_counter()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="minimize")
    st.optimize(lambda t: abs(t.suggest_float("x", -1, 1)), n_trials=5)
    log(PASS, "Optuna runs", f"5 trials in {time.perf_counter() - t0:.1f}s")

    # 65,777-row CSV write, the real submission shape
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        sub = pd.DataFrame({
            "gnaf_id": [f"GANSW{700000000+i}" for i in range(65_777)],
            "rent_with_car_space": np.round(rng.uniform(400, 1500, 65_777) * 2) / 2,
            "rent_without_car_space": np.round(rng.uniform(300, 1400, 65_777) * 2) / 2,
        })
        p = Path(d) / "t.csv"
        sub.to_csv(p, index=False)
        size_mb = p.stat().st_size / 1e6
    log(PASS, "65,777-row CSV write", f"{time.perf_counter() - t0:.1f}s, {size_mb:.1f} MB")


# ------------------------------------------------------------------ 4. tools
def check_git() -> None:
    if not shutil.which("git"):
        log(FAIL, "git installed", "install git tonight")
        return
    log(PASS, "git installed", run(["git", "--version"]).stdout.strip())

    if not Path(".git").exists():
        log(WARN, "git repo here", "not a repo yet — git init and push to a PUBLIC GitHub repo")
        return

    r = run(["git", "status", "--porcelain"])
    dirty = [l for l in r.stdout.splitlines() if l.strip()]
    log(PASS if not dirty else WARN, "working tree",
        "clean" if not dirty else f"{len(dirty)} uncommitted change(s)")

    # the disqualifier: supplied data or the key tracked in git
    tracked = run(["git", "ls-files"]).stdout.splitlines()
    bad = [f for f in tracked
           if f.endswith(".env") or f.startswith("data/")
           or (f.endswith(".csv") and "rents_" not in f)]
    log(PASS if not bad else FAIL, "no secrets/data tracked",
        "clean" if not bad else f"REMOVE: {bad[:5]}")

    nb = [f for f in tracked if f.endswith(".ipynb")]
    if nb:
        log(WARN, "notebooks tracked",
            f"{len(nb)} — clear outputs before committing (nbconvert --clear-output --inplace)")

    gi = Path(".gitignore")
    if gi.exists():
        txt = gi.read_text()
        need = [p for p in (".env", "data/", "out/") if p not in txt]
        log(PASS if not need else WARN, ".gitignore covers key paths",
            "ok" if not need else f"missing {need}")
    else:
        log(WARN, ".gitignore present", "none found")


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        log(WARN, "ffmpeg installed",
            "needed to get videos under 95 MB — brew/apt/choco install ffmpeg")
        return
    ver = run(["ffmpeg", "-version"]).stdout.splitlines()[0][:60]
    log(PASS, "ffmpeg installed", ver)

    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw.mp4", Path(d) / "out.mp4"
        mk = run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=3",
                  "-pix_fmt", "yuv420p", str(raw)])
        if mk.returncode or not raw.exists():
            log(WARN, "ffmpeg encode test", "could not generate test clip")
            return
        cp = run(["ffmpeg", "-y", "-i", str(raw), "-vcodec", "libx264",
                  "-crf", "28", "-preset", "fast", str(out)])
        if cp.returncode or not out.exists():
            log(FAIL, "ffmpeg libx264", "x264 encoder missing — install a full ffmpeg build")
        else:
            log(PASS, "ffmpeg x264 compress",
                f"{raw.stat().st_size/1e6:.1f} MB -> {out.stat().st_size/1e6:.2f} MB")


def check_scaffold() -> None:
    want = ["pipeline.py", "make_synthetic.py", "make_report.py", "README.md", "requirements.txt"]
    missing = [f for f in want if not Path(f).exists()]
    log(PASS if not missing else WARN, "scaffold files present",
        "all" if not missing else f"missing {missing}")


# ------------------------------------------------------------------- summary
def main() -> int:
    print("\nMICROBURBS TASK 4 — PREFLIGHT\n" + "=" * 60)
    print("\n-- runtime")
    check_python()
    ok = check_packages()
    check_resources()
    print("\n-- clock")
    check_clock()
    print("\n-- project")
    check_scaffold()
    check_git()
    print("\n-- video")
    check_ffmpeg()
    if ok:
        print("\n-- model path (this is the one that matters)")
        try:
            check_pipeline()
        except Exception as e:
            log(FAIL, "model path", f"{type(e).__name__}: {e}")
    else:
        log(FAIL, "model path", "skipped — install the missing packages first")

    f = [r for r in results if r[0] == FAIL]
    w = [r for r in results if r[0] == WARN]
    print("\n" + "=" * 60)
    print(f"{len(results)-len(f)-len(w)} passed, {len(w)} warnings, {len(f)} failures")
    if f:
        print("\nFIX TONIGHT:")
        for _, n, d in f:
            print(f"  - {n}: {d}")
        return 1
    if w:
        print("\nWorth a look:")
        for _, n, d in w:
            print(f"  - {n}: {d}")
    print("\nReady. Key lands 10:30 your time; uploads close 14:30 your time.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
