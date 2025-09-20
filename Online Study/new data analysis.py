#!/usr/bin/env python3
# hri_terminal_report.py
import argparse
import math
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

# Matplotlib (robust import; allow headless)
MPL_OK = True
def ensure_matplotlib_backend(no_show: bool):
    global MPL_OK, plt
    try:
        if no_show:
            import matplotlib
            matplotlib.use("Agg")   # headless
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        MPL_OK = False
        plt = None

# ---------------------------
# Pretty terminal helpers
# ---------------------------
def hr(char="─", n=80):
    return char * n

def section(title, width=80):
    bar = hr("═", width)
    return f"{bar}\n{title}\n{bar}"

def fmt_pct(x, denom, digits=1):
    if denom and denom > 0:
        return f"{(100.0 * x / denom):.{digits}f}%"
    return "—"

def fmt_num(x, digits=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"

def print_table(rows, headers=None, width=80):
    if not rows:
        print("(no data)")
        return
    cols = len(rows[0]) if rows else 0
    if headers and len(headers) != cols:
        headers = None
    data = [list(map(lambda z: "" if z is None else str(z), r)) for r in rows]
    widths = [len(str(h)) for h in headers] if headers else [0] * cols
    for r in data:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    total = sum(widths) + 3 * (cols - 1)
    if total > width:
        overflow = total - width
        widths[-1] = max(6, widths[-1] - overflow)
    if headers:
        line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
        print(line)
        print(hr(n=min(width, len(line))))
    for r in data:
        cells = []
        for i, cell in enumerate(r):
            c = str(cell)
            if len(c) > widths[i]:
                c = c[: max(0, widths[i] - 1)] + "…"
            cells.append(c.ljust(widths[i]))
        print(" | ".join(cells))

# ---------------------------
# Domain helpers
# ---------------------------
def parse_age(val: str) -> float:
    s = str(val).strip()
    if re.match(r"^\d+$", s):
        v = float(s)
        if 16 <= v <= 100:
            return v
        return np.nan
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        v = (a + b) / 2.0
        return v if 16 <= v <= 100 else np.nan
    m = re.match(r"^(\d+)\s*\+$", s)
    if m:
        v = float(m.group(1))
        return v if 16 <= v <= 100 else np.nan
    return np.nan

def canonical_degree(x: str) -> str:
    s = str(x).lower().strip()
    if not s or s in {"nan", "none", ""}:
        return ""
    if "mechatron" in s: return "Mechatronics"
    if "mechanical" in s: return "Mechanical"
    if "electrical" in s or "ece" in s: return "Electrical/Computer"
    if "software" in s or "information technology" in s or s == "it" or "computer science" in s: return "Software/IT"
    if "civil" in s: return "Civil"
    if "chemical" in s or "process" in s: return "Chemical/Process"
    if "biomed" in s: return "Biomedical"
    if "science" in s: return "Science"
    if "business" in s or "commerce" in s: return "Business/Commerce"
    return "Other"

def compute_spearman(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    rx = x[mask].rank()
    ry = y[mask].rank()
    return rx.corr(ry)

# ---------------------------
# Plot helpers (one figure per chart; no custom colors)
# ---------------------------
def maybe_save(showing: bool, save_dir: str | None, filename: str):
    if save_dir:
        import os
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[saved] {path}")
    if showing:
        plt.show()
    plt.close()

def plot_trust_bars(df, col_trust, save_dir, showing):
    counts = df[col_trust].value_counts()
    labels = ["Yes", "Maybe", "No"]
    y = [int(counts.get(k, 0)) for k in labels]
    fig = plt.figure()
    plt.bar(range(len(labels)), y)
    plt.xticks(range(len(labels)), labels)
    plt.title("Trusting a Robot (Yes/Maybe/No)")
    plt.ylabel("Count")
    plt.tight_layout()
    maybe_save(showing, save_dir, "01_trust_distribution.png")

def plot_comfort_vs_age(df, save_dir, showing):
    mask = df["age_numeric"].notna() & df["comfort_norm"].notna()
    if mask.sum() < 2:
        return
    x = df.loc[mask, "age_numeric"].to_numpy()
    y = df.loc[mask, "comfort_norm"].to_numpy()
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    fig = plt.figure()
    plt.scatter(x, y, alpha=0.9)
    plt.plot(xs, a * xs + b, linewidth=2)
    plt.title("Comfort vs Age (0–1)")
    plt.xlabel("Age")
    plt.ylabel("Comfort")
    plt.tight_layout()
    maybe_save(showing, save_dir, "02_comfort_vs_age.png")

def plot_comfort_by_ageband(df, save_dir, showing):
    groups = df.groupby("age_band")["comfort_norm"]
    means = groups.mean()
    if means.empty:
        return
    sems = groups.std() / np.sqrt(groups.count())
    idx = np.arange(len(means))
    fig = plt.figure()
    plt.bar(idx, means.values, yerr=sems.values, capsize=4)
    plt.xticks(idx, means.index.astype(str))
    plt.ylim(0, 1)
    plt.title("Mean Comfort by Age Band (±SE)")
    plt.ylabel("Comfort")
    plt.tight_layout()
    maybe_save(showing, save_dir, "03_comfort_by_age_band.png")

def plot_exposure_means(df, save_dir, showing):
    tmp = df[["exposure", "trust_numeric", "comfort_norm"]].dropna()
    if tmp.empty:
        return
    order = ["Interacted", "Saw only", "No exposure"]
    # Trust
    t_means = tmp.groupby("exposure")["trust_numeric"].mean().reindex(order)
    fig = plt.figure()
    plt.bar(range(len(order)), t_means.values)
    plt.xticks(range(len(order)), order, rotation=15)
    plt.ylabel("Trust (0–1)")
    plt.title("Mean Trust by Exposure")
    plt.tight_layout()
    maybe_save(showing, save_dir, "04_mean_trust_by_exposure.png")
    # Comfort
    c_means = tmp.groupby("exposure")["comfort_norm"].mean().reindex(order)
    fig = plt.figure()
    plt.bar(range(len(order)), c_means.values)
    plt.xticks(range(len(order)), order, rotation=15)
    plt.ylabel("Comfort (0–1)")
    plt.title("Mean Comfort by Exposure")
    plt.tight_layout()
    maybe_save(showing, save_dir, "05_mean_comfort_by_exposure.png")

def plot_comfort_by_discipline(df, save_dir, showing):
    sub = df[df["degree_cat"] != ""]
    if sub.empty:
        return
    groups = sub.groupby("degree_cat")["comfort_norm"]
    if groups.count().empty:
        return
    means = groups.mean().sort_values(ascending=False)
    sems = (groups.std() / np.sqrt(groups.count())).reindex(means.index)
    idx = np.arange(len(means))
    fig = plt.figure()
    plt.barh(idx, means.values, xerr=sems.values, capsize=4)
    plt.yticks(idx, means.index)
    plt.xlabel("Comfort (0–1)")
    plt.title("Comfort by Discipline (±SE)")
    plt.tight_layout()
    maybe_save(showing, save_dir, "06_comfort_by_discipline.png")

# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="HRI terminal report (pretty console output).")
    p.add_argument(
        "--csv", "-c",
        default=r"C:\Users\61479\OneDrive - Queensland University of Technology\EGH400-2\Online Study\EGH400 - Human Robot Interaction (Responses) - Form Responses 1.csv",
        help="Path to CSV responses (defaults to your survey file)"
    )
    # Plots are ON by default now
    p.add_argument("--no-plots", action="store_true", help="Disable chart generation")
    p.add_argument("--no-show", action="store_true", help="Do not display figures (useful on servers)")
    p.add_argument("--save-dir", type=str, default=None, help="Directory to save PNGs (optional)")
    p.add_argument("--width", type=int, default=100, help="Console width for formatting")
    args = p.parse_args()

    # Matplotlib init
    ensure_matplotlib_backend(no_show=args.no_show)

    try:
        df = pd.read_csv(args.csv, encoding="utf-8-sig")
    except Exception as e:
        print("Could not read CSV:", e, file=sys.stderr)
        sys.exit(1)

    width = args.width

    # --- Column names (adjust here if your CSV differs) ---
    COL_AGE = "Age"
    COL_COMFORT = "comfort level"
    COL_TRUST = "trusting a robot"
    COL_EXPO = "Have you ever seen or interacted with a robot in a public space? "
    COL_DEGREE = "Degree"
    COL_FEATURES = "What would make you feel more comfortable interacting with a robot guide? "

    missing = [c for c in [COL_AGE, COL_COMFORT, COL_TRUST] if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # --- Comfort normalisation 1..6 -> 0..1 ---
    comfort_raw = pd.to_numeric(df[COL_COMFORT], errors="coerce")
    if comfort_raw.notna().sum() == 0:
        print("No numeric comfort values found.", file=sys.stderr)
        sys.exit(1)
    mn, mx = float(comfort_raw.min()), float(comfort_raw.max())
    df["comfort_norm"] = (comfort_raw - mn) / (mx - mn) if mx > mn else np.nan

    # --- Age parsing ---
    df["age_numeric"] = df[COL_AGE].apply(parse_age)
    bins = [0, 24, 34, 49, 64, 200]
    labels = ["<25", "25–34", "35–49", "50–64", "65+"]
    df["age_band"] = pd.cut(df["age_numeric"], bins=bins, labels=labels, include_lowest=True)

    # --- Trust mapping ---
    trust_map = {"Yes": 1.0, "Maybe": 0.5, "No": 0.0}
    df["trust_numeric"] = df[COL_TRUST].map(trust_map)

    # --- Exposure mapping ---
    expo_map = {
        "Yes - Interacted with one": "Interacted",
        "Yes - saw one but did not interact": "Saw only",
        "No": "No exposure",
    }
    if COL_EXPO in df.columns:
        df["exposure"] = df[COL_EXPO].map(expo_map)
    else:
        df["exposure"] = np.nan

    # --- Degree buckets ---
    if COL_DEGREE in df.columns:
        df["degree_cat"] = df[COL_DEGREE].apply(canonical_degree)
    else:
        df["degree_cat"] = ""

    # ---------------------------
    # Terminal report
    # ---------------------------
    n = len(df)
    print(section("HRI Terminal Report", width))

    # Overview
    print("\n" + hr(n=width))
    print("Overview")
    print(hr(n=width))
    trust_counts = df[COL_TRUST].value_counts()
    yes = int(trust_counts.get("Yes", 0))
    maybe = int(trust_counts.get("Maybe", 0))
    no = int(trust_counts.get("No", 0))

    rows = [
        ["Total responses", n],
        ["Comfort (mean, 0–1)", fmt_num(df["comfort_norm"].mean(), 3)],
        ["Comfort (median, 0–1)", fmt_num(df["comfort_norm"].median(), 3)],
        ["Trust: Yes", f"{yes}  ({fmt_pct(yes, n)})"],
        ["Trust: Maybe", f"{maybe}  ({fmt_pct(maybe, n)})"],
        ["Trust: No", f"{no}  ({fmt_pct(no, n)})"],
    ]
    print_table(rows, headers=["Metric", "Value"], width=width)

    # Comfort vs Age
    print("\n" + hr(n=width))
    print("Comfort vs Age")
    print(hr(n=width))
    pearson = df["age_numeric"].corr(df["comfort_norm"])
    spearman = compute_spearman(df["age_numeric"], df["comfort_norm"])
    rows = [
        ["Pearson r", fmt_num(pearson, 3)],
        ["Spearman ρ", fmt_num(spearman, 3)],
        ["N (valid pairs)", int((df["age_numeric"].notna() & df["comfort_norm"].notna()).sum())],
    ]
    print_table(rows, headers=["Statistic", "Value"], width=width)

    # Comfort by Age Band
    print("\n" + hr(n=width))
    print("Comfort by Age Band (mean ± SE, N)")
    print(hr(n=width))
    grp = df.groupby("age_band")["comfort_norm"]
    means = grp.mean()
    counts = grp.count()
    sems = grp.std() / np.sqrt(counts.replace(0, np.nan))
    age_rows = []
    for band in labels:
        m = means.get(band, np.nan)
        se = sems.get(band, np.nan)
        c = int(counts.get(band, 0))
        age_rows.append([band, fmt_num(m, 3), fmt_num(se, 3), c])
    print_table(age_rows, headers=["Age band", "Mean", "SE", "N"], width=width)

    # Exposure vs Trust & Comfort
    print("\n" + hr(n=width))
    print("Exposure vs Trust/Comfort (means)")
    print(hr(n=width))
    tmp = df[["exposure", "trust_numeric", "comfort_norm"]].dropna()
    expo_order = ["Interacted", "Saw only", "No exposure"]
    rows = []
    for g in expo_order:
        sub = tmp[tmp["exposure"] == g]
        if not sub.empty:
            rows.append([g, fmt_num(sub["trust_numeric"].mean(), 3), fmt_num(sub["comfort_norm"].mean(), 3), len(sub)])
    print_table(rows, headers=["Group", "Trust (0–1)", "Comfort (0–1)", "N"], width=width)

    # Comfort by Discipline (top by N)
    print("\n" + hr(n=width))
    print("Comfort by Discipline (mean ± SE, N)")
    print(hr(n=width))
    deg = df[df["degree_cat"] != ""].groupby("degree_cat")["comfort_norm"]
    if not deg.count().empty:
        means = deg.mean().sort_values(ascending=False)
        sems = (deg.std() / np.sqrt(deg.count())).reindex(means.index)
        ns = deg.count().reindex(means.index)
        rows = []
        for k in means.index:
            rows.append([k, fmt_num(means[k], 3), fmt_num(sems[k], 3), int(ns[k])])
        print_table(rows, headers=["Discipline", "Mean", "SE", "N"], width=width)
    else:
        print("(no discipline data)")

    # Top Comfort Features
    print("\n" + hr(n=width))
    print("Top Comfort Features (counts and % of all respondents)")
    print(hr(n=width))
    counter = Counter()
    if COL_FEATURES in df.columns:
        for cell in df[COL_FEATURES].dropna():
            for part in re.split(r"[;,/]| and ", str(cell)):
                p = part.strip().lower()
                if p:
                    counter[p] += 1
    feat_rows = []
    for lab, c in counter.most_common(15):
        feat_rows.append([lab.title(), c, fmt_pct(c, n)])
    if feat_rows:
        print_table(feat_rows, headers=["Feature", "Count", "% of all"], width=width)
    else:
        print("(no feature selections found)")

    # -------------- Plots --------------
    if not args.no_plots:
        if not MPL_OK:
            print("\n(matplotlib not available; cannot create plots)", file=sys.stderr)
        else:
            showing = not args.no_show
            plot_trust_bars(df, COL_TRUST, args.save_dir, showing)
            plot_comfort_vs_age(df, args.save_dir, showing)
            plot_comfort_by_ageband(df, args.save_dir, showing)
            plot_exposure_means(df, args.save_dir, showing)
            plot_comfort_by_discipline(df, args.save_dir, showing)

if __name__ == "__main__":
    main()
