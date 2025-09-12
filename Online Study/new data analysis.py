#!/usr/bin/env python3
# hri_terminal_report.py
import argparse
import math
import re
import sys
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

# Matplotlib is optional (only if --plots)
try:
    import matplotlib.pyplot as plt  # noqa: F401
    MPL_OK = True
except Exception:
    MPL_OK = False

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
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="HRI terminal report (pretty console output).")
    # 🔽 Your path is now the default; you can still override with --csv
    p.add_argument(
        "--csv", "-c",
        default=r"C:\Users\61479\OneDrive - Queensland University of Technology\EGH400-2\Online Study\EGH400 - Human Robot Interaction (Responses) - Form Responses 1.csv",
        help="Path to CSV responses (defaults to your survey file)"
    )
    p.add_argument("--plots", action="store_true", help="Also show matplotlib charts")
    p.add_argument("--width", type=int, default=100, help="Console width for formatting")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv, encoding="utf-8-sig")
    except Exception as e:
        print("Could not read CSV:", e, file=sys.stderr)
        sys.exit(1)

    width = args.width

    # --- Columns used (adjust names here if your CSV uses different labels) ---
    COL_AGE = "Age"
    COL_COMFORT = "comfort level"
    COL_TRUST = "trusting a robot"
    COL_EXPO = "Have you ever seen or interacted with a robot in a public space? "
    COL_DEGREE = "Degree"
    COL_FEATURES = "What would make you feel more comfortable interacting with a robot guide? "

    # --- Comfort normalisation 1..6 -> 0..1 ---
    comfort_raw = pd.to_numeric(df[COL_COMFORT], errors="coerce")
    if comfort_raw.notna().sum() == 0:
        print("No numeric comfort values found.")
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
    df["exposure"] = df[COL_EXPO].map(expo_map)

    # --- Degree buckets ---
    df["degree_cat"] = df[COL_DEGREE].apply(canonical_degree)

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

    # -------------- Optional plots --------------
    if args.plots:
        if not MPL_OK:
            print("\n(matplotlib not available; cannot show plots)", file=sys.stderr)
            return
        # 1) Trust distribution
        df[COL_TRUST].value_counts().reindex(["Yes", "Maybe", "No"]).plot(kind="bar", title="Trusting a Robot (Yes/Maybe/No)")
        plt.ylabel("Count"); plt.tight_layout(); plt.show()

        # 2) Comfort vs Age scatter + fit
        mask = df["age_numeric"].notna() & df["comfort_norm"].notna()
        if mask.sum() >= 2:
            a, b = np.polyfit(df.loc[mask, "age_numeric"], df.loc[mask, "comfort_norm"], 1)
            xs = np.linspace(df.loc[mask, "age_numeric"].min(), df.loc[mask, "age_numeric"].max(), 200)
            plt.scatter(df.loc[mask, "age_numeric"], df.loc[mask, "comfort_norm"], alpha=0.9)
            plt.plot(xs, a * xs + b, linewidth=2)
            plt.title("Comfort vs Age (0–1)")
            plt.xlabel("Age"); plt.ylabel("Comfort"); plt.tight_layout(); plt.show()

        # 3) Comfort by Age Band (mean ± SE)
        groups = df.groupby("age_band")["comfort_norm"]
        means = groups.mean()
        sems = groups.std() / np.sqrt(groups.count())
        plt.bar(range(len(means)), means.values, yerr=sems.values, capsize=4)
        plt.xticks(range(len(means)), means.index)
        plt.ylim(0, 1); plt.title("Mean Comfort by Age Band (±SE)")
        plt.ylabel("Comfort"); plt.tight_layout(); plt.show()

        # 4) Exposure vs trust/comfort
        tmp = df[["exposure", "trust_numeric", "comfort_norm"]].dropna()
        if not tmp.empty:
            tmp.groupby("exposure")["trust_numeric"].mean().reindex(["Interacted", "Saw only", "No exposure"]).plot(kind="bar", title="Mean Trust by Exposure")
            plt.ylabel("Trust (0–1)"); plt.tight_layout(); plt.show()

            tmp.groupby("exposure")["comfort_norm"].mean().reindex(["Interacted", "Saw only", "No exposure"]).plot(kind="bar", title="Mean Comfort by Exposure")
            plt.ylabel("Comfort (0–1)"); plt.tight_layout(); plt.show()

        # 5) Comfort by Discipline
        deg_groups = df[df["degree_cat"] != ""].groupby("degree_cat")["comfort_norm"]
        if not deg_groups.count().empty:
            deg_means = deg_groups.mean().sort_values(ascending=False)
            deg_sems = (deg_groups.std() / np.sqrt(deg_groups.count())).reindex(deg_means.index)
            plt.barh(range(len(deg_means)), deg_means.values, xerr=deg_sems.values, capsize=4)
            plt.yticks(range(len(deg_means)), deg_means.index)
            plt.xlabel("Comfort (0–1)"); plt.title("Comfort by Discipline (±SE)")
            plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
