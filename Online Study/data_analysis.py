#!/usr/bin/env python3
"""
HRI survey analysis from CSV
- Comfort level vs age with R^2 (linear regression)
- Comfort/trust vs prior experience
- Analysis by disability-related items (if present)
- Multi-select "what would make you more comfortable?" -> counts & % of all respondents + bar chart
- Distance stats (mean/median, etc.) + histogram
- Comfort being approached: overall and by age

Usage:
  python hri_from_csv.py --csv "path/to/responses.csv" --out "outputs"

Notes:
- Column names are matched heuristically; override names via CLI if needed.
- Charts are saved as PNGs; summaries as CSVs and a summary.txt.
"""

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column whose lowercase name contains any of the candidate snippets (lowercased)."""
    low = {c.lower(): c for c in df.columns}
    for c in df.columns:
        cl = c.lower()
        for snippet in candidates:
            if snippet in cl:
                return c
    return None

def map_yes_no_maybe(series: pd.Series) -> pd.Series:
    """Map common yes/no/maybe/depends strings to numeric: yes=1, maybe/depends=0.5, no=0."""
    s = series.astype(str).str.strip().str.lower()
    def classify(x: str) -> float:
        if "yes" in x:
            return 1.0
        if "no" in x:
            return 0.0
        if "maybe" in x or "depend" in x or "unsure" in x or "not sure" in x:
            return 0.5
        return np.nan
    return s.map(classify)

def map_yes_no(series: pd.Series) -> pd.Series:
    """Map yes/no to 1/0; other is NaN."""
    s = series.astype(str).str.strip().str.lower()
    return s.map(lambda x: 1.0 if "yes" in x else (0.0 if "no" in x else np.nan))

def parse_age_midpoint(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Convert age strings like '18-22', '57+', '23-27' or numeric ages to:
    - numeric midpoint (float) for regression
    - band label (str) for group summaries
    """
    raw = series.astype(str).str.strip()
    # Numeric ages
    as_num = pd.to_numeric(raw, errors="coerce")
    bands = pd.Series(index=series.index, dtype=object)
    mid = pd.Series(index=series.index, dtype=float)

    for i, v in raw.items():
        s = v
        # Numeric
        if pd.notna(as_num.loc[i]):
            age = float(as_num.loc[i])
            mid.loc[i] = age
            # Create bands ~5-year chunks by default
            if age < 25: bands.loc[i] = "<25"
            elif age < 35: bands.loc[i] = "25-34"
            elif age < 50: bands.loc[i] = "35-49"
            elif age < 65: bands.loc[i] = "50-64"
            else: bands.loc[i] = "65+"
            continue

        # Range 'a-b'
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
        if m:
            a = float(m.group(1)); b = float(m.group(2))
            mid.loc[i] = (a + b) / 2.0
            bands.loc[i] = f"{int(a)}-{int(b)}"
            continue

        # 'a+'
        m2 = re.match(r"^\s*(\d+)\s*\+\s*$", s)
        if m2:
            a = float(m2.group(1))
            mid.loc[i] = a
            bands.loc[i] = f"{int(a)}+"
            continue

        # Fallback: unknown
        mid.loc[i] = np.nan
        bands.loc[i] = s if s else np.nan

    return mid, bands

def to_numeric_distance(series: pd.Series) -> pd.Series:
    """Extract numeric value from strings like '0.5 m' -> 0.5."""
    s = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def r2_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Simple linear regression y = a*x + b via polyfit; return (a, b, R^2).
    """
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2

def save_bar(values: pd.Series, title: str, xlabel: str, ylabel: str, path: str):
    plt.figure()
    values.plot(kind="bar", rot=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def save_hist(series: pd.Series, bins: int, title: str, xlabel: str, ylabel: str, path: str):
    plt.figure()
    series.dropna().plot(kind="hist", bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def save_scatter_with_fit(x: pd.Series, y: pd.Series, a: float, b: float, title: str, xlabel: str, ylabel: str, path: str):
    plt.figure()
    plt.scatter(x, y)
    # Regression line
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
    ys = a * xs + b
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------
# Main analysis
# ---------------------------

def main(args):
    os.makedirs(args.out, exist_ok=True)

    # Read CSV
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    # Drop fully empty rows/cols
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    n_resp = len(df)

    # Heuristic column detection (override via CLI switches if provided)
    age_col = args.age or find_col(df, ["what is your age"])
    trust_col = args.trust or find_col(df, ["trust a robot to guide you"])
    comfort_public_col = args.comfort_public or find_col(df, ["comfortable if a robot approached", "public setting"])
    distance_col = args.distance or find_col(df, ["closest distance", "comfortable standing next to a robot", "distance"])
    prior_exp_col = args.prior or find_col(df, ["have you interacted with a robot before"])
    multi_col = args.multi or find_col(df, ["what would make you feel more comfortable", "select all"])

    # Disability-like columns (we'll scan for several)
    disability_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["disab", "impair", "access", "mobility", "hearing", "vision", "neurodiv", "condition"]):
            disability_cols.append(c)

    # --- Outputs registry ---
    summary_lines = []
    def add(line):
        print(line)
        summary_lines.append(line)

    add(f"Total respondents: {n_resp}")
    add("Detected columns:")
    add(f"  Age: {age_col}")
    add(f"  Trust (museum guide): {trust_col}")
    add(f"  Comfort when approached (public): {comfort_public_col}")
    add(f"  Distance: {distance_col}")
    add(f"  Prior experience: {prior_exp_col}")
    add(f"  Multi-select comfort features: {multi_col}")
    add(f"  Disability-related: {disability_cols if disability_cols else 'None detected'}")

    # ---------------------------
    # 1) Comfort level based on age with R^2  (using comfort_public)
    # ---------------------------
    if age_col and comfort_public_col:
        age_mid, age_band = parse_age_midpoint(df[age_col])
        comfort_score = map_yes_no_maybe(df[comfort_public_col])  # 1, 0.5, 0
        reg = pd.DataFrame({"age_mid": age_mid, "comfort": comfort_score}).dropna()
        if not reg.empty and reg["age_mid"].nunique() > 1:
            a, b, r2 = r2_linear(reg["age_mid"].values, reg["comfort"].values)
            add(f"[Comfort vs Age] R^2 = {r2:.3f} (y = {a:.4f}*age + {b:.4f}), N={len(reg)}")
            # Scatter + fit
            save_scatter_with_fit(reg["age_mid"], reg["comfort"], a, b,
                                  "Comfort if Approached vs Age (1=yes, 0.5=maybe, 0=no)",
                                  "Age (midpoint)", "Comfort score",
                                  os.path.join(args.out, "comfort_vs_age_regression.png"))
        else:
            add("[Comfort vs Age] Not enough variation to compute regression.")
        # By age band: Yes/Maybe/No %
        band_score = pd.DataFrame({"age_band": age_band, "comfort": comfort_score}).dropna()
        if not band_score.empty:
            grp = band_score.groupby("age_band")["comfort"]
            out = pd.DataFrame({
                "N": grp.size(),
                "Yes %": grp.apply(lambda s: (s==1).mean()*100),
                "Maybe %": grp.apply(lambda s: (s==0.5).mean()*100),
                "No %": grp.apply(lambda s: (s==0).mean()*100),
                "Mean comfort": grp.mean()
            }).round(1).reset_index()
            out.to_csv(os.path.join(args.out, "comfort_by_age_band.csv"), index=False)
    else:
        add("[Comfort vs Age] Missing age or comfort_public column; skipped.")

    # ---------------------------
    # 2) Analysis based on any disabilities (group comfort & trust if available)
    # ---------------------------
    if disability_cols:
        rows = []
        for c in disability_cols:
            # Try a simple yes/no mapping; everything else -> Other/Unknown
            m = df[c].astype(str).str.strip().str.lower()
            group = np.where(m.str.contains("yes"), "Disability=Yes",
                    np.where(m.str.contains("no"), "Disability=No", "Disability=Other/Unknown"))
            group = pd.Series(group, index=df.index)

            def summarize(target_col: Optional[str], label: str):
                if not target_col:
                    return
                score = map_yes_no_maybe(df[target_col])
                tmp = pd.DataFrame({"g": group, "s": score}).dropna()
                if tmp.empty: return
                g = tmp.groupby("g")["s"]
                rows.append({
                    "Column": c, "Target": label,
                    "Group": "Disability=Yes",   "N": int((tmp["g"]=="Disability=Yes").sum()),
                    "Yes %": round((tmp[(tmp["g"]=="Disability=Yes") & (tmp["s"]==1)].shape[0]/max(1,(tmp["g"]=="Disability=Yes").sum()))*100,1),
                    "Maybe %": round((tmp[(tmp["g"]=="Disability=Yes") & (tmp["s"]==0.5)].shape[0]/max(1,(tmp["g"]=="Disability=Yes").sum()))*100,1),
                    "No %": round((tmp[(tmp["g"]=="Disability=Yes") & (tmp["s"]==0)].shape[0]/max(1,(tmp["g"]=="Disability=Yes").sum()))*100,1),
                    "Mean": round(tmp.loc[tmp["g"]=="Disability=Yes","s"].mean(),3)
                })
                rows.append({
                    "Column": c, "Target": label,
                    "Group": "Disability=No",    "N": int((tmp["g"]=="Disability=No").sum()),
                    "Yes %": round((tmp[(tmp["g"]=="Disability=No") & (tmp["s"]==1)].shape[0]/max(1,(tmp["g"]=="Disability=No").sum()))*100,1),
                    "Maybe %": round((tmp[(tmp["g"]=="Disability=No") & (tmp["s"]==0.5)].shape[0]/max(1,(tmp["g"]=="Disability=No").sum()))*100,1),
                    "No %": round((tmp[(tmp["g"]=="Disability=No") & (tmp["s"]==0)].shape[0]/max(1,(tmp["g"]=="Disability=No").sum()))*100,1),
                    "Mean": round(tmp.loc[tmp["g"]=="Disability=No","s"].mean(),3)
                })
                rows.append({
                    "Column": c, "Target": label,
                    "Group": "Disability=Other/Unknown", "N": int((tmp["g"]=="Disability=Other/Unknown").sum()),
                    "Yes %": round((tmp[(tmp["g"]=="Disability=Other/Unknown") & (tmp["s"]==1)].shape[0]/max(1,(tmp["g"]=="Disability=Other/Unknown").sum()))*100,1),
                    "Maybe %": round((tmp[(tmp["g"]=="Disability=Other/Unknown") & (tmp["s"]==0.5)].shape[0]/max(1,(tmp["g"]=="Disability=Other/Unknown").sum()))*100,1),
                    "No %": round((tmp[(tmp["g"]=="Disability=Other/Unknown") & (tmp["s"]==0)].shape[0]/max(1,(tmp["g"]=="Disability=Other/Unknown").sum()))*100,1),
                    "Mean": round(tmp.loc[tmp["g"]=="Disability=Other/Unknown","s"].mean(),3)
                })

            summarize(comfort_public_col, "Comfort if approached (public)")
            summarize(trust_col, "Trust robot to guide (museum)")

        if rows:
            dis_df = pd.DataFrame(rows)
            dis_df.to_csv(os.path.join(args.out, "analysis_by_disability.csv"), index=False)
            add("[Disability] analysis_by_disability.csv written.")
        else:
            add("[Disability] No analyzable rows found after mapping; skipped.")
    else:
        add("[Disability] No disability-related columns detected; skipped.")

    # ---------------------------
    # 3) Comfort based on prior experience
    # ---------------------------
    if prior_exp_col and comfort_public_col:
        exp = map_yes_no(df[prior_exp_col]).map({1.0:"Experienced", 0.0:"New/Low"})
        score = map_yes_no_maybe(df[comfort_public_col])
        tmp = pd.DataFrame({"exp": exp, "s": score}).dropna()
        if not tmp.empty:
            g = tmp.groupby("exp")
            out = pd.DataFrame({
                "N": g.size(),
                "Yes %": g.apply(lambda s: (s["s"]==1).mean()*100),
                "Maybe %": g.apply(lambda s: (s["s"]==0.5).mean()*100),
                "No %": g.apply(lambda s: (s["s"]==0).mean()*100),
                "Mean comfort": g["s"].mean()
            }).round(1).reset_index()
            out.to_csv(os.path.join(args.out, "comfort_by_prior_experience.csv"), index=False)
            # Bar chart of Yes% by experience
            save_bar(out.set_index("exp")["Yes %"], "Comfort if Approached – Yes% by Prior Experience",
                     "Experience Group", "Yes %",
                     os.path.join(args.out, "comfort_by_experience_bar.png"))
            add("[Experience] comfort_by_prior_experience.csv written.")
        else:
            add("[Experience] Not enough data after mapping; skipped.")
    else:
        add("[Experience] Missing columns; skipped.")

    # Trust vs experience (bonus)
    if prior_exp_col and trust_col:
        exp = map_yes_no(df[prior_exp_col]).map({1.0:"Experienced", 0.0:"New/Low"})
        score = map_yes_no_maybe(df[trust_col])
        tmp = pd.DataFrame({"exp": exp, "s": score}).dropna()
        if not tmp.empty:
            g = tmp.groupby("exp")
            out = pd.DataFrame({
                "N": g.size(),
                "Yes %": g.apply(lambda s: (s["s"]==1).mean()*100),
                "Maybe %": g.apply(lambda s: (s["s"]==0.5).mean()*100),
                "No %": g.apply(lambda s: (s["s"]==0).mean()*100),
                "Mean score": g["s"].mean()
            }).round(1).reset_index()
            out.to_csv(os.path.join(args.out, "trust_by_prior_experience.csv"), index=False)

    # ---------------------------
    # 4) Multi-select comfort features
    # ---------------------------
    if multi_col:
        raw = df[multi_col].dropna().astype(str)
        choices = []
        for cell in raw:
            parts = [p.strip() for p in cell.replace(";", ",").split(",") if p.strip()]
            choices.extend(parts)
        if choices:
            counts = pd.Series(choices).value_counts()
            pct_of_all = (counts / n_resp * 100).round(1)
            out = pd.DataFrame({"Count": counts, "% of respondents": pct_of_all})
            out.index.name = "Option"
            out.reset_index().to_csv(os.path.join(args.out, "comfort_features_counts.csv"), index=False)
            # Bar chart
            save_bar(out["% of respondents"].sort_values(ascending=False),
                     "What would make you more comfortable? (% of all respondents)",
                     "Option (descending)", "% of respondents",
                     os.path.join(args.out, "comfort_features_bar.png"))
            add("[Comfort features] comfort_features_counts.csv written.")
        else:
            add("[Comfort features] No selections found; skipped.")
    else:
        add("[Comfort features] Multi-select column not detected; skipped.")

    # ---------------------------
    # 5) Distance stats
    # ---------------------------
    if distance_col:
        dist = to_numeric_distance(df[distance_col])
        if dist.notna().any():
            stats = {
                "N": int(dist.notna().sum()),
                "Mean_m": round(dist.mean(), 3),
                "Median_m": round(dist.median(), 3),
                "Std_m": round(dist.std(ddof=1), 3) if dist.notna().sum() > 1 else 0.0,
                "P10_m": round(dist.quantile(0.10), 3),
                "P25_m": round(dist.quantile(0.25), 3),
                "P75_m": round(dist.quantile(0.75), 3),
                "P90_m": round(dist.quantile(0.90), 3),
                "Min_m": round(dist.min(), 3),
                "Max_m": round(dist.max(), 3),
            }
            pd.DataFrame([stats]).to_csv(os.path.join(args.out, "distance_summary.csv"), index=False)
            save_hist(dist, bins=10, title="Comfortable Distance Distribution", xlabel="Metres",
                      ylabel="Frequency", path=os.path.join(args.out, "distance_hist.png"))
            add(f"[Distance] Mean = {stats['Mean_m']} m, Median = {stats['Median_m']} m (distance_summary.csv).")
        else:
            add("[Distance] No numeric distance values parsed; skipped.")
    else:
        add("[Distance] Distance column not detected; skipped.")

    # ---------------------------
    # 6) Comfort being approached – overall and by age
    # ---------------------------
    if comfort_public_col:
        s = map_yes_no_maybe(df[comfort_public_col])
        overall = pd.Series({
            "N": int(s.notna().sum()),
            "Yes %": round((s==1).mean()*100, 1),
            "Maybe/Depends %": round((s==0.5).mean()*100, 1),
            "No %": round((s==0).mean()*100, 1),
        })
        overall.to_frame(name="Overall").to_csv(os.path.join(args.out, "comfort_public_overall.csv"))
        # Bar of overall yes/maybe/no
        counts = pd.Series({
            "Yes": int((s==1).sum()),
            "Maybe/Depends": int((s==0.5).sum()),
            "No": int((s==0).sum())
        })
        save_bar(counts, "Comfort if Approached in Public (overall)", "Response", "Count",
                 os.path.join(args.out, "comfort_public_overall_bar.png"))

        # By age band
        if age_col:
            _, age_band = parse_age_midpoint(df[age_col])
            tmp = pd.DataFrame({"age_band": age_band, "s": s}).dropna()
            if not tmp.empty:
                g = tmp.groupby("age_band")["s"]
                by_age = pd.DataFrame({
                    "N": g.size(),
                    "Yes %": g.apply(lambda s: (s==1).mean()*100),
                    "Maybe %": g.apply(lambda s: (s==0.5).mean()*100),
                    "No %": g.apply(lambda s: (s==0).mean()*100)
                }).round(1).reset_index()
                by_age.to_csv(os.path.join(args.out, "comfort_public_by_age.csv"), index=False)
                # Bar of Yes% by age band
                save_bar(by_age.set_index("age_band")["Yes %"], "Comfort if Approached – Yes% by Age Band",
                         "Age Band", "Yes %",
                         os.path.join(args.out, "comfort_public_yes_by_age_bar.png"))
    else:
        add("[Comfort public] Column not detected; skipped.")

    # ---------------------------
    # (Optional) Trust vs age R^2 for completeness
    # ---------------------------
    if age_col and trust_col:
        age_mid, _ = parse_age_midpoint(df[age_col])
        trust_score = map_yes_no_maybe(df[trust_col])
        reg = pd.DataFrame({"age_mid": age_mid, "trust": trust_score}).dropna()
        if not reg.empty and reg["age_mid"].nunique() > 1:
            a, b, r2 = r2_linear(reg["age_mid"].values, reg["trust"].values)
            add(f"[Trust vs Age] R^2 = {r2:.3f} (y = {a:.4f}*age + {b:.4f}), N={len(reg)}")
            save_scatter_with_fit(reg["age_mid"], reg["trust"], a, b,
                                  "Trust to be Guided vs Age (1=yes, 0.5=maybe, 0=no)",
                                  "Age (midpoint)", "Trust score",
                                  os.path.join(args.out, "trust_vs_age_regression.png"))
        else:
            add("[Trust vs Age] Not enough variation to compute regression.")

    # ---------------------------
    # Write a short summary.txt
    # ---------------------------
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze HRI survey CSV for comfort/trust & related metrics.")
    parser.add_argument("--csv", required=True, help="Path to the responses CSV")
    parser.add_argument("--out", default="hri_outputs_csv", help="Output directory")
    # Optional explicit column names (if auto-detect misses)
    parser.add_argument("--age", default=None, help="Exact column name for age")
    parser.add_argument("--trust", default=None, help="Exact column name for trust-to-guide question")
    parser.add_argument("--comfort_public", default=None, help="Exact column for 'comfortable if approached in public'")
    parser.add_argument("--distance", default=None, help="Exact column for numeric distance")
    parser.add_argument("--prior", default=None, help="Exact column for prior experience with robots")
    parser.add_argument("--multi", default=None, help="Exact column for multi-select comfort features")
    args = parser.parse_args()
    main(args)
