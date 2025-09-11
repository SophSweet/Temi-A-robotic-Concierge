#!/usr/bin/env python3
"""
HRI survey analysis from CSV
- Comfort level vs age with R^2 (linear regression)
- Comfort/trust vs prior experience
- Analysis by disability-related items (if present)
- Multi-select "what would make you more comfortable?" -> counts & % of all respondents + bar chart
- Distance stats (mean/median, etc.) + histogram
- Comfort being approached: overall and by age

"""
""" HRI survey analysis from preloaded arrays (no pandas).
- Correlation/regression: Age vs Comfort (with R^2)
- Categorical comparison: Degree vs Comfort (bar of means + standard errors)
- Optional: Trust vs Age (if detected)
- Saves figures to ./hri_outputs_arrays and prints a short summary"""


import re
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG: Optional overrides (use your exact header text if auto-detect misses)
# ---------------------------
OVERRIDE_AGE_KEY     = None     # e.g. "What is your age?"
OVERRIDE_COMFORT_KEY = None     # e.g. "Would you feel comfortable if a robot approached you in public?"
OVERRIDE_DEGREE_KEY  = None     # e.g. "What degree are you studying?"
OVERRIDE_TRUST_KEY   = None     # optional trust question

TOP_DEGREE_GROUPS = 7           # show top-K degree buckets, rest → "Other"
AGE_MIN, AGE_MAX   = 16, 100    # drop clearly-invalid ages

# ---------------------------
# Import arrays dict
# ---------------------------
try:
    from extract_arrays import arrays
except Exception as e:
    raise SystemExit("Could not import 'arrays' from extract_arrays_simple.py. "
                     "Make sure it's in the same folder and has loaded your CSV.") from e

# ---------------------------
# Helpers
# ---------------------------
def find_key_like(candidates):
    keys = list(arrays.keys())
    low = [k.lower() for k in keys]
    for snip in candidates:
        s = snip.lower()
        for i, kl in enumerate(low):
            if s in kl:
                return keys[i]
    return None

def key_or_fallback(override, fallbacks):
    if override and override in arrays:
        return override
    return find_key_like(fallbacks) if fallbacks else None

def to_str_arr(vals):
    return np.array([("" if v is None else str(v)).strip() for v in vals], dtype=object)

def clean_age(values):
    """Return numeric ages (float) with ranges/+'s handled and invalids dropped."""
    vals = to_str_arr(values)
    out = np.full(vals.shape, np.nan, dtype=float)
    for i, s in enumerate(vals):
        if not s: 
            continue
        # Try plain number
        try:
            a = float(s)
            out[i] = a
            continue
        except: 
            pass
        # Range a-b
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
        if m:
            a = (float(m.group(1)) + float(m.group(2))) / 2.0
            out[i] = a
            continue
        # a+
        m2 = re.match(r"^\s*(\d+)\s*\+\s*$", s)
        if m2:
            out[i] = float(m2.group(1))
            continue
        # else leave NaN

    # clamp to plausible range
    out[(out < AGE_MIN) | (out > AGE_MAX)] = np.nan
    return out

def clean_text_na(values):
    vals = to_str_arr(values)
    bad = {"", "nan", "none", "prefer not to say", "n/a", "na"}
    return np.array([("" if v.lower() in bad else v) for v in vals], dtype=object)

def map_comfort(values):
    """
    Map comfort/trust to 0..1.
    - If numeric (1..5-like), min-max normalise (0..1).
    - Else map common yes/no/maybe & Likert text.
    """
    vals = to_str_arr(values)
    # Try numeric scale first
    as_num = []
    all_num = True
    for v in vals:
        try:
            as_num.append(float(v))
        except:
            all_num = False
            break
    if all_num and len(as_num) > 0:
        as_num = np.array(as_num, dtype=float)
        mn, mx = np.nanmin(as_num), np.nanmax(as_num)
        return (as_num - mn) / (mx - mn) if mx > mn else np.full(vals.shape, np.nan)

    # Text mapping
    out = np.full(vals.shape, np.nan, dtype=float)
    for i, s in enumerate(vals):
        sl = s.lower()
        if not sl:
            continue

        # 5-point Likert phrases
        if "very comfortable" in sl:      out[i] = 1.0
        elif "somewhat comfortable" in sl:out[i] = 0.75
        elif "neither" in sl or "neutral" in sl: out[i] = 0.5
        elif "somewhat uncomfortable" in sl:      out[i] = 0.25
        elif "very uncomfortable" in sl:  out[i] = 0.0

        # yes/no/maybe
        elif "yes" in sl:                 out[i] = 1.0
        elif "no" in sl:                  out[i] = 0.0
        elif "maybe" in sl or "depend" in sl or "unsure" in sl or "not sure" in sl:
            out[i] = 0.5
        elif "comfortable" in sl and "un" not in sl:
            out[i] = 1.0
        elif "uncomfortable" in sl:
            out[i] = 0.0
        # else leave NaN
    return out

def linear_regression(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    if x2.size < 2 or np.allclose(x2, x2[0]):
        return np.nan, np.nan, np.nan, mask
    a, b = np.polyfit(x2, y2, 1)
    yhat = a*x2 + b
    ss_res = np.sum((y2 - yhat)**2)
    ss_tot = np.sum((y2 - np.mean(y2))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return a, b, r2, mask

def spearman_rho(x, y):
    """Spearman rank correlation (NaNs dropped)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    if x2.size < 2:
        return np.nan
    rx = np.argsort(np.argsort(x2))
    ry = np.argsort(np.argsort(y2))
    rx = rx.astype(float); ry = ry.astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()))
    return float((rx*ry).sum()/denom) if denom > 0 else np.nan

def group_stats(y, groups):
    uniq = [g for g in sorted(set(groups)) if g]
    means, stds, ns = [], [], []
    for g in uniq:
        vals = np.array([y[i] for i in range(len(y)) if groups[i]==g and np.isfinite(y[i])], dtype=float)
        if vals.size == 0:
            m, s, n = np.nan, np.nan, 0
        else:
            m, s, n = float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size>1 else 0.0, int(vals.size)
        means.append(m); stds.append(s); ns.append(n)
    return uniq, np.array(means), np.array(stds), np.array(ns)

def jitter(n, scale=0.05):
    return (np.random.rand(n)-0.5)*2*scale

# Canonicalise degree text -> bucket
def canonicalise_degree(values):
    vals = clean_text_na(values)
    def bucket(s):
        sl = s.lower()
        if not sl: return ""
        # merge common synonyms
        if "mechatron" in sl:         return "Mechatronics Eng"
        if "mechanical" in sl:        return "Mechanical Eng"
        if "electrical" in sl or "computer and electrical" in sl or "ece" in sl:
            return "Electrical/Computer Eng"
        if "software" in sl or "it" == sl or "information technology" in sl or "computer science" in sl:
            return "Software/IT"
        if "civil" in sl:             return "Civil Eng"
        if "chemical" in sl or "process" in sl: return "Chemical/Process"
        if "biomed" in sl or "bio-med" in sl:   return "Biomedical"
        if "mining" in sl:            return "Mining Eng"
        if "aero" in sl:              return "Aerospace Eng"
        if "science" in sl:           return "Science"
        if "business" in sl or "commerce" in sl: return "Business/Commerce"
        if "design" in sl:            return "Design"
        return "Other"
    return np.array([bucket(v) for v in vals], dtype=object)

# ---------------------------
# Detect columns
# ---------------------------
age_key     = key_or_fallback(OVERRIDE_AGE_KEY,     ["age", "your age", "what is your age"])
comfort_key = key_or_fallback(OVERRIDE_COMFORT_KEY, ["comfort", "comfortable", "robot approached", "approach you in public", "how comfortable"])
degree_key  = key_or_fallback(OVERRIDE_DEGREE_KEY,  ["degree", "discipline", "major", "field of study"])
trust_key   = key_or_fallback(OVERRIDE_TRUST_KEY,   ["trust", "guide you", "museum"])

print("Detected keys:")
print("  Age     :", age_key)
print("  Comfort :", comfort_key)
print("  Degree  :", degree_key)
print("  Trust   :", trust_key)

if not age_key or not comfort_key:
    raise SystemExit("\nERROR: Need both Age and Comfort. Set OVERRIDE_* to exact headers if detection missed.")

# ---------------------------
# Prepare & clean
# ---------------------------
age     = clean_age(arrays[age_key])
comfort = map_comfort(arrays[comfort_key])

# ---------------------------
# 1) Age ↔ Comfort
# ---------------------------
a, b, r2, mask = linear_regression(age, comfort)
rho = spearman_rho(age, comfort)
print("\n[Age ↔ Comfort]")
print(f"  Linear: slope={a:.4f}, intercept={b:.4f}, R^2={r2:.4f}")
print(f"  Spearman ρ={rho:.4f} (rank correlation, robust for ordinal data)")
print(f"  N used   ={int(np.isfinite(age[mask]).sum())}")

plt.figure()
plt.scatter(age[mask], comfort[mask], alpha=0.9)
xs = np.linspace(float(np.nanmin(age[mask])), float(np.nanmax(age[mask])), 200)
ys = a*xs + b
plt.plot(xs, ys, linewidth=2.0)
plt.title("Comfort vs Age (0..1)\nLinear fit + Spearman ρ shown in console")
plt.xlabel("Age (midpoint if ranged)")
plt.ylabel("Comfort score (0..1)")
plt.tight_layout()
plt.show()

# Age bands
bins   = [0, 24, 34, 49, 64, 200]
labels = ["<25", "25-34", "35-49", "50-64", "65+"]
age_band = np.array([labels[np.digitize([v], bins)[0]-1] if np.isfinite(v) else "" for v in age], dtype=object)
uniq, means, stds, ns = group_stats(comfort, age_band)
sems = np.where(ns>0, stds/np.sqrt(np.maximum(ns, 1)), np.nan)

print("\n[Comfort by Age Band]  (Mean ± SE, N)")
for g, m, se, n in zip(uniq, means, sems, ns):
    se_val = (se if np.isfinite(se) else np.nan)
    print(f"  {g:<6}  {m:.3f} ± {se_val:.3f}   N={int(n)}")

plt.figure()
x = np.arange(len(uniq))
plt.bar(x, means, yerr=sems, capsize=4)
for i,(m,n_) in enumerate(zip(means, ns)):
    plt.text(i, m+(0.02 if np.isfinite(m) else 0), f"n={int(n_)}", ha="center", va="bottom", fontsize=9)
plt.xticks(x, uniq)
plt.title("Mean Comfort by Age Band (±SE, n shown)")
plt.xlabel("Age Band")
plt.ylabel("Mean Comfort (0..1)")
plt.tight_layout()
plt.show()

# ---------------------------
# 2) Degree ↔ Comfort
# ---------------------------
if degree_key:
    degree_raw = arrays[degree_key]
    degree = canonicalise_degree(degree_raw)

    # Count & keep top groups
    groups, counts = np.unique(degree[degree!=""], return_counts=True)
    order = np.argsort(-counts)
    groups  = groups[order]
    counts  = counts[order]

    top = set(groups[:TOP_DEGREE_GROUPS])
    degree_top = np.array([g if g in top else ("Other" if g!="" else "") for g in degree], dtype=object)

    uniq_d, means_d, stds_d, ns_d = group_stats(comfort, degree_top)
    # Sort by mean comfort desc
    order2 = np.argsort(-np.where(np.isfinite(means_d), means_d, -999))
    uniq_d  = [uniq_d[i] for i in order2]
    means_d = means_d[order2]
    ns_d    = ns_d[order2]
    sems_d  = np.where(ns_d>0, (stds_d[order2]/np.sqrt(np.maximum(ns_d,1))), np.nan)

    print("\n[Comfort by Degree (canonicalised, top groups; others→Other)]")
    for g, m, se, n in zip(uniq_d, means_d, sems_d, ns_d):
        se_val = (se if np.isfinite(se) else np.nan)
        print(f"  {g:<24}  {m:.3f} ± {se_val:.3f}   N={int(n)}")

    # Horizontal bar for readability
    plt.figure()
    y = np.arange(len(uniq_d))
    plt.barh(y, means_d, xerr=sems_d, capsize=4)
    for i,(m,n_) in enumerate(zip(means_d, ns_d)):
        plt.text(m + 0.02, i, f"n={int(n_)}", va="center", fontsize=9)
    plt.yticks(y, uniq_d)
    plt.xlabel("Mean Comfort (0..1)")
    plt.title("Comfort by Degree/Discipline (±SE, n shown)")
    plt.tight_layout()
    plt.show()
else:
    print("\n[Degree ↔ Comfort] Degree column not found; skipped.")

# ---------------------------
# 3) Optional: Trust ↔ Age
# ---------------------------
if OVERRIDE_TRUST_KEY or trust_key:
    key = OVERRIDE_TRUST_KEY if OVERRIDE_TRUST_KEY else trust_key
    trust = map_comfort(arrays[key])
    a2, b2, r2_2, mask2 = linear_regression(age, trust)
    rho2 = spearman_rho(age, trust)
    print("\n[Trust ↔ Age]")
    print(f"  Linear: slope={a2:.4f}, intercept={b2:.4f}, R^2={r2_2:.4f}")
    print(f"  Spearman ρ={rho2:.4f}")

    plt.figure()
    plt.scatter(age[mask2], trust[mask2], alpha=0.9)
    xs = np.linspace(float(np.nanmin(age[mask2])), float(np.nanmax(age[mask2])), 200)
    ys = a2*xs + b2
    plt.plot(xs, ys, linewidth=2.0)
    plt.title("Trust vs Age (0..1)")
    plt.xlabel("Age (midpoint if ranged)")
    plt.ylabel("Trust score (0..1)")
    plt.tight_layout()
    plt.show()

print("\nDone. (Figures were shown; nothing was saved.)")