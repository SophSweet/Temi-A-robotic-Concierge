#!/usr/bin/env python3
import pandas as pd, numpy as np, re
import matplotlib.pyplot as plt

# Load data
PATH_TO_CSV = r"C:\Users\61479\OneDrive - Queensland University of Technology\EGH400-2\Online Study\EGH400 - Human Robot Interaction (Responses) - Form Responses 1.csv"

df = pd.read_csv(PATH_TO_CSV, encoding="utf-8-sig")

# Normalise comfort (1–6 → 0–1)
comfort = df['comfort level'].astype(float)
df['comfort_norm'] = (comfort - comfort.min()) / (comfort.max() - comfort.min())

# Parse age ranges into midpoints
def parse_age(val: str):
    s = str(val).strip()
    if re.match(r'^\d+$', s):
        return float(s)
    m = re.match(r'^(\d+)\s*-\s*(\d+)$', s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.match(r'^(\d+)\s*\+$', s)
    if m:
        return float(m.group(1))
    return np.nan
df['age_numeric'] = df['Age'].apply(parse_age)

# Age bands
bins = [0, 24, 34, 49, 64, 200]
labels = ['<25','25–34','35–49','50–64','65+']
df['age_band'] = pd.cut(df['age_numeric'], bins=bins, labels=labels, include_lowest=True)

# Trust mapping
trust_map = {'Yes':1.0, 'Maybe':0.5, 'No':0.0}
df['trust_numeric'] = df['trusting a robot'].map(trust_map)

# Exposure categories
expo_map = {
    'Yes - Interacted with one':'Interacted',
    'Yes - saw one but did not interact':'Saw only',
    'No':'No exposure'
}
df['exposure'] = df['Have you ever seen or interacted with a robot in a public space? '].map(expo_map)

# Discipline canonicalisation
def canonical_degree(x: str):
    s = str(x).lower().strip()
    if not s or s in ['nan','none','']:
        return ''
    if 'mechatron' in s: return 'Mechatronics'
    if 'mechanical' in s: return 'Mechanical'
    if 'electrical' in s or 'ece' in s: return 'Electrical/Computer'
    if 'software' in s or 'information technology' in s or s=='it' or 'computer science' in s: return 'Software/IT'
    if 'civil' in s: return 'Civil'
    if 'chemical' in s or 'process' in s: return 'Chemical/Process'
    if 'biomed' in s: return 'Biomedical'
    if 'science' in s: return 'Science'
    if 'business' in s or 'commerce' in s: return 'Business/Commerce'
    return 'Other'
df['degree_cat'] = df['Degree'].apply(canonical_degree)

# 1) Trust distribution
trust_counts = df['trusting a robot'].value_counts()
trust_counts.plot(kind='bar', title='Trusting a Robot (Yes/Maybe/No)')
plt.ylabel('Count'); plt.tight_layout(); plt.show()

# 2) Comfort vs age scatter + linear fit
mask = df['age_numeric'].notna() & df['comfort_norm'].notna()
a, b = np.polyfit(df.loc[mask, 'age_numeric'], df.loc[mask, 'comfort_norm'], 1)
xs = np.linspace(df['age_numeric'].min(), df['age_numeric'].max(), 200)
plt.scatter(df.loc[mask,'age_numeric'], df.loc[mask,'comfort_norm'], alpha=0.9)
plt.plot(xs, a*xs + b, linewidth=2)
plt.title('Comfort vs Age (0–1)')
plt.xlabel('Age'); plt.ylabel('Comfort'); plt.tight_layout(); plt.show()

# 3) Comfort by age band (mean ± SE)
groups = df.groupby('age_band')['comfort_norm']
means = groups.mean()
sems = groups.std() / np.sqrt(groups.count())
plt.bar(range(len(means)), means.values, yerr=sems.values, capsize=4)
plt.xticks(range(len(means)), means.index)
plt.ylim(0,1); plt.title('Mean Comfort by Age Band (±SE)')
plt.ylabel('Comfort'); plt.tight_layout(); plt.show()

# 4) Exposure vs trust/comfort
tmp = df[['exposure','trust_numeric','comfort_norm']].dropna()
trust_means = tmp.groupby('exposure')['trust_numeric'].mean()
comfort_means = tmp.groupby('exposure')['comfort_norm'].mean()
trust_means.plot(kind='bar', title='Mean Trust by Exposure'); plt.ylabel('Trust (0–1)'); plt.tight_layout(); plt.show()
comfort_means.plot(kind='bar', title='Mean Comfort by Exposure'); plt.ylabel('Comfort (0–1)'); plt.tight_layout(); plt.show()

# 5) Comfort by discipline
deg_groups = df.groupby('degree_cat')['comfort_norm']
deg_means = deg_groups.mean().sort_values(ascending=False)
deg_sems = (deg_groups.std() / np.sqrt(deg_groups.count())).reindex(deg_means.index)
plt.barh(range(len(deg_means)), deg_means.values, xerr=deg_sems.values, capsize=4)
plt.yticks(range(len(deg_means)), deg_means.index)
plt.xlabel('Comfort (0–1)'); plt.title('Comfort by Discipline (±SE)')
plt.tight_layout(); plt.show()

# 6) Top comfort features (multi‑select counts)
from collections import Counter
counter = Counter()
for cell in df['What would make you feel more comfortable interacting with a robot guide? '].dropna():
    for part in re.split(r'[;,/]| and ', str(cell)):
        p = part.strip().lower()
        if p: counter[p] += 1
labels, counts = zip(*counter.most_common(10))
plt.barh(range(len(labels))[::-1], list(counts)[::-1])
plt.yticks(range(len(labels))[::-1], [lbl.title() for lbl in labels][::-1])
plt.title('Top Comfort Features'); plt.tight_layout(); plt.show()
