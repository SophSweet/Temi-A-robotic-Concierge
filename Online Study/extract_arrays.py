# extract_arrays_simple.py
#
# Super simple: read a CSV in the SAME folder and create arrays named
# exactly as the column headers (stored in a dict called `arrays`).
#
# Usage:
#   python extract_arrays_simple.py
# Then in Python:
#   from extract_arrays_simple import arrays
#   age = arrays["Age"]     # example (use your exact header text)

import csv
import glob

# Use your absolute path here:
FILENAME = r"C:\Users\61479\OneDrive - Queensland University of Technology\EGH400-2\Online Study\EGH400 - Human Robot Interaction (Responses) - Form Responses 1.csv"


def load_arrays_from_csv(filename: str):
    with open(filename, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty.")

    headers = rows[0]
    data_rows = rows[1:]

    # Initialize arrays dict with empty lists for each header
    arrays = {h: [] for h in headers}

    # Fill arrays
    for row in data_rows:
        for h, val in zip(headers, row):
            arrays[h].append(val)

    return arrays

# Auto-pick the first CSV in the folder if none specified
if not FILENAME:
    csvs = sorted(glob.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV found in this folder. Put this .py file next to your CSV.")
    FILENAME = csvs[0]

arrays = load_arrays_from_csv(FILENAME)

if __name__ == "__main__":
    print(f"Loaded: {FILENAME}")
    print("Arrays created (use exact header text):")
    for name in arrays.keys():
        print(f" - {name}")
