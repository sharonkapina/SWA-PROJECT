import os
import pandas as pd
import re

# Extract year from Publication Date with fallback for non-string values
def extract_year_from_date(pub_date):
    try:
        if not isinstance(pub_date, str):
            pub_date = str(pub_date)
        match = re.search(r'(20\d{2})', pub_date)
        if match:
            return int(match.group(1))
    except:
        return None
    return None

# Assign priority to URLs: sustain > esg > annual > report > other
def url_priority(url):
    url = str(url).lower()
    if "sustain" in url:
        return 1
    if "esg" in url:
        return 2
    if "annual" in url:
        return 3
    if "report" in url:
        return 4
    return 5

def process_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"\n Checking {file_path}")
    except Exception as e:
        print(f" Failed to read {file_path}: {e}")
        return

    # Ensure required columns are present
    if "Publication Date" not in df.columns or "URL" not in df.columns:
        print(f" Skipped {file_path}: Missing required columns.")
        return

    # Extract year and calculate URL priority
    df["year"] = df["Publication Date"].apply(extract_year_from_date)
    original_df = df.copy()
    df["priority"] = df["URL"].apply(url_priority)

    # Sort by year and priority if year exists; otherwise by priority only
    if df["year"].notna().any():
        df_sorted = df.sort_values(by=["year", "priority"], ascending=[False, True])
    else:
        print(" No valid year found, using keyword priority only")
        df_sorted = df.sort_values(by=["priority"])

    if df_sorted.empty:
        print(f" No valid records in {file_path}")
        return

    # Keep only the top-ranked row
    best_row = df_sorted.iloc[[0]].copy()
    best_row.drop(columns=["year", "priority"], inplace=True, errors="ignore")

    # Identify removed entries
    removed = original_df[~original_df.index.isin(best_row.index)]

    org_name = os.path.basename(os.path.dirname(file_path))
    print(f" [{org_name}] retained: {best_row['URL'].values[0]}")
    if not removed.empty:
        print(f" [{org_name}] removed {len(removed)} entries:")
        for u in removed["URL"].values:
            print(f"   - {u}")

    # Overwrite content.csv with the filtered single row
    best_row.to_csv(file_path, index=False)

def main(output_dir):
    for org in os.listdir(output_dir):
        org_path = os.path.join(output_dir, org)
        if not os.path.isdir(org_path):
            continue
        file_path = os.path.join(org_path, "content.csv")
        if os.path.exists(file_path):
            process_file(file_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "output")
    main(output_path)
