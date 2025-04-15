import pandas as pd
from urllib.parse import urlparse
import requests
import os
from datetime import datetime
from pathlib import Path

API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("GOOGLE_CX_ID")
MAX_RESULTS = 20

SDG_KEYWORDS = [
    "annual report", "sustainability report", "ESG report", "corporate responsibility"
]

LOG_PATH = Path(__file__).parent / "url_scraper_log.txt"

def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")
    print(message)

def google_search(query, max_results=30):
    results = []
    for start in range(1, max_results, 10):
        url = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}&start={start}'
        response = requests.get(url)
        log(f"[API] {response.status_code} :: {query}")
        try:
            data = response.json()
            if "items" in data:
                results.extend([item["link"] for item in data["items"]])
            if len(data.get("items", [])) < 10:
                break
        except Exception as e:
            log(f"Failed to parse JSON: {e}")
            break
    return results

def detect_file_type(url):
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "PDF"
    elif path.endswith(".xls") or path.endswith(".xlsx"):
        return "Excel"
    elif path.endswith(".html") or path.endswith(".htm") or not "." in path:
        return "HTML"
    return "Other"

def is_trusted_link(link, org_name):
    domain = urlparse(link).netloc.lower()
    org_clean = org_name.lower().replace(" ", "").replace("&", "and").replace("(", "").replace(")", "")
    if org_clean in domain:
        return True
    keywords = org_name.lower().split()
    match_count = sum(1 for word in keywords if word in domain)
    return match_count >= 2

def generate_urls(user_inputs: dict, matched_orgs: list):
    start_year = int(user_inputs["year"])
    current_year = datetime.now().year
    doc_type = ", ".join(user_inputs["doc_labels"])
    sdg_labels = user_inputs["sdg_labels"]
    sdg_descriptions = user_inputs.get("sdg_full_labels", [])
    country = ", ".join(user_inputs.get("country", []))

    rows = []
    seen_links = set()
    log(f"\n Generating URLs for {len(matched_orgs)} orgs")

    for i, org in enumerate(matched_orgs):
        if i > 0:
            break

        org_name = org["organisation_name"]

        for year in range(current_year, start_year - 1, -1):
            for sdg, sdg_desc in zip(sdg_labels, sdg_descriptions):
                for keyword in SDG_KEYWORDS:
                    query = f"{org_name} {year} {country} {sdg_desc}"
                    log(f"\n {query}")
                    links = google_search(query)

                    for link in set(links):
                        if link in seen_links:
                            continue
                        seen_links.add(link)

                        file_type = detect_file_type(link)
                        trusted = is_trusted_link(link, org_name)
                        log(f" Checking link: {link} → Trusted? {trusted}")

                        if trusted:
                            rows.append({
                                "Organization": org_name,
                                "Year": year,
                                "URL": link,
                                "File Type": file_type,
                                "Flag": "Trusted"
                            })
                        else:
                            log(f" Discarded: {link} → Untrusted domain")

    df = pd.DataFrame(rows)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_urls.csv")
    df.to_csv(output_path, index=False)
    log(f"\n Saved {len(df)} URLs to {output_path}")
