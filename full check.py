import pandas as pd
from urllib.parse import urlparse
import requests
import os
from datetime import datetime

API_KEY = "AIzaSyB3jidLSSfU7GFhHPdas4wMiIe9Z0LT5YA"
CX = "e42c9e1ec36f4446d"
MAX_RESULTS = 30

SDG_KEYWORDS = [
    "climate", "equality", "energy", "education", "poverty",
    "health", "gender", "water", "sustainable",
    "annual report", "sustainability report", "UN goals"
]

def google_search(query, max_results=30):
    results = []
    for start in range(1, max_results, 10):
        url = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}&start={start}'
        response = requests.get(url)
        print(f"[API] {response.status_code} :: {query}")
        try:
            data = response.json()
            if "items" in data:
                results.extend([item["link"] for item in data["items"]])
            if len(data.get("items", [])) < 10:
                break
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
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
    domain = urlparse(link).netloc.lower().replace("www.", "")
    org_clean = org_name.lower().replace(" ", "").replace("&", "and")
    return org_clean in domain

def generate_urls(user_inputs: dict, matched_orgs: list):
    start_year = int(user_inputs["year"])
    current_year = datetime.now().year
    sdg_descriptions = user_inputs.get("sdg_full_labels", [])
    country = ", ".join(user_inputs.get("country", []))

    rows = []
    print(f"🔍 Generating URLs for {len(matched_orgs)} orgs")

    report_keywords = [
        "annual report", "sustainability report", "ESG report",
        "CSR report", "climate report", "governance report"
    ]

    for org in matched_orgs:
        org_name = org["organisation_name"]
        print(f"\n🏢 Searching for reports from: {org_name}")

        for year in range(current_year, start_year - 1, -1):
            for report_type in report_keywords:
                query = f"{org_name} {report_type} {year} {country}"
                print(f"🔎 {query}")
                links = google_search(query)

                for link in set(links):
                    file_type = detect_file_type(link)
                    trusted = is_trusted_link(link, org_name)

                    # Only keep links likely to be actual reports
                    if "report" in link.lower():
                        rows.append({
                            "Organization": org_name,
                            "Year": year,
                            "URL": link,
                            "File Type": file_type,
                            "Flag": "Trusted" if trusted else "Third-party"
                        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_urls.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} URLs to {output_path}")


