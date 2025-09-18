# This cell wraps the core functionality of Url General.ipynb to be compatible with SWA_UI.py
# It replaces the original urlscrapper.generate_urls(user_inputs, matched_orgs)

async def generate_urls(user_inputs: dict, matched_orgs: list, session=None):
    import pandas as pd
    import requests
    import os
    import re
    from urllib.parse import urlparse, urljoin
    from datetime import datetime, timedelta

    # API_KEY = "AIAIzaSyARRV1gdSGsk7UtXXanr3vEFxhByDShlCA"
    # CX = "132e42c9e1ec36f4446d"
    API_KEY = "AIzaSyA2skrUBqaz373tToj0VA_OqR9Rp_NfUlQ"
    CX = '0154ab7fd8c124b84'
    #API_KEY = 'AIzaSyDXspj9RvJ67gAurHHgDd2KENPIqmjFLqk'
    #CX = '771c12498eb0440ef'
    MAX_RESULTS = 10

    # —— Setup: load existing CSV if present ——
    output_path = os.path.abspath("generated_urls.csv")
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path, dtype=str)
        # parse the historical Last Scraped timestamps with flexible format
        df_existing["Last Scraped"] = pd.to_datetime(
            df_existing["Last Scraped"],
            infer_datetime_format=True,
            dayfirst=False,
            errors="coerce"
        )
        rows = df_existing.to_dict(orient="records")
        seen_links = {}
        for r in rows:
            key = (r["Organization"], r["URL"])
            seen_links[key] = (
                r["Last Scraped"],
                set(str(r.get("SDG_Goals","")).split(", "))
            )
    else:
        rows = []
        seen_links = {}

    # —— Helper functions ——
    def google_search(query, start_year, max_results=10):
        results = []
        years = " OR ".join(str(y) for y in range(start_year, datetime.now().year + 1))
        query = f"{query} {years}"
        for start in range(1, max_results, 10):
            url = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}&start={start}'
            resp = requests.get(url).json()
            print("🧪 Google API Raw Response:", resp)
            items = resp.get("items", [])
            results.extend(item["link"] for item in items)
            if len(items) < 10:
                break
        return results

    def detect_file_type(link):
        path = urlparse(link).path.lower()
        if path.endswith(".pdf"):
            return "PDF"
        if path.endswith(".xls") or path.endswith(".xlsx"):
            return "Excel"
        if path.endswith(".html") or path.endswith(".htm") or "." not in path:
            return "HTML"
        return "Other"

    def is_trusted_link(link, org_name):
        netloc = urlparse(link).netloc.lower()
        # normalize organization name by removing spaces and parentheses
        base = re.sub(r'\s+|\([^)]*\)', '', org_name.lower())
        if base and base in netloc:
            return True
        # extract abbreviation inside parentheses and match against domain
        m = re.search(r'\(([^)]+)\)', org_name)
        if m and m.group(1).lower() in netloc:
            return True
        return False

    # —— Extract user inputs ——
    start_year       = int(user_inputs["year"])
    freq_months      = int(user_inputs.get("Frequency", "1"))
    freq_days        = freq_months * 30
    sdg_labels       = set(user_inputs.get("sdg_labels", []))
    country          = ", ".join(user_inputs.get("country", []))
    #industry_input  = ", ".join(user_inputs.get("industry", []))
    doc_types        = [d.split()[0] for d in user_inputs.get("doc_labels", [])]

    now = datetime.now()
    total = len(matched_orgs)


    # —— Main loop ——
    for i, org in enumerate(matched_orgs, 1):
        if session:
            pct = int((i / total) * 100)
            session.send_custom_message("progress_update", {"progress": pct})

        org_name = org["organisation_name"]
        industry = org.get("industry", "")
        division = org.get("division", "")

        for doc_type in doc_types:
            if doc_type.lower() == "pdf":
                # for PDFs, use filetype:pdf to get direct PDF URLs
                query = f"{org_name} sustainability OR ESG filetype:pdf Annual Report"
            else:
                query = f"{org_name} sustainability OR ESG Australia OR Annual Report"

            print(f"🔎 Searching: {query} ({start_year}–{now.year})")
            links = google_search(query, start_year)

            for link in links:
                key = (org_name, link)
                last_scraped, existing_sdgs = seen_links.get(key, (None, set()))

                # if seen recently...
                if last_scraped and (now - last_scraped).days < freq_days:
                    # ...and no new SDGs, skip
                    if sdg_labels.issubset(existing_sdgs):
                        print(f" CASE: SKIPPED - No new SDG, already scraped on {last_scraped} for {link}")
                        continue
                    # ...but if new SDGs exist, merge & update
                    else:
                        updated_sdgs = existing_sdgs.union(sdg_labels)
                        print(f"🔄 CASE: MERGED - New SDG(s) {sdg_labels - existing_sdgs} added for {link}")
                        seen_links[key] = (now, updated_sdgs)
                        rows = [r for r in rows if not (r["Organization"]==org_name and r["URL"]==link)]
                        rows.append({
                            "Organization"     : org_name,
                            "Division"         : division,
                            "Industry"         : industry,
                            "Country"          : country,
                            "SDG_Goals"        : ", ".join(sorted(updated_sdgs)),
                            "Year Range Start" : start_year,
                            "URL"              : link,
                            "File Type"        : detect_file_type(link),
                            "Flag"             : "Trusted" if is_trusted_link(link, org_name) else "Third-party",
                            # write full timestamp
                            "Last Scraped"     : now.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        continue

                # new link or outside frequency window
                print(f" CASE: NEW - Fresh scrape or expired frequency for {link}")
                ft      = detect_file_type(link)
                trusted = is_trusted_link(link, org_name)
                if ft.lower() != doc_type.lower():
                    continue

                updated_sdgs = existing_sdgs.union(sdg_labels)
                seen_links[key] = (now, updated_sdgs)
                rows = [r for r in rows if not (r["Organization"]==org_name and r["URL"]==link)]
                rows.append({
                    "Organization"     : org_name,
                    "Division"         : division,
                    "Industry"         : industry,
                    "Country"          : country,
                    "SDG_Goals"        : ", ".join(sorted(updated_sdgs)),
                    "Year Range Start" : start_year,
                    "URL"              : link,
                    "File Type"        : ft,
                    "Flag"             : "Trusted" if trusted else "Third-party",
                    "Last Scraped"     : now.strftime("%Y-%m-%d %H:%M:%S")
                })

    # —— Write merged results ——
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(" URL results merged & saved to:", output_path)
