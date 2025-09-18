import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
out_filename = "SDG17_Q004_UN_COMPACT.csv"
out_path = os.path.join(base_dir, out_filename)

base_url = "https://unglobalcompact.org/what-is-gc/participants/search"
participant_base = "https://unglobalcompact.org"
total_pages = 37
results = []

headers = {
    "User-Agent": "Mozilla/5.0"
}

for page in range(1, total_pages + 1):
    params = {
        "search[countries][]": "10",  # Australia
        "search[reporting_status][]": "active",
        "search[sort_direction]": "asc",
        "search[per_page]": "10",
        "search[sort_field]": "",
        "search[keywords]": "",
        "page": str(page)
    }

    print(f"Scraping page {page}...")
    response = requests.get(base_url, params=params, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", class_="participants-table")
    if not table:
        print(f"\u26a0\ufe0f No table found on page {page}")
        continue

    rows = table.find("tbody").find_all("tr")
    for row in rows:
        try:
            th = row.find("th", class_="notranslate name")
            if not th:
                print("⚠️ Skipping row without name column.")
                continue

            name = th.get_text(separator=" ", strip=True)
            link = participant_base + th.find("a")["href"].strip()

            country_td = row.find("td", class_="country")
            joined_td = row.find("td", class_="joined-on")

            if not country_td or not joined_td:
                print("⚠️ Skipping row with missing country or joined-on.")
                continue

            country = country_td.text.strip()
            joined = joined_td.text.strip()

            results.append({
                "Company Name": name,
                "Profile URL": link,
                "Country": country,
                "Joined On": joined
            })
        except Exception as e:
            print(f"⚠️ Failed to parse a row: {e}")
            continue

    sleep(5) 

df = pd.DataFrame(results)
df.to_csv(out_path, index=False)
print(f"\u2705 data save to ：{out_path}")
