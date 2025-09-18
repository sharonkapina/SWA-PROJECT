import os
import json
import re

def extract_orgs_from_json(json_dir):
    org_data = []

    def extract_industry(file_name):
        match = re.search(r"ANZSIC_([A-Z])_(.+?)_2025", file_name)
        if match:
            division = f"Division {match.group(1)}"
            industry = match.group(2).replace("_", " ").replace(",", "").replace("&", "and").title()
            return division, industry
        return "", "Unknown"

    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                    raw = raw.removeprefix("```json").removesuffix("```").strip()

                    data = json.loads(raw)
                    division, industry = extract_industry(file)

                    for entry in data.get("data", []):
                        name = entry.get("organisation_name")
                        if name:
                            org_data.append({
                                "organisation_name": name,
                                "division": division,
                                "industry": industry
                            })
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")

    return org_data

