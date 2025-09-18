import pandas as pd
import re
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
import torch


# Load SDG questions
base_dir = os.path.dirname(os.path.abspath(__file__))
df_sdg = pd.read_csv(os.path.join(base_dir, "SDG Question/deduplicated_sdg_questions_by_organization.csv"))

# Load all content.csv
output_dir = os.path.join(base_dir, "output")
content_data = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.lower() == "content.csv":
            org_name = os.path.basename(root)
            path = os.path.join(root, file)
            try:
                cdf = pd.read_csv(path)
                cdf["Organization"] = org_name
                content_data.append(cdf)
            except: continue

all_content_df = pd.concat(content_data, ignore_index=True) if content_data else pd.DataFrame(columns=["Organization", "URL", "Raw Content", "File Type", "Publication Date", "Date Collected"])
all_content_df.columns = [c.strip() for c in all_content_df.columns]

# Sentence split
records = []
split_re = re.compile(r'(?<=\d)\s+(?=[A-Z])')  
num_re = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b')

for _, doc in all_content_df.iterrows():
    org = doc["Organization"]
    url = doc.get("URL", "")
    ptype = doc.get("File Type", "")
    pub_date = str(doc.get("Publication Date", ""))
    last_date = str(doc.get("Date Collected", ""))
    raw = str(doc.get("Raw Content", ""))

    if "===== PAGE" in raw:
        page_blocks = re.split(r"===== PAGE (\d+) =====", raw)
        for i in range(1, len(page_blocks), 2):
            page = page_blocks[i]
            text = page_blocks[i+1] if i+1 < len(page_blocks) else ""
            cleaned = re.sub(r"(?<![.!?])\n", " ", text)
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', cleaned) if s.strip()]
            for sent in sents:
                nums = num_re.findall(sent)
                if len(nums) > 4 and split_re.search(sent):
                    parts = split_re.split(sent)
                    for part in parts:
                        part = part.strip()
                        if part:
                            records.append({
                                "Organization": org, "URL": url, "Page": page,
                                "Document Type": ptype, "Publication Date": pub_date,
                                "Last updated Date": last_date, "Sentence": part
                            })
                else:
                    records.append({
                        "Organization": org, "URL": url, "Page": page,
                        "Document Type": ptype, "Publication Date": pub_date,
                        "Last updated Date": last_date, "Sentence": sent
                    })
    else:
        page = "1"
        cleaned = re.sub(r"(?<![.!?])\n", " ", raw)
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', cleaned) if s.strip()]
        for sent in sents:
            nums = num_re.findall(sent)
            if len(nums) > 4 and split_re.search(sent):
                parts = split_re.split(sent)
                for part in parts:
                    part = part.strip()
                    if part:
                        records.append({
                            "Organization": org, "URL": url, "Page": page,
                            "Document Type": ptype, "Publication Date": pub_date,
                            "Last updated Date": last_date, "Sentence": part
                        })
            else:
                records.append({
                    "Organization": org, "URL": url, "Page": page,
                    "Document Type": ptype, "Publication Date": pub_date,
                    "Last updated Date": last_date, "Sentence": sent
                })

cand_df = pd.DataFrame(records)

# Question expansion
query_map = {
    "Q003": ["In your organization, what is the recycling rate (measured by sum of recyclables + organics divided by total garbage)?",
             "How much of your organization's total waste is recycled or reused?",
             "What percentage of your waste avoids landfill?",
             "Do you report recycling statistics or landfill diversion rates?",
             "Does your organization recycle e-waste or IT equipment?",
             "How many tonnes of waste were recycled in your company?"],
    "Q005": ["Does your organization have an active IFRS certification in sustainability?",
             "Has your organization adopted IFRS Sustainability Disclosure Standards?",
             "Do you follow any IFRS-based ESG reporting standards?",
             "Are your disclosures aligned with ISSB or IFRS guidance?",
             "Is your company reporting sustainability data using IFRS frameworks?"],
    "Q004": ["Is your organisation certified with UN Compact?",
             "Is your organization a participant in the UN Global Compact?",
             "Has your company signed the United Nations Compact principles?",
             "Do you report progress under the UN Global Compact?",
             "Are you listed as a UNGC participant?"],
    "Q002": ["In your organization, what percentage of annual energy usage is derived from renewable energy?",
             "How much of your electricity comes from solar, wind, or hydro sources?",
             "Are you purchasing GreenPower or using clean energy?",
             "Has your organization reduced scope 2 emissions through renewables?",
             "Is renewable energy contributing to your emissions reduction?"],
    "Q001": ["In your organization, what is the percentage of women in managerial positions?",
             "How many women are in executive or management positions?",
             "What proportion of leadership roles are held by women?",
             "Are women represented in decision-making roles?",
             "Do you disclose gender diversity in leadership?"]
}

# Scoring logic
print("Encoding candidate sentences with SBERT...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device) #all-MiniLM-L6-v2 faster #another model can be consider BGE-M3 take x3 times 
embeddings_tensor = model.encode(cand_df["Sentence"].tolist(), convert_to_tensor=True, show_progress_bar=True)

SIMILARITY_THRESHOLD = 0.65
TOP_K = 6

filtered, f_urls, f_pages, f_types, f_pubs, f_lasts = [], [], [], [], [], []

KEYWORD_BONUS = 0.1
NUMERIC_BONUS = 0.06
PERCENT_BONUS = 0.1

keywords_by_question = {
    "Q003": ["recycl", 'landfill',"reuse", "diversion", "divert", "e-waste", "organics", 'reuse', 'total waste','total waste recycl',"ecosystem","Diversion rate", "overall"],
    "Q002": ["renewable", "solar", "wind", "greenpower", "clean energy", 'renewable generation',"green energy","usage", "Overall", "Operated generation","total energy","renewable energy"],
    "Q001": ["women", "executive", "manager", "leadership", "female"],
    "Q004": ["un global compact", "ungc", "certified", "signed"],
    "Q005": ["ifrs", "issb", "sustainability disclosure", "esg standards", 'non-IFRS', "International Financial Reporting Standards"]
}

for _, row in df_sdg.iterrows():
    org = row["Organization"]
    q_id = row["SDG Ques ID"]
    org_cands = cand_df[cand_df["Organization"] == org]

    if org_cands.empty or q_id not in query_map:
        filtered.append("")
        f_urls.append("")
        f_pages.append("")
        f_types.append("")
        f_pubs.append("")
        f_lasts.append("")
        continue

    indices = org_cands.index.tolist()
    cand_embs = embeddings_tensor[indices]
    best_scores, best_indices = [], []

    # print(f"\nMatching for Organization: {org}, Question ID: {q_id}") # validation_1
    for alt_q in query_map[q_id]:
        #print(f"  Query variant: {alt_q}") # validation_2
        q_emb = model.encode([alt_q], convert_to_tensor=True)[0]
        scores = util.cos_sim(q_emb, cand_embs)[0]

        for idx, score in zip(indices, scores.tolist()):
            sentence = cand_df.loc[idx]["Sentence"].lower()
            bonus = 0.0
            if any(k in sentence for k in keywords_by_question.get(q_id, [])):
                bonus += (KEYWORD_BONUS + 0.07) if q_id in ["Q004", "Q005"] else KEYWORD_BONUS
            if q_id not in ["Q004", "Q005"]:
                if re.search(r"\b\d+(\.\d+)?\b", sentence):
                    bonus += NUMERIC_BONUS
                if "%" in sentence or "per cent" in sentence or "percent" in sentence or "percentage" in sentence or "rate" in sentence or "Ratio" in sentence:
                    bonus += PERCENT_BONUS

            final_score = score + bonus
            if final_score >= SIMILARITY_THRESHOLD:
                best_scores.append(final_score)
                best_indices.append(idx)

    if not best_scores:
        filtered.append("")
        f_urls.append("")
        f_pages.append("")
        f_types.append("")
        f_pubs.append("")
        f_lasts.append("")
        continue

    # Sort and deduplicate
    scored_indices = sorted(zip(best_scores, best_indices), reverse=True)
    seen_sentences = set()
    unique_scored = []
    for score, idx in scored_indices:
        norm_sentence = cand_df.loc[idx]["Sentence"].strip().lower()
        if norm_sentence not in seen_sentences:
            unique_scored.append((score, idx))
            seen_sentences.add(norm_sentence)
    scored_indices = unique_scored[:TOP_K]

    sentences, pages, urls, types, pubs, lasts = [], [], [], [], [], []
    #print("  Top matched sentences:") # validation_3
    for score, idx in scored_indices:
        rec = cand_df.loc[idx]
        #print(f"    [Score: {score:.3f}] {rec['Sentence']}") #validation_4
        sentences.append(f"[Page {rec['Page']}] {rec['Sentence']}")
        pages.append(str(rec["Page"]))
        urls.append(rec["URL"])
        types.append(rec["Document Type"])
        pubs.append(str(rec["Publication Date"]))
        lasts.append(str(rec["Last updated Date"]))

    filtered.append(" | ".join(sentences))
    f_pages.append(" | ".join(set(pages)))
    f_urls.append(" | ".join(set(urls)))
    f_types.append(types[0] if types else "")
    f_pubs.append(pubs[0] if pubs else "")
    f_lasts.append(lasts[0] if lasts else "")

# Attach SBERT results to df_sdg
df_sdg["Filtered Content"] = filtered
df_sdg["URL"] = f_urls
df_sdg["Page number"] = f_pages
df_sdg["Document Type"] = f_types
df_sdg["Publication Date"] = f_pubs
df_sdg["Last updated Date"] = f_lasts

# ++ Compact Q004 Matching Start ++

un_compact_df = pd.read_csv("/Users/sharonkapina/Downloads/SWA_CODE/output/SDG17_Q004_UN_COMPACT.csv")
un_compact_df.columns = [c.strip() for c in un_compact_df.columns]

compact_variants = []
def standardize_name(name):
    name = re.sub(r"\b(limited|ltd|pty ltd|corporation|inc|plc|group|co|singtel|Pty Limited)\b", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()

def extract_names(org):
    if pd.isna(org):
        return "", ""
    match = re.search(r"(.*?)\s*\((.*?)\)", org)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return org.strip(), ""

def extract_compact_variants(name):
    main = name
    bracket = ""
    match = re.search(r"(.*?)\s*\((.*?)\)", name)
    if match:
        main = match.group(1).strip()
        bracket = match.group(2).strip()
    return [standardize_name(main), standardize_name(bracket)] if bracket else [standardize_name(main)]

for name in un_compact_df["Company Name"].tolist():
    compact_variants.append(extract_compact_variants(name))

success_matches = []

for idx, row in df_sdg.iterrows():
    if row["SDG Ques ID"] == "Q004":
        org = row["Organization"]
        main_name, keyword = extract_names(org)
        main_name_std = standardize_name(main_name)
        keyword_std = standardize_name(keyword)
        matched = False
        candidate_index = None
        for i, variants in enumerate(compact_variants):
            for variant in variants:
                if variant:
                    score_main = fuzz.token_sort_ratio(main_name_std, variant)
                    score_keyword = fuzz.token_sort_ratio(keyword_std, variant) if keyword_std else 0
                    if max(score_main, score_keyword) >= 85:
                        candidate_index = i
                        matched = True
                        break
            if matched:
                break
        if matched:
            info = un_compact_df.iloc[candidate_index]
            current_content = df_sdg.at[idx, "Filtered Content"]
            current_url = df_sdg.at[idx, "URL"]
            new_content = f"{info['Company Name']} {info['Country']} joined un compact on {info['Joined On']}"
            new_url = info['Profile URL']
            #appended_content = (current_content + " | " if pd.notna(current_content) and current_content else "") + new_content
            #appended_url = (current_url + " | " if pd.notna(current_url) and current_url else "") + new_url
            df_sdg.at[idx, "Filtered Content"] = new_content #appended_content
            df_sdg.at[idx, "URL"] = new_url #appended_url
            success_matches.append(org)

if success_matches:
    print(f"✔️ Matched organizations Q004: {len(success_matches)}")
    print("Matched Organizations:")
    for org in success_matches:
        print(f"- {org}")
# ++ Compact Q004 Matching End ++

#Filter Clean
def contains_both_number_and_percentage(sentence: str) -> bool:
    if not isinstance(sentence, str):
        return False
    has_digit_number = re.search(r"\b\d+(\.\d+)?\b", sentence)
    number_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    has_word_number = any(word in sentence.lower() for word in number_words)
    has_percent = any(p in sentence.lower() for p in ["%", "per cent", "percent", "percentage", "rate", "ratio"])

    return bool((has_digit_number or has_word_number) and has_percent)


def clean_filtered_content(text):
    if not isinstance(text, str):
        return ""
    sentences = [s.strip() for s in text.split(" | ") if s.strip()]
    valid_sentences = [s for s in sentences if contains_both_number_and_percentage(s)]
    return " | ".join(valid_sentences)

for qid in ["Q001", "Q002", "Q003"]:
    mask = df_sdg["SDG Ques ID"] == qid
    df_sdg.loc[mask, "Filtered Content"] = df_sdg.loc[mask, "Filtered Content"].apply(clean_filtered_content)

# Final output
final_output = os.path.join(base_dir, "sdg_questions_with_full_filtered_content.csv")
df_sdg.to_csv(final_output, index=False)
print(f"Full filtered SDG content saved to: {final_output}")

##### for model train
valid_mask = df_sdg["Filtered Content"].notna() & (df_sdg["Filtered Content"].str.strip() != "")
filtered_only_df = df_sdg[valid_mask].copy()

cleaned_output = os.path.join(base_dir, "sdg_filtered_content(for model).csv")
filtered_only_df.to_csv(cleaned_output, index=False)
print(f" Cleaned (non-empty) filtered dataset saved to: {cleaned_output}")

########
def generate_sdg_filtering_summary(df: pd.DataFrame, output_path: str):
    total_per_question = df.groupby("SDG Question").size()
    valid_mask = df["Filtered Content"].notna() & (df["Filtered Content"].str.strip() != "")
    filtered_per_question = df[valid_mask].groupby("SDG Question").size()

    summary_df = pd.DataFrame({
        "Total Count": total_per_question,
        "Filtered Count": filtered_per_question
    }).fillna(0)
    summary_df["Filtered Count"] = summary_df["Filtered Count"].astype(int)
    summary_df["Percentage"] = (summary_df["Filtered Count"] / summary_df["Total Count"] * 100).round(2)

    total_row = pd.DataFrame({
        "Total Count": [summary_df["Total Count"].sum()],
        "Filtered Count": [summary_df["Filtered Count"].sum()],
        "Percentage": [(summary_df["Filtered Count"].sum() / summary_df["Total Count"].sum() * 100).round(2)]
    }, index=["TOTAL"])

    summary_df = pd.concat([summary_df, total_row])
    summary_df.to_csv(output_path)
    print(f" SDG filtering summary saved to: {output_path}")
summary_output = os.path.join(base_dir, "sdg_question_filtering_summary.csv")
generate_sdg_filtering_summary(df_sdg, summary_output)