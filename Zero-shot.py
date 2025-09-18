from transformers import pipeline
import pandas as pd
import numpy as np
import os
import re

def generate_label(qid, question):
    qid = str(qid)
    percent_words = "%, percent, percentage, per cent"
    if qid == "Q001":
        return f"This sentence states the percentage of women in management or leadership positions, explicitly mentioning one of: {percent_words}."
    elif qid == "Q002":
        return f"This sentence states the percentage of total renewable energy used or generated, explicitly mentioning one of: {percent_words}."
    elif qid == "Q003":
        return f"This sentence states the recycling or diversion rate or total waste recycled or V, explicitly mentioning one of: {percent_words}."
    elif qid == "Q004":
        return "This sentence explicitly mentions UN Compact certification."
    elif qid == "Q005":
        return "This sentence explicitly mentions IFRS certification."
    else:
        return f"This sentence directly answers the question with explicit numeric data, preferably mentioning one of: {percent_words}."

def contains_percent(s):
    #  %，percent, percentage, per cent 
    return bool(re.search(r'(%|\bpercent\b|\bpercentage\b|\bper\s+cent\b)', s, re.IGNORECASE))

def contains_total_overall(s):
    return bool(re.search(r'\b(total|overall|of people)\b', s, re.IGNORECASE))

def contains_leadership(s):
    return bool(re.search(r'\b(lead(ership|er|ers)?|manage(r|ment)?|executive|executives|director)\b', s, re.IGNORECASE))

# read data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '')
data = pd.read_csv(data_path)

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

THRESHOLD = 0.569

selected_sentences = []
selected_pages = []
#all_candidate_sentences = []
#all_scores = []

for idx, row in data.iterrows():
    filtered_content = row.get('Filtered Content', '/Users/sharonkapina/Downloads/SWA_CODE/sdg_filtered_content(for model).csv')
    question = row.get('SDG Question', '')
    qid = row.get('SDG Ques ID', '')

    if pd.isna(filtered_content) or not filtered_content.strip():
        selected_sentences.append(np.nan)
        selected_pages.append(1)
        #all_candidate_sentences.append("")
        #all_scores.append("")
        continue

    label = generate_label(qid, question)
    sentences = [sent.strip() for sent in str(filtered_content).split('|') if sent.strip()]
    #print(f"Row {idx} Organization: {row.get('Organization','')}, QID: {qid}")
    #print("Candidate sentences:", sentences)

    if qid in ["Q001"]:
        percent_sentences = [s for s in sentences if contains_percent(s)]
        leadership_sentences = [s for s in percent_sentences if contains_leadership(s)]

        candidate_sentences = leadership_sentences


    elif qid in ["Q002", "Q003"]:
        percent_sentences = [s for s in sentences if contains_percent(s)]
        total_overall_percent = [s for s in percent_sentences if contains_total_overall(s)]
        if total_overall_percent:
            candidate_sentences = total_overall_percent
        elif percent_sentences:
            candidate_sentences = percent_sentences
        else:
            candidate_sentences = sentences
    else:
        candidate_sentences = sentences

    if not candidate_sentences:
        selected_sentences.append(np.nan)
        selected_pages.append(1)
        #all_candidate_sentences.append("")
        #all_scores.append("")
        continue

    # zero-shot score
    scores = []
    for sent in candidate_sentences:
        result = classifier(sent, candidate_labels=[label])
        scores.append(result['scores'][0])
        
    #all_candidate_sentences.append(" | ".join(candidate_sentences))
    #all_scores.append(" | ".join([f"{s:.3f}" for s in scores]))

    top_idx = int(np.argmax(scores))
    top_score = scores[top_idx]
    top_sentence = candidate_sentences[top_idx]


    if top_score < THRESHOLD:
        selected_sentences.append(np.nan)
        selected_pages.append(1)
    else:
        selected_sentences.append(top_sentence)
        match = re.search(r"(?:\[?Page\s*)(\d+)", top_sentence, re.IGNORECASE)
        if match:
            page_number = int(match.group(1))
        else:
            page_number = 1  
        selected_pages.append(page_number)

data['Filtered Content'] = selected_sentences
data['Page number'] = selected_pages
#data['All Sentences'] = all_candidate_sentences
#data['All Scores'] = all_scores


data = data[data['Filtered Content'].astype(str).str.strip() != '']
def remove_page_prefix(text):
    if isinstance(text, str):       
        return re.sub(r'\[\s*Page\s*\d+\s*\]', '', text, flags=re.IGNORECASE).strip()
    return text

data['Filtered Content'] = data['Filtered Content'].apply(remove_page_prefix)

output_path = os.path.join(base_dir, 'zero_shot_selected_sentences.csv')
data.to_csv(output_path, index=False)

print(f"Result saved to: {output_path}")