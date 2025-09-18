# ====================  SETUP SECTION ====================
# Install required libraries
!pip install transformers sentence-transformers word2number --quiet

#  Import libraries
import pandas as pd
import re
import torch
import gc
from transformers import pipeline
from google.colab import files
from datetime import datetime
from word2number import w2n

# FILE UPLOAD 
uploaded = files.upload()
input_file = list(uploaded.keys())[0]
output_file = "ALL_models_compared.csv"

#  MODELS SETUP 
qa_models = {
    "Electra": "ahotrod/electra_large_discriminator_squad2_512",
    "BERT": "deepset/bert-large-uncased-whole-word-masking-squad2",
    "DistilBERT": "distilbert-base-cased-distilled-squad",
    "ALBERT": "ktrapeznikov/albert-xlarge-v2-squad-v2"
}

#  HELPERS 
def clean_text(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

def extract_numeric_value(text):
    if pd.isna(text) or not isinstance(text, str):
        print("\n>> Text is NaN or not a string. Returning None.")
        return None

    text = text.lower()
    print(f"\n>>> Analyzing text: {text}")
    values = []

    # 1. Match numeric values with % or percent words
    matches = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*(%|percent|per\s+cent)", text)
    print(f" - Numeric % matches: {matches}")
    values += [float(m[0]) for m in matches if m[0]]

    # 2. Match word numbers like "twenty percent"
    matches_word = re.findall(r"([a-z\s\-]+?)\s*(percent|per\s+cent)", text)
    print(f" - Word % matches: {[m[0] for m in matches_word]}")
    for m in matches_word:
        try:
            num = w2n.word_to_num(m[0].strip())
            print(f"   ✓ Converted '{m[0].strip()}' → {num}")
            values.append(float(num))
        except Exception as e:
            print(f"   ✗ Failed to convert '{m[0].strip()}': {e}")
            continue

    # 3. Fallback: percent appears NEAR number (like "percent 18")
    matches_raw = re.findall(r"(percent|percentage|%)\D{0,10}?(\d{1,3}(?:\.\d+)?)", text)
    print(f" - Raw % near-word matches: {matches_raw}")
    values += [float(m[1]) for m in matches_raw if m[1]]

    # 4. Final fallback: isolated number with no keywords around it
    if not values:
        fallback = re.findall(r"\b\d{1,3}(?:\.\d+)?\b", text)
        print(f" - Fallback numeric values: {fallback}")
        try:
            values += [float(n) for n in fallback if float(n) <= 100]
        except Exception as e:
            print(f"   ✗ Failed to parse fallback values: {e}")

    print(f" >> All extracted values: {values}")
    if values:
        print(f"  Final selected value: {values[-1]}")
        return values[-1]

    print("  No numeric value found. Returning None.")
    return None


def match_to_range(value, options):
    print(f"\tMatching numeric value: {value} against options: {options}")
    for option in options:
        try:
            label, text = option.split(".", 1)
            text = text.strip().replace("%", "")
            if "<" in text:
                bound = float(re.findall(r"\d+", text)[0])
                if value < bound:
                    print(f"\tMatched to: {option}")
                    return option, 1.0
            elif ">" in text:
                bound = float(re.findall(r"\d+", text)[0])
                if value > bound:
                    print(f"\tMatched to: {option}")
                    return option, 1.0
            elif "-" in text:
                bounds = [float(x) for x in re.findall(r"\d+", text)]
                if len(bounds) == 2 and bounds[0] <= value <= bounds[1]:
                    print(f"\tMatched to: {option}")
                    return option, 1.0
        except Exception as e:
            print(f"\tError parsing option '{option}': {e}")
            continue
    print("\tNo match found.")
    return None, 0.0

def get_numeric_year(text):
    if pd.isna(text) or not isinstance(text, str):
        return None
    match = re.search(r'\b\d{1,2}-[a-zA-Z]{3}-\d{2,4}\b', text)
    if match:
        date_str = match.group(0)
        try:
            dt = datetime.strptime(date_str, "%d-%b-%Y")
        except:
            try:
                dt = datetime.strptime(date_str, "%d-%b-%y")
            except:
                return None
        return dt.year
    match = re.search(r'\b\d{4}\b', text)
    if match:
        return int(match.group())
    return None


# ==================== MAIN LOOP ====================
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()
df["Filtered Content"] = df["Filtered Content"].fillna("").astype(str).str.strip()
df["Answer Options"] = df["Answer Options"].fillna("").astype(str).str.strip()

current_year = pd.Timestamp.now().year

for model_name, model_path in qa_models.items():
    print(f"\n🔍 Running model: {model_name}")
    qa_model = pipeline("question-answering", model=model_path, tokenizer=model_path, device=0 if torch.cuda.is_available() else -1)

    df[f"Answer_{model_name}"] = ""
    df[f"Mapped_{model_name}"] = ""
    df[f"Score_{model_name}"] = 0.0
    df[f"QA_Confidence_{model_name}"] = 0.0

    for i, row in df.iterrows():
        qid = row["SDG Ques ID"]
        context = clean_text(row["Filtered Content"])
        options_raw = clean_text(row["Answer Options"])
        original_question = clean_text(row["SDG Question"])
        options = [opt.strip() for opt in options_raw.split("|||") if opt.strip()]

        if not context or not options:
            df.at[i, f"Mapped_{model_name}"] = "1. I don't know."
            continue

        try:
            result = qa_model(question=original_question, context=context)
            answer = result.get("answer", "").strip()
            qa_score = round(result.get("score", 0.0), 4)
        except:
            answer = ""
            qa_score = 0.0

        df.at[i, f"Answer_{model_name}"] = answer
        df.at[i, f"QA_Confidence_{model_name}"] = qa_score

        lowered = answer.lower()
        fallback = "1. I don't know."

        if qid in ["Q001", "Q002", "Q003"]:
            numeric_val = extract_numeric_value(answer)
            if numeric_val is not None:
                mapped, score = match_to_range(numeric_val, options)
                if mapped:
                    df.at[i, f"Mapped_{model_name}"] = mapped
                    df.at[i, f"Score_{model_name}"] = score
                    continue
            df.at[i, f"Mapped_{model_name}"] = fallback
            df.at[i, f"Score_{model_name}"] = 0.0

        elif qid == "Q004":
            try:
                result1 = qa_model(question="Has the company joined the UN Global Compact?", context=context)
                result2 = qa_model(question="In what year did the company join the UN Global Compact?", context=context)
                answer1 = result1.get("answer", "").strip()
                answer2 = result2.get("answer", "").strip()
                combined = f"{answer1} {answer2}"
                df.at[i, f"Answer_{model_name}"] = combined
                df.at[i, f"QA_Confidence_{model_name}"] = round((result1.get("score", 0) + result2.get("score", 0)) / 2, 4)
                lowered = combined.lower()
                year = get_numeric_year(combined)
                if any(k in lowered for k in ["no", "not", "never"]):
                    mapped = "2. No."
                elif "planning" in lowered and str(current_year) in lowered:
                    mapped = "3. No, but it is in planning for this year."
                elif "certification in progress" in lowered or "in progress" in lowered:
                    mapped = "4. No, but certification is in progress."
                elif year:
                    if year == current_year:
                        mapped = "5. Yes, certified, active this year."
                    elif year < current_year:
                        mapped = "6. Yes, certified , active this year, and have renewed certification once or more."
                    else:
                        mapped = "4. No, but certification is in progress."
                else:
                    mapped = fallback
                df.at[i, f"Mapped_{model_name}"] = mapped
                df.at[i, f"Score_{model_name}"] = 1.0 if mapped != fallback else 0.0
            except:
                df.at[i, f"Mapped_{model_name}"] = fallback
                df.at[i, f"Score_{model_name}"] = 0.0


        elif qid == "Q005":
            try:
                result1 = qa_model(question="Is the company compliant with IFRS?", context=context)
                result2 = qa_model(question="In what year did the company become IFRS compliant?", context=context)
                answer1 = result1.get("answer", "").strip()
                answer2 = result2.get("answer", "").strip()
                combined = f"{answer1} {answer2}"
                df.at[i, f"Answer_{model_name}"] = combined
                df.at[i, f"QA_Confidence_{model_name}"] = round((result1.get("score", 0) + result2.get("score", 0)) / 2, 4)
                lowered = combined.lower()
                year = get_numeric_year(combined)
                if re.search(r"(complies|compliant|in accordance with|reports|uses).{0,30}?(ifrs|international financial reporting standards)", lowered):
                    if year:
                        years_since = current_year - year
                        if years_since >= 10:
                            mapped = "6. 10+ year active certification."
                        elif years_since >= 7:
                            mapped = "5. 7-9 year active certification."
                        elif years_since >= 4:
                            mapped = "4. 4-6 year active certification."
                        else:
                            mapped = "3. 1-3 year active certification."
                    else:
                        mapped = "3. 1-3 year active certification."
                elif any(k in lowered for k in ["no", "not", "never"]):
                    mapped = "2. No."
                elif "ifrs" in lowered or "international financial reporting standards" in lowered:
                    mapped = "3. 1-3 year active certification."
                else:
                    mapped = fallback
                df.at[i, f"Mapped_{model_name}"] = mapped
                df.at[i, f"Score_{model_name}"] = 1.0 if mapped != fallback else 0.0
            except:
                df.at[i, f"Mapped_{model_name}"] = fallback
                df.at[i, f"Score_{model_name}"] = 0.0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# SAVE RESULTS
df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
files.download(output_file)
