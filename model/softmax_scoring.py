import pandas as pd
import re
from google.colab import files

# === Upload file ===
uploaded = files.upload()
input_file = list(uploaded.keys())[0]

# === Load dataset ===
df = pd.read_csv(input_file, dtype=str)  

# === Output file name ===
output_file = "ALL_models_softmax_scores.csv"

# === Define models ===
models = ["BERT", "Electra", "DistilBERT", "ALBERT"]

# === Prepare columns: convert confidence to numeric ===
for model in models:
    qa_col = f"QA_Confidence_{model}"
    sim_col = f"Score_{model}"
    df[qa_col] = pd.to_numeric(df[qa_col], errors='coerce').fillna(0.0)
    df[sim_col] = pd.to_numeric(df[sim_col], errors='coerce').fillna(0.0)
    df[f"Combined_Confidence_{model}"] = df[qa_col]  # if only using QA_Confidence now

# === Filter out "I don't know" entries ===
valid_rows = {
    model: df[df[f"Mapped_{model}"] != "1. I don't know."] for model in models
}

# === Overall Average Confidence ===
overall_confidence = {
    model: round(valid_rows[model][f"Combined_Confidence_{model}"].mean(), 4)
    for model in models
}
df_overall = pd.DataFrame.from_dict(overall_confidence, orient='index', columns=["Avg_Confidence"])
df_overall.index.name = "Model"

# === Confidence by SDG Goal ===
confidence_by_goal = {
    model: valid_rows[model].groupby("SDG Goal")[f"Combined_Confidence_{model}"].mean().round(4)
    for model in models
}
df_goal = pd.concat(confidence_by_goal, axis=1).reset_index()
df_goal.columns.name = None

# === Confidence by SDG Question ID ===
confidence_by_question = {
    model: valid_rows[model].groupby("SDG Ques ID")[f"Combined_Confidence_{model}"].mean().round(4)
    for model in models
}
df_question = pd.concat(confidence_by_question, axis=1).reset_index()
df_question.columns.name = None

# === Save all sections to a single CSV ===
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Overall Average Model Confidence\n")
    df_overall.to_csv(f)
    f.write("\nAverage Confidence by SDG Goal\n")
    df_goal.to_csv(f, index=False)
    f.write("\nAverage Confidence by SDG Question ID\n")
    df_question.to_csv(f, index=False)

print(" Self-confidence summary saved.")
files.download(output_file)
