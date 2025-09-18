# === Labeled Answer Scoring (Excludes "I don't know") ===
import pandas as pd
import re
from google.colab import files

# === Upload file ===
uploaded = files.upload()
input_file = list(uploaded.keys())[0]

# === Load dataset ===
df = pd.read_csv(input_file, dtype=str)  

# === Output file name ===
output_file = "model_scoring.csv"

# === Define models ===
models = ["BERT", "Electra", "DistilBERT", "ALBERT"]

# === Clean & extract numeric values ===
df["Labeled Answer"] = df["label answer"].fillna("").str.strip()
df["Labeled Answer_num"] = df["Labeled Answer"].str.extract(r'^(\d+)')

for model in models:
    mapped_col = f"Mapped_{model}"
    df[mapped_col] = df[mapped_col].fillna("").str.strip()
    df[f"{mapped_col}_num"] = df[mapped_col].str.extract(r'^(\d+)')

    # Score only confident predictions (exclude "I don't know.")
    valid_mask = df[mapped_col] != "1. I don't know."
    df[f"Correct_{model}"] = 0
    df.loc[valid_mask, f"Correct_{model}"] = (
        df.loc[valid_mask, f"{mapped_col}_num"] == df.loc[valid_mask, "Labeled Answer_num"]
    ).astype(int)

# === Filter valid rows only for scoring (per model)
valid_rows = {
    model: df[df[f"Mapped_{model}"] != "1. I don't know."] for model in models
}

# === Compute overall accuracy
overall_accuracy = {
    model: round(valid_rows[model][f"Correct_{model}"].mean(), 4)
    for model in models
}
df_overall = pd.DataFrame.from_dict(overall_accuracy, orient='index', columns=["Accuracy"])
df_overall.index.name = "Model"

# === Accuracy by SDG Goal
accuracy_by_goal = {
    model: valid_rows[model].groupby("SDG Goal")[f"Correct_{model}"].mean().round(4)
    for model in models
}
df_goal = pd.concat(accuracy_by_goal, axis=1).reset_index()
df_goal.columns.name = None

# === Accuracy by Question ID
accuracy_by_question = {
    model: valid_rows[model].groupby("SDG Ques ID")[f"Correct_{model}"].mean().round(4)
    for model in models
}
df_question = pd.concat(accuracy_by_question, axis=1).reset_index()
df_question.columns.name = None

# === Compute EM / F1 Scoring ===
metrics = []

for model in models:
    mapped_col = f"Mapped_{model}"

    TP = ((df[f"{mapped_col}_num"] == df["Labeled Answer_num"]) & (df[mapped_col] != "1. I don't know.")).sum()
    FP = ((df[f"{mapped_col}_num"] != df["Labeled Answer_num"]) & (df[mapped_col] != "1. I don't know.")).sum()
    FN = (df[mapped_col] == "1. I don't know.").sum()

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0

    metrics.append({
        "Model": model,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Accuracy (All)": round(accuracy, 4),
        "TP": TP,
        "FP": FP,
        "FN": FN
    })

df_metrics = pd.DataFrame(metrics)

# === Save all sections to a single CSV ===
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Overall Accuracy\n")
    df_overall.to_csv(f)
    f.write("\nAccuracy by SDG Goal\n")
    df_goal.to_csv(f, index=False)
    f.write("\nAccuracy by SDG Question ID\n")
    df_question.to_csv(f, index=False)
    f.write("\nF1 Precision Recall Scores\n")
    df_metrics.to_csv(f, index=False)

print(" Clean + F1 scoring completed! Results saved to: from_labelled.csv")
files.download(output_file)