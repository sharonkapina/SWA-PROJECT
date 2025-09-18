import pandas as pd
import re
import os

# Load URL datasets and SDG Questions
df_q = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/Division URL/generated_urls_Q.csv")
df_b = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/Division URL/generated_urls_B.csv")
df_c = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/Division URL/generated_urls_C.csv")
df_d = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/Division URL/generated_urls_D.csv")
df_j = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/Division URL/generated_urls_J.csv")
urls_df = pd.concat([df_b, df_c, df_d, df_j, df_q], ignore_index=True)
sdg_questions_df = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/SDG Question/SDGs&Questions.csv")
output_path = os.path.dirname(os.path.abspath(__file__))

# Extract numeric SDG IDs from the "SDG_Goals" column
def extract_sdg_ids(text):
    return re.findall(r"Goal (\d+)", text)

urls_df["SDG_Goal_IDs"] = urls_df["SDG_Goals"].apply(extract_sdg_ids)
urls_exploded = urls_df.explode("SDG_Goal_IDs").copy()
urls_exploded["SDG_Goal_IDs"] = urls_exploded["SDG_Goal_IDs"].astype(str)

# Extract Goal number for merging
sdg_questions_df["Goal_Number"] = sdg_questions_df["SDG Goal"].str.extract(r"Goal (\d+)")

# Merge on SDG Goal number
merged_df = urls_exploded.merge(
    sdg_questions_df,
    left_on="SDG_Goal_IDs",
    right_on="Goal_Number",
    how="left"
)

# Final column selection
final_df = merged_df[[
    "Organization",
    "Country",
    "Industry",
    "SDG Goal",
    "Question",
    "Possible Answers",
    "SDGID",
    "QuestionID"
]].rename(columns={
    "Question": "SDG Question",
    "Possible Answers": "Answer Options",
    "SDGID": "SDG Goal ID",
    "QuestionID": "SDG Ques ID"
})

# Combine industries for same organization & question
final_df = final_df.groupby(["Organization", "Country", "SDG Goal", "SDG Question", "Answer Options", "SDG Goal ID", "SDG Ques ID"], as_index=False).agg({
    "Industry": lambda x: " | ".join(sorted(set(x)))
})
deduplicated_df = final_df.drop_duplicates(subset=["Organization", "SDG Ques ID"]).copy()

# Save merged and deduplicated results
deduplicated_df.to_csv(os.path.join(output_path, "deduplicated_sdg_questions_by_organization.csv"), index=False)

# Filter for SDG 5, 7, 12, 17
df_5 = deduplicated_df[deduplicated_df["SDG Goal"].str.contains("Goal 5")].copy()
df_7 = deduplicated_df[deduplicated_df["SDG Goal"].str.contains("Goal 7")].copy()
df_12 = deduplicated_df[deduplicated_df["SDG Goal"].str.contains("Goal 12")].copy()
df_17 = deduplicated_df[deduplicated_df["SDG Goal"].str.contains("Goal 17")].copy()

# Export split datasets
df_5.to_csv(os.path.join(output_path, "sdg5_questions.csv"), index=False)
df_7.to_csv(os.path.join(output_path, "sdg7_questions.csv"), index=False)
df_12.to_csv(os.path.join(output_path, "sdg12_questions.csv"), index=False)
df_17.to_csv(os.path.join(output_path, "sdg17_questions.csv"), index=False)

print("All SDG split files and merged dataset saved successfully.")
