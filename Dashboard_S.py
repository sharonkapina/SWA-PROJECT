import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Set page config ---
st.set_page_config(page_title="SWA Confidence Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    scoring_df = pd.read_csv("/Users/sharonkapina/Downloads/SWA_CODE/ALL_models_compared.csv")
    homepage_df = pd.read_csv("/Users/sharonkapina/Downloads/SWA_CODE/result with label answer/zero_shot_selected_sentences.csv")
    return scoring_df, homepage_df

scoring_df, homepage_df = load_data()

# --- Clean Goal Labels ---
scoring_df["SDG Goal"] = scoring_df["SDG Goal"].str.extract(r"(Goal \d+)", expand=False)

# --- Define Level 1 Scoring Logic ---
def assign_level1(row):
    conf_score = row.get("QA_Confidence_DistilBERT", 0)
    url_present = pd.notna(row.get("URL")) and str(row["URL"]).strip() != ""
    page_present = pd.notna(row.get("Page number")) and str(row["Page number"]).strip() != ""

    if conf_score >= 0.6 and url_present and page_present:
        return "High"
    elif conf_score < 0.6 and conf_score >= 0.4:
        return "Medium"
    else:
        return "Low"

scoring_df["Level_1_Confidence"] = scoring_df.apply(assign_level1, axis=1)

# --- Sidebar Filters ---
st.sidebar.title("📊 Filters")
industries = [
    "Information Media And Telecommunications", "Electricity Gas Water And Waste Services",
    "Mining", "Manufacturing", "Health Care And Social Assistance"
]
sdgs = ["Goal 5", "Goal 7", "Goal 12", "Goal 17"]
doc_types = scoring_df["Document Type"].dropna().unique().tolist()
years = sorted(scoring_df["Publication Date"].dropna().unique().astype(int).tolist())

selected_industries = st.sidebar.multiselect("🏭 Industry", industries, default=industries)
selected_sdgs = st.sidebar.multiselect("🎯 SDG Goals", sdgs, default=sdgs)
selected_country = st.sidebar.selectbox("🌍 Country", ["AU"])
selected_doctypes = st.sidebar.multiselect("📄 Document Type", doc_types, default=["PDF", "HTML"])
selected_years = st.sidebar.multiselect("📅 Year", years, default=years)

# --- Filter Data ---
filtered = scoring_df.copy()
filtered = filtered[filtered["Country"] == selected_country]
filtered = filtered[filtered["Document Type"].isin(selected_doctypes)]
filtered = filtered[filtered["Publication Date"].isin(selected_years)]
if selected_industries:
    filtered = filtered[filtered["Industry"].isin(selected_industries)]
if selected_sdgs:
    filtered = filtered[filtered["SDG Goal"].isin(selected_sdgs)]

# --- Completion Summary (UAT 24) ---
st.title("🌿 Sustainable World Alliance – Confidence Scoring Dashboard")
st.subheader("✅ Data Completion Summary")
expected_orgs = 50
completed_orgs = homepage_df["Organization"].nunique()
failed_orgs = expected_orgs - completed_orgs

col1, col2, col3 = st.columns(3)
col1.metric("Expected Organizations", expected_orgs)
col2.metric("Completed", completed_orgs)
col3.metric("Failed", failed_orgs)

status_df = pd.DataFrame({
    "Status": ["Completed", "Failed"],
    "Count": [completed_orgs, failed_orgs]
})
st.bar_chart(status_df.set_index("Status"))

# --- Level 1: Confidence Ratings per Answer ---
st.subheader("🔍 Level 1 Confidence Ratings (Per Answer)")
level1_counts = filtered["Level_1_Confidence"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
fig1, ax1 = plt.subplots()
sns.barplot(x=level1_counts.index, y=level1_counts.values, palette="Blues", ax=ax1)
ax1.set_ylabel("Answer Count")
st.pyplot(fig1)

# --- Level 2: Confidence per SDG per Org ---
st.subheader("📘 Level 2: Confidence per SDG per Org")

# Rule-based computation for Level 2
def compute_level2_from_level1(df):
    grouped = df.groupby(['Organization', 'SDG Goal'])['Level_1_Confidence'].value_counts().unstack().fillna(0)
    grouped.columns = [col.capitalize() for col in grouped.columns]

    level2_records = []

    for (org, goal), row in grouped.iterrows():
        high = row.get('High', 0)
        medium = row.get('Medium', 0)
        low = row.get('Low', 0)
        total = high + medium + low
        # 
        score = (2*high + 1*medium) / (2*total) if total > 0 else 0

        if score >= 0.8:
            level2 = 'High'
        elif score >= 0.3:
            level2 = 'Medium'
        else:
            level2 = 'Low'

        level2_records.append({
            "Organization": org,
            "SDG Goal": goal,
            "Level_2_Confidence": level2
        })
    return pd.DataFrame(level2_records)

level2 = compute_level2_from_level1(scoring_df)

# Apply filters
if selected_sdgs:
    level2 = level2[level2["SDG Goal"].isin(selected_sdgs)]
if selected_industries:
    orgs_in_industry = scoring_df[scoring_df["Industry"].isin(selected_industries)]["Organization"].unique()
    level2 = level2[level2["Organization"].isin(orgs_in_industry)]

# Plot
level2_counts = level2["Level_2_Confidence"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
fig2, ax2 = plt.subplots()
sns.barplot(x=level2_counts.index, y=level2_counts.values, palette="Greens", ax=ax2)
ax2.set_ylabel("SDG Count")
st.pyplot(fig2)


# --- Level 3: Per Organisation ---
st.subheader("🏢 Level 3: Confidence per Organisation")

def compute_level3_from_level2(level2_df):
    score_map = {'High': 2, 'Medium': 1, 'Low': 0}
    level2_df['Level2_Score'] = level2_df['Level_2_Confidence'].map(score_map)

    level3_records = []
    for org, group in level2_df.groupby('Organization'):
        total = len(group)
        sum_score = group['Level2_Score'].sum()
        score = sum_score / (2 * total) if total > 0 else 0

        if score >= 0.8:
            level3 = 'High'
        elif score >= 0.5:
            level3 = 'Medium'
        else:
            level3 = 'Low'

        level3_records.append({
            "Organization": org,
            "Level_3_Confidence": level3,
            "Level_3_Score": score 
        })
    return pd.DataFrame(level3_records)

level3 = compute_level3_from_level2(level2) 
level3_counts = level3["Level_3_Confidence"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
fig3, ax3 = plt.subplots()
sns.barplot(x=level3_counts.index, y=level3_counts.values, palette="Purples", ax=ax3)
ax3.set_ylabel("Organization Count")
st.pyplot(fig3)

# --- Level 4: Overall Confidence (All Orgs) ---
st.subheader("🌍 Level 4: Confidence Overall (All Orgs)")

def compute_level4_from_level3(level3_df):
    score_map = {'High': 2, 'Medium': 1, 'Low': 0}
    level3_df['Level3_Score'] = level3_df['Level_3_Confidence'].map(score_map)

    total = len(level3_df)
    sum_score = level3_df['Level3_Score'].sum()
    score = sum_score / (2 * total) if total > 0 else 0

    if score >= 0.8:
        level4 = 'High'
    elif score >= 0.3:
        level4 = 'Medium'
    else:
        level4 = 'Low'

    return pd.DataFrame([{
        "Level_4_Confidence": level4,
        "Level_4_Score": score 
    }])

level4 = compute_level4_from_level3(level3)
level4_rating = level4["Level_4_Confidence"].iloc[0] if not level4.empty else "N/A"
st.success(f"Overall Confidence Rating: *{level4_rating}*")


# --- Confidence Level by Industry ---
st.subheader("🏭 Confidence Distribution by Industry")

industry_conf = filtered.groupby(['Industry', 'Level_1_Confidence']).size().unstack(fill_value=0)
industry_conf = industry_conf.reindex(columns=["High", "Medium", "Low"], fill_value=0)

fig5, ax5 = plt.subplots(figsize=(10, 5))
industry_conf.plot(kind="bar", stacked=True, colormap="Set2", ax=ax5)
ax5.set_ylabel("Answer Count")
ax5.set_title("Level 1 Confidence Distribution per Industry")
st.pyplot(fig5)

# --- SDG Completion Heatmap by Industry ---
st.subheader("🔥 SDG Goal Distribution by Industry")

heatmap_df = filtered.groupby(['Industry', 'SDG Goal']).size().unstack(fill_value=0)
fig8, ax8 = plt.subplots(figsize=(10, 5))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt='d', ax=ax8)
ax8.set_title("SDG Goal Coverage by Industry")
st.pyplot(fig8)

# --- Top 10 Organizations by High Confidence Answers ---
st.subheader("🏅 Top 10 Organizations with Most High-Confidence Answers")

org_high = filtered[filtered["Level_1_Confidence"] == "High"]
top_orgs = org_high["Organization"].value_counts().head(10)

fig7, ax7 = plt.subplots()
sns.barplot(x=top_orgs.values, y=top_orgs.index, palette="coolwarm", ax=ax7)
ax7.set_xlabel("High-Confidence Answer Count")
st.pyplot(fig7)


# --- Pie Chart for SDG Goal Completion (UAT 25) ---
st.subheader("🥧 SDG Goal Completion Distribution")

pie_df = filtered.copy()
pie_df["SDG Goal"] = pie_df["SDG Goal"].str.extract(r"(Goal \d+)", expand=False)
pie_counts = pie_df["SDG Goal"].value_counts().reindex(["Goal 5", "Goal 7", "Goal 12", "Goal 17"], fill_value=0)

fig_pie, ax_pie = plt.subplots()
ax_pie.pie(pie_counts, labels=pie_counts.index, autopct="%1.1f%%", startangle=90)
ax_pie.axis("equal")
st.pyplot(fig_pie)






