**Sustainable World Alliance (SWA) Project**

This project develops an NLP-driven system for extracting and mapping SDG-related insights from corporate sustainability reports. 
The pipeline integrates web scraping, semantic filtering, and transformer-based QA models to produce structured, confidence-scored results.

**Main Components**

**Interface & URL Scraping**
  Collects user-defined parameters (industry, year, SDG)
  Generates target URLs via Google API queries

**SDG Question Tagging**
  Tags scraped URLs with relevant SDG questions
  Deduplicates and organizes by organization

**Raw Content Scraping & Filtering**
  Extracts text from HTML/PDFs
  Refines with semantic similarity, keyword boosts, and fuzzy matching
  Produces filtered datasets per SDG

**Model-Based QA & Evaluation**
  Applies BERT, ALBERT, DistilBERT, Electra, and SBERT for QA
  Uses regex + classification for structured answers
  Evaluates with accuracy, precision, recall, and F1

**Dashboard (Streamlit)**
  Visualizes SDG coverage, confidence scores, and organization-level results
  Includes heatmaps, bar charts, and completion tracking

**  How to Run**

# 1. Clone repo
git clone https://github.com/yourusername/swa-nlp-sdg-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline modules
python user_interface.py
python get_content.py
python filter_sdg.py
python models.py

# 4. Launch dashboard
streamlit run Dashboard_S.py

**Outputs**

sdg_filtered_content.csv → Processed dataset

zero_shot_selected_sentences.csv → Final answers per SDG question

ALL_models_compared.csv → Model predictions + scores

Streamlit Dashboard → Interactive confidence scoring & insights
