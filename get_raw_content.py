import os
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import pandas as pd
import fitz  # Ensure correct fitz from PyMuPDF
import csv

# === Setup logging === (3/4) Log any download or PDF parsing issues
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, 'collection.log')
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

FAILED_PDF_PATH = os.path.join(LOG_DIR, 'failed_pdfs.csv')
if os.path.exists(FAILED_PDF_PATH):
    os.remove(FAILED_PDF_PATH)

# === Setup Selenium === (2) Detect whether JavaScript rendering is required for the given URL
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)

error_occurred = False

#===function define===
def is_js_rendered(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if 'javascript' in response.text.lower():
            return True
        return False
    except Exception:
        return True

def is_pdf_link(url):
    try:
        head = requests.head(url, allow_redirects=True, timeout=10)
        content_type = head.headers.get('Content-Type', '').lower()
        return 'application/pdf' in content_type or url.lower().endswith('.pdf')
    except Exception:
        return url.lower().endswith('.pdf')

# (1) Fetch HTML content using either requests or Selenium, and clean out irrelevant tags
def fetch_html_content(url, use_selenium=False):
    try:
        if use_selenium:
            driver.get(url)
            time.sleep(3)
            html = driver.page_source
        else:
            response = requests.get(url, timeout=10)
            html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        body_text = soup.get_text(separator=' ', strip=True)
        return body_text
    except Exception as e:
        global error_occurred
        error_occurred = True
        logging.error(f"Error fetching HTML from {url}: {str(e)}")  # (3) Log content retrieval failures
        return None

def record_failed_pdf(org, url, error_message):
    global error_occurred
    error_occurred = True
    row = {'Organization': org, 'URL': url, 'Error': error_message}
    df_row = pd.DataFrame([row])
    if os.path.exists(FAILED_PDF_PATH):
        df_row.to_csv(FAILED_PDF_PATH, mode='a', header=False, index=False)
    else:
        df_row.to_csv(FAILED_PDF_PATH, mode='w', header=True, index=False)


def download_pdf_and_extract_text(url, save_path):
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=15)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logging.info(f"Downloaded PDF: {url} -> {save_path}")

            doc = fitz.open(save_path)
            text = ""
            for i, page in enumerate(doc, start=1):
                text += f"\n===== PAGE {i} =====\n"
                text += page.get_text()
            return text, doc.page_count
        except Exception as e:
            logging.error(f"Attempt {attempt+1}/3 - Error downloading or reading PDF {url}: {str(e)}")  # (3) Log content retrieval failures
            time.sleep(2)
    return None, 0
    
# (5/6) Save content into a CSV per organization folder and sort it by date collected
def save_content_to_csv(org, url, file_type, content, page_count):
    folder = os.path.join('output', org)
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, 'content.csv')

    # Escape newlines for CSV readability
    safe_content = content.replace('\n', ' ').replace('\r', ' ').replace('"', "''") if content else ""
    row = {
        'URL': url,
        'Date Collected': datetime.now().isoformat(),
        'File Type': file_type,
        'Page Count': page_count,
        'Raw Content': safe_content
    }

    df_row = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
    else:
        df_row.to_csv(csv_path, mode='w', header=True, index=False, quoting=csv.QUOTE_ALL)
    logging.info(f"Saved entry to CSV for {org}")

def sort_csv_by_date(folder_path):
    csv_path = os.path.join(folder_path, 'content.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df_sorted = df.sort_values(by='Date Collected', ascending=False)
            df_sorted.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        except Exception as e:
            global error_occurred
            error_occurred = True
            logging.error(f"Error sorting CSV in {folder_path}: {str(e)}") # (3) Log content retrieval failures

# === Load dataset === 
#df = pd.read_csv('Updated_dataset.csv')
df = pd.read_csv('/Users/sharonkapina/Desktop/SWA_WORKING PROJECT/generated_urls.csv') #setup path for running in data brick
# (1) Iterate through URLs and collect/stores raw content per organization
for _, row in df.iterrows():
    url = row['URL']
    org = row['Organization']
    current_org = org

    logging.info(f"Processing URL: {url} for {org}")

    if is_pdf_link(url):
        folder = os.path.join('output', org)
        os.makedirs(folder, exist_ok=True)
        parsed = urlparse(url)
        short_name = parsed.netloc.replace('.', '_') + parsed.path.replace('/', '_')
        filename = f"{short_name[:100]}.pdf"
        filepath = os.path.join(folder, filename)

        content, page_count = download_pdf_and_extract_text(url, filepath)
        if content:
            save_content_to_csv(org, url, 'PDF', content, page_count)
        else:
            logging.error(f"PDF content could not be extracted: {url}") # (3) Log content retrieval failures
            record_failed_pdf(org, url, "Failed to download or parse PDF after 3 attempts.")
            continue
    else:
        use_selenium = is_js_rendered(url)
        html = fetch_html_content(url, use_selenium=use_selenium)
        if html:
            save_content_to_csv(org, url, 'HTML', html, '')
        else:
            logging.error(f"Failed to retrieve HTML content from {url}") # (3) Log content retrieval failures

    sort_csv_by_date(os.path.join('output', org))

# Cleanup
driver.quit()
logging.shutdown()

if error_occurred:
    print("Data collection complete. Some errors occurred. Please check the log.")
    print("See also logs/failed_pdfs.csv for failed PDF URLs.")
else:
    print("Data collection complete. No errors detected.")

#library test
import fitz
print("fitz is from:", fitz.__file__)           
