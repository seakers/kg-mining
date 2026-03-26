import os
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
import time

def setup_driver(download_folder):
    options = Options()
    # options.add_argument("--headless")  # Uncomment to run without GUI
    
    # Set download preferences
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.dir", download_folder)
    options.set_preference("browser.download.useDownloadDir", True)
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
    options.set_preference("pdfjs.disabled", True)  # Disable PDF viewer
    
    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )
    return driver

def download_pdf_selenium(driver, url, expected_filename):
    try:
        driver.get(url)
        time.sleep(5)  # Wait for download to start/complete
        print(f"Downloaded: {expected_filename}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Setup
download_folder = os.path.abspath("papers_for_rag")
driver = setup_driver(download_folder)

# You may need to manually log in once
input("Log in to IEEE in the browser window, then press Enter...")

# Now download PDFs
filenames = os.listdir("sortgs_output")
start = filenames.index("Radiation_budget.csv")
filenames = filenames[start:]
pdfnames = os.listdir("papers_for_rag")

for filename in filenames:
    df = pd.read_csv(f"sortgs_output/{filename}")
    for idx, row in df.iterrows():
        title = row["Title"].replace(" ", "_").replace("/", "")
        url = row["PDF"]
        
        if f"{title}.pdf" not in pdfnames:
            download_pdf_selenium(driver, url, title)
            time.sleep(random.uniform(3, 8))

driver.quit()