import os
import glob
from lxml import etree
import pandas as pd
from tqdm import tqdm

# Configuration
RAW_DATA_PATH = os.path.join("data", "raw", "pubmed")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "pubmed_extracted.parquet")


def parse_pubmed_xml(file_path):
    """
    Stream-parses a single PubMed XML file to extract Title, Abstract, and ID.
    Uses 'yield' to save memory.
    """
    context = etree.iterparse(file_path, events=("end",), tag="PubmedArticle")

    for event, elem in context:
        try:
            # Locate the MedlineCitation element
            medline = elem.find("MedlineCitation")
            if medline is None:
                continue

            article = medline.find("Article")
            if article is None:
                continue

            # 1. Extract Title
            title_elem = article.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else None

            # 2. Extract Abstract
            # Abstracts can be split into multiple parts (Background, Methods, etc.)
            abstract_elem = article.find("Abstract")
            abstract_text = ""
            if abstract_elem is not None:
                texts = abstract_elem.findall("AbstractText")
                abstract_text = " ".join([t.text for t in texts if t.text])

            # 3. Extract PMID (ID)
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None

            # Only yield if we have meaningful content
            if title and abstract_text:
                yield {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract_text,
                    "source_file": os.path.basename(file_path)
                }

            # clear element to free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        except Exception as e:
            # Skip malformed records but don't stop the whole process
            continue

    del context


def main():
    # 1. verify paths
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: Raw data path not found: {RAW_DATA_PATH}")
        return

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    # 2. Find all XML files
    xml_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.xml"))
    print(f"🔎 Found {len(xml_files)} XML files in {RAW_DATA_PATH}")

    if not xml_files:
        print("❌ No XML files found. Please check your data folder.")
        return

    # 3. Process files
    all_articles = []

    # tqdm creates a progress bar in your terminal
    for xml_file in tqdm(xml_files, desc="Processing XML Files"):
        for article in parse_pubmed_xml(xml_file):
            all_articles.append(article)

    # 4. Save to Parquet (Faster and smaller than CSV)
    print(f"💾 Saving {len(all_articles)} extracted articles to {OUTPUT_FILE}...")
    df = pd.DataFrame(all_articles)
    df.to_parquet(OUTPUT_FILE, index=False)
    print("✅ Extraction Complete.")


if __name__ == "__main__":
    main()