import logging
import os
import sys
import time
import asyncio
import shutil
import aiohttp

import trafilatura
import fitz  # PyMuPDF
from newspaper import Article
from markdownify import markdownify as md
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import tempfile

import re
from pathlib import Path
from datetime import datetime
from typing import List

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
script_name = str(LOG_DIR / f"{Path(__file__).stem}_{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(script_name),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "RAG" / "raw_data"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def fetch_page(url: str, retries: int = 3, delay: int = 2):
    for attempt in range(retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=60000)
                html = await page.content()
                await browser.close()
                # return html
            article = Article(url)
            article.download(input_html=html)
            article.parse()
            return article
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url} : {e}")
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
            sys.exit(1)
    raise Exception(f"Failed to fetch {url} after {retries} attempts")

def extract_date(html):
    soup = BeautifulSoup(html, "html.parser")
    meta_tags = [
        {"name": "pubdate"},
        {"name": "publishdate"},
        {"name": "publish_date"},
        {"name": "date"},
        {"name": "datePublished"},
        {"name": "updated"},
        {"name": "dateModified"},
        {"name": "Last-Modified"},
        {"property": "article:published_time"},
        {"property": "article:modified_time"},
        {"itemprop": "datePublished"},
        {"itemprop": "dateModified"},
    ]
    for tag in meta_tags:
        meta = soup.find("meta", tag)
        if meta and meta.get("content"):
            return meta["content"]

    text_patterns = [
        r"Updated on (\d{4}-\d{2}-\d{2})",
        r"Published on (\d{4}-\d{2}-\d{2})",
        r"Last updated: (\d{4}-\d{2}-\d{2})",
        r"Written on (\d{4}-\d{2}-\d{2})",
    ]
    text = soup.get_text(separator="\n")
    for pattern in text_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

async def fetch_pdf(url: str, idx: int):
    async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                r.raise_for_status()
                pdf_bytes = await r.read()
        
    # Save to temp file (Docling needs file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    
    try:
        # Configure Docling - simpler options for newer versions
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = False  # Skip tables
        
        # Initialize converter
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF]
        )
        
        # Convert PDF to structured document
        result = converter.convert(tmp_path)
        metadata = {
            'title': result.document.name or 'Unknown',
            'authors': [],
            'date': None,
            'raw_metadata': {}
        }

        # Get PDF metadata if available
        if hasattr(result.document, 'metadata') and result.document.metadata:
            raw = result.document.metadata
            metadata['raw_metadata'] = raw
            
            # Common metadata fields
            if hasattr(raw, 'title') and raw.title:
                metadata['title'] = raw.title
            if hasattr(raw, 'author') and raw.author:
                # Authors can be a string or list
                authors = raw.author if isinstance(raw.author, list) else [raw.author]
                metadata['authors'] = [a.strip() for a in authors if a]
            if hasattr(raw, 'creation_date'):
                metadata['date'] = str(raw.creation_date)
            if hasattr(raw, 'mod_date'):
                metadata['date'] = str(raw.mod_date)
            
        
        # Export as markdown (preserves structure)
        markdown_text = result.document.export_to_markdown()
        
        with open(output_file_name, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f'title: "{metadata["title"]}"\n')
            f.write(f"authors: {', '.join(metadata['authors'])}\n")
            f.write(f"date: {metadata['date_posted']}\n")
            f.write(f"source: {url}\n")
            f.write("---\n\n")
            f.write(markdown_text)
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

    # doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # metadata = doc.metadata
    # title = metadata.get("title", f"Document_{idx}")
    # authors = metadata.get("author", "Unknown Author")
    # date = metadata.get("modDate", None)

    # # --- Convert PDF pages to HTML ---
    # html_pages = []
    # for page in doc:
    #     blocks = page.get_text("dict")["blocks"]
    #     page_html = ""
    #     for b in blocks:
    #         if b['type'] == 0:  # text block
    #             for line in b["lines"]:
    #                 for span in line["spans"]:
    #                     size = span["size"]
    #                     text = span["text"].strip()
    #                     if not text:
    #                         continue
    #                     # Simple heuristic for headings
    #                     if size >= 15:
    #                         page_html += f"<h2>{text}</h2>\n"
    #                     elif size >= 12:
    #                         page_html += f"<h3>{text}</h3>\n"
    #                     else:
    #                         page_html += f"<p>{text}</p>\n"
    #     html_pages.append(page_html)
    # pdf_html = "\n".join(html_pages)

    # # --- Clean HTML with Trafilatura ---
    # clean_html = trafilatura.extract(
    #     pdf_html,
    #     output_format="html",
    #     include_formatting=True,
    #     include_links=True,
    #     include_images=False,
    #     include_tables=True
    # )

    # # --- Convert HTML → Markdown ---
    # markdown_text = md(clean_html or pdf_html, heading_style="ATX")

    # # --- Save to Markdown file ---
    # output_path = os.path.join(output_file_name, f"{idx}.md")
    # if date == None:
    #     date = extract_date(pdf_html)
    # if date == None:
    #     date = "Unknown"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write("---\n")
    #     f.write(f'title: "{title}"\n')
    #     f.write(f"author: {authors}\n")
    #     f.write(f"date: {date}\n")
    #     f.write(f"source: {url}\n")
    #     f.write("---\n\n")
    #     f.write(markdown_text)

    # # print(f"✅ Saved Markdown: {output_path}")


async def scrape_article(url: str, idx: int, retries: int = 3):
    if url == None or url.strip() == "":
        logger.error("URL is empty or None.")
        sys.exit(1)
    if "pdf" in url.lower():
        logger.info(f"Processing PDF URL: {url}")
        await fetch_pdf(url, idx)
        return
    article = await fetch_page(url)
    
    # html = trafilatura.fetch_url(url)
    article_html = trafilatura.extract(
        article.html,
        output_format="html",
        include_formatting=True,
        include_links=True,
        include_images=False,
        include_tables=True
    )
    markdown_text = md(article_html, heading_style="ATX")
    markdown_text
    date = (
                article.publish_date.isoformat()
                if article.publish_date
                else extract_date(article.html)
            )
    if date == None:
        date = "Unknown"
    output_file_name = OUTPUT_DIR / f"{idx}.md"
    with open(output_file_name, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f'title: "{article.title}"\n')
        f.write(f"authors: {', '.join(article.authors)}\n")
        f.write(f"date: {date}\n")
        f.write(f"source: {url}\n")
        f.write("---\n\n")
        f.write(markdown_text)

async def main(urls: List[str]):
    for i, url in enumerate(urls):
        try:
            await scrape_article(url, i)
            logger.info(f"Processed and Saved URL: {url}")
            await asyncio.sleep(1)  # brief pause between requests
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")

if __name__ == "__main__":
    urls = [
        "https://www.cdc.gov/tb/treatment/index.html",
        "https://www.mayoclinic.org/diseases-conditions/lung-cancer/diagnosis-treatment/drc-20374627",
        "https://www.cancer.org/cancer/understanding-cancer/what-is-cancer.html",
        "https://ajronline.org/doi/full/10.2214/AJR.07.3896",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3876596/",
        "https://jphe.amegroups.org/article/view/3668/html",
        "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0161176",
        "https://jamanetwork.com/journals/jama/fullarticle/2777242",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11003524/",
        "https://www.sciencedirect.com/science/article/pii/S2950162824000195",
        "https://www.mdpi.com/2075-4418/15/7/908",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC12000946/",
        "https://ascopubs.org/doi/10.1200/GO.21.00100",
        "https://ccts.amegroups.org/article/view/46726/html",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8113854/",
        "https://bmccancer.biomedcentral.com/articles/10.1186/s12885-024-13350-y",
        "https://www.e-emj.org/journal/view.php?number=1607",
        "https://www.cancer.gov/about-cancer/treatment/types",
        "https://www.cancer.gov/about-cancer/treatment/types/photodynamic-therapy",
        "https://www.cancer.gov/about-cancer/treatment/types/immunotherapy",
        "https://www.cancer.gov/about-cancer/treatment/types/hyperthermia",
        "https://www.cancer.gov/about-cancer/treatment/types/hormone-therapy",
        "https://www.cancer.gov/about-cancer/treatment/types/chemotherapy",
        "https://www.cancer.gov/about-cancer/treatment/types/targeted-therapies",
        "https://www.cancer.gov/about-cancer/treatment/types/surgery",
        "https://www.cancer.gov/about-cancer/treatment/types/stem-cell-transplant",
        "https://www.cancer.gov/about-cancer/treatment/types/radiation-therapy",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types.html",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types/stem-cell-transplant.html",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types/targeted-therapy.html",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types/chemotherapy.html",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types/angiogenesis-inhibitors.html",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types/hyperthermia.html",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types/lasers-in-cancer-treatment.html",
        "https://www.cancer.org/cancer/managing-cancer/treatment-types/tumor-treating-fields.html",
        "https://www.nccih.nih.gov/health/cancer-and-complementary-health-approaches-what-you-need-to-know",
        "https://www.who.int/news-room/fact-sheets/detail/cancer",
        "https://www.cancerresearchuk.org/about-cancer/treatment/prehabilitation",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/small-cell-lung-cancer",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/non-small-cell-lung-cancer",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/chemotherapy-treatment",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/radiotherapy",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/surgery",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/chemoradiotherapy",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/immunotherapy-targeted",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/treating-symptoms-metastatic",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/thermal-ablation",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/laser-therapy",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/photodynamic-therapy-pdt",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/diathermy-electrocautery",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/help-you-breathe",
        "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/cryotherapy",
        "https://bmccancer.biomedcentral.com/articles/10.1186/s12885-024-13350-y",
        "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
        "https://www.maxhealthcare.in/our-specialities/cancer-care-oncology/conditions-treatments/lung-cancer?utm_source=chatgpt.com",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC9082420/",
    ]
    asyncio.run(main(urls))