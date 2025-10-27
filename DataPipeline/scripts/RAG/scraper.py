import argparse
import asyncio
import json
import logging
import os
import re
import ssl
import sys
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import nltk
import tiktoken
import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from newspaper import Article
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from playwright.async_api import async_playwright

nltk.set_proxy('')

warnings.filterwarnings('ignore', category=UserWarning, module='nltk')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

OUT_FILE = INPUT_DIR = (
    Path(__file__).parent.parent.parent / "data" / "RAG" /
    "raw_data" / "raw_data.jsonl"
)

# Setup logging
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"scraper_{datetime.utcnow().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

REFERENCE_HEADINGS = [
    "references",
    "bibliography",
    "works cited",
    "reference",
    "literature cited"
]

COUNTRY_TLD_MAP = {
    "us": "United States",
    "uk": "United Kingdom",
    "de": "Germany",
    "ca": "Canada",
    "au": "Australia",
    "in": "India",
}

COUNTRY_DOMAIN_MAP = {
    "nih.gov": "United States",
    "cancer.org": "United States",
    "jamanetwork.com": "United States",
    "bmj.com": "UK",
    "nature.com": "UK",
}

GLOBAL_METRICS = {
    "total_urls": 0,
    "successfully_fetched": 0,
    "with_authors": 0,
    "with_publish_date": 0,
    "with_country": 0,
    "country_count": {},
}

TOPIC_KEYWORDS = {
    "lung cancer": [
        "lung cancer",
        "nsclc",
        "small cell carcinoma",
        "adenocarcinoma",
        "large cell carcinoma",
        "squamous cell carcinoma"
    ],
    "tuberculosis": ["tuberculosis", "tb bacteria"],
    "general cancer": [
        "what is cancer",
        "cancer overview",
        "chest x-ray",
        "ct scans"
    ],
    "treatment": [
        "chemotherapy",
        "immunotherapy",
        "hyperthermia",
        "targeted-therapies",
        "surgery",
        "transplant",
        "radiation-therapy",
        "treatment options"
    ],
    "causes": [
        "risk factors",
        "carcinogens",
        "etiology",
        "genetic predisposition",
        "oncogenes",
        "tumor suppressor gene mutation",
        "smoking",
        "tobacco exposure",
        "asbestos",
        "air pollution",
        "occupational exposure",
        "radon",
        "secondhand smoke",
        "viral infection",
        "hpv",
        "immune suppression",
        "chronic inflammation",
        "radiation exposure"
    ]
}

RE_BRACKET_CITATION = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
RE_AUTHOR_YEAR = re.compile(
    r"\([A-Za-z]+(?:\s+et\s+al)?(?:,\s*\d{4})\)"
)
RE_FIG_TABLE = re.compile(
    r"^(figure|fig\.|table)\s*\d+",
    re.IGNORECASE | re.MULTILINE
)


def log(message):
    print(f"[{datetime.utcnow().isoformat()}Z] {message}")


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def download_nltk_resources():
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words'),
        ('taggers/maxent_treebank_pos_tagger', 'maxent_treebank_pos_tagger'),
    ]

    for path, name in required_resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Failed to download {name}: {e}")


def is_likely_person_name(text):
    if not text or not isinstance(text, str):
        return False

    text = text.strip()
    if len(text) > 100:
        return False

    non_name_patterns = [
        r'^https?://',
        r'@',
        r'\d{4}-\d{2}-\d{2}',  # dates
        r'^\d+$',              # just numbers
        r'copyright',
        r'all rights reserved',
        r'editorial',
        r'staff writer',
        r'news team',
        r'^for\s+',
    ]

    text_lower = text.lower()
    for pattern in non_name_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False

    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)

        # Named Entity check
        for chunk in chunks:
            if isinstance(chunk, Tree) and chunk.label() == 'PERSON':
                return True

        words = text.split()

        # Allow single-word capitalized names
        if len(words) == 1:
            w = words[0]
            if w[0].isupper() and w.isalpha() and len(w) > 1:
                return True

        # Multi-word heuristic
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w):
                titles = [
                    'dr', 'dr.', 'prof', 'prof.', 'mr', 'mr.',
                    'mrs', 'mrs.', 'ms', 'ms.'
                ]
                if any(w.lower() in titles for w in words):
                    return True
                if 2 <= len(words) <= 3:
                    return True

        return False

    except Exception:
        words = text.split()
        if len(words) == 1:
            w = words[0]
            if w[0].isupper() and w.isalpha() and len(w) > 1:
                return True
        elif 2 <= len(words) <= 4:
            if all(w and w[0].isupper() for w in words):
                common_non_names = [
                    'for', 'with', 'by', 'the', 'and', 'cancer', 'disease'
                ]
                if not any(word.lower() in common_non_names for word in words):
                    return True
        return False



def filter_author_names(authors):
    if not authors:
        return []

    filtered = []
    for author in authors:
        if is_likely_person_name(author):
            filtered.append(author)
        else:
            log(f"Filtered out non-name author: '{author}'")

    return filtered

def update_total_global_metrics(record):
    """Update global metrics based on the scraped record."""
    GLOBAL_METRICS["total_urls"] += 1
    if "error" not in record:
        GLOBAL_METRICS["successfully_fetched"] += 1
        if record.get("authors"):
            GLOBAL_METRICS["with_authors"] += 1
        if record.get("publish_date"):
            GLOBAL_METRICS["with_publish_date"] += 1
        if record.get("country"):
            GLOBAL_METRICS["with_country"] += 1
            country = record["country"]
            current_count = GLOBAL_METRICS["country_count"].get(country, 0)
            GLOBAL_METRICS["country_count"][country] = current_count + 1


def save_record_to_file(record, filename=""):
    try:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"Saved record for {record.get('link', 'unknown URL')}")
    except Exception as e:
        log(f"Failed to save record: {e}")

def extract_text_markdown(html):
    """Extract text and title from HTML as markdown."""
    article_html = trafilatura.extract(
        html,
        output_format="html",
        include_formatting=True,
        include_links=True,
        include_images=False,
        include_tables=True
    )
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata else ""
    markdown_txt = md(article_html, heading_style="ATX")
    return title, markdown_txt

def detect_source_type(url):
    """
    Detect the type of source based on the URL domain and patterns
    """
    url_lower = url.lower()
    domain = urlparse(url).netloc.lower()

    # Define source type mappings
    source_mappings = {
        "NIH/PubMed": [
            "nih.gov",
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov/pmc",
            "pmc.ncbi.nlm.nih.gov"
        ],
        "Clinical Journal": [
            "jamanetwork.com",
            "ajronline.org",
            "ascopubs.org",
            "biomedcentral.com",
            "bmj.com",
            "thelancet.com",
            "nejm.org",
            "nature.com/articles",
            "sciencedirect.com",
            "mdpi.com",
            "jphe.amegroups.org",
            "ccts.amegroups.org",
            "e-emj.org"
        ],
        "Research Database": [
            "journals.plos.org",
            "plosone.org",
            "frontiersin.org",
            "hindawi.com",
            "springer.com",
            "wiley.com",
            "academic.oup.com"
        ],
        "Trusted Health Portal": [
            "cancer.org",
            "cancer.gov",
            "mayoclinic.org",
            "cdc.gov",
            "who.int",
            "webmd.com",
            "healthline.com",
            "medicalnewstoday.com",
            "nccih.nih.gov",
            "cancerresearchuk.org",
            "maxhealthcare.in"
        ],
        "Medical Institution": [
            "clevelandclinic.org",
            "hopkinsmedicine.org",
            "mskcc.org",
            "mdanderson.org",
            "mayo.edu",
            "stanfordhealthcare.org"
        ],
        "Government Health Agency": [
            ".gov/health",
            ".gov/diseases",
            ".gov/about-cancer",
            ".gov/tb",
            "cdc.gov",
            "fda.gov",
            "hhs.gov"
        ]
    }

    # Check each source type
    for source_type, patterns in source_mappings.items():
        for pattern in patterns:
            if pattern in domain or pattern in url_lower:
                # Special case for NIH - distinguish between
                # PubMed Central and other NIH
                if source_type == "NIH/PubMed":
                    if "pmc" in url_lower or "/articles/PMC" in url:
                        return "PubMed Central"
                    elif "pubmed" in url_lower:
                        return "PubMed"
                    else:
                        return "NIH"
                return source_type

    # Additional pattern matching for specific cases
    if "/articles/" in url_lower or "/article/" in url_lower:
        journals = ["plos", "biomedcentral", "mdpi", "nature", "springer"]
        if any(journal in domain for journal in journals):
            return "Open Access Journal"

    # Default fallback
    return "Medical Website"


async def fetch_url_async(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            html = await page.content()
            await browser.close()
            log(f"Fetched URL successfully: {url}")
            return html
    except Exception as e:
        log(f"Failed to fetch URL {url}: {e}")
        return None


def strip_references(text):
    lower = text.lower()
    min_index = None
    for h in REFERENCE_HEADINGS:
        for sep in ["\n", "\r\n"]:
            idx = lower.find(sep + h)
            if idx != -1 and (min_index is None or idx < min_index):
                min_index = idx
        idx = lower.find(h + "\n")
        if idx != -1 and (min_index is None or idx < min_index):
            min_index = idx
    return text[:min_index].strip() if min_index is not None else text.strip()


def clean_text(text):
    t = RE_BRACKET_CITATION.sub("", text)
    t = RE_AUTHOR_YEAR.sub("", t)
    t = RE_FIG_TABLE.sub("", t)
    t = re.sub(r'[\n\t]+', ' ', t)
    t = re.sub(r'\s{2,}', ' ', t)
    t = re.sub(r"[^A-Za-z0-9\s\.,;:\'\"\(\)\[\]\?\!\-]", "", t)
    return t.strip()


def compute_token_stats(text):
    words = len(text.split())
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(text))
    return words, tokens


def detect_country(url, html=None):
    try:
        domain = urlparse(url).netloc.lower()
        for d, country in COUNTRY_DOMAIN_MAP.items():
            if d in domain:
                return country
        if html:
            soup = BeautifulSoup(html, "html.parser")
            meta = soup.find("meta", {"name": "geo.country"})
            if meta and meta.get("content"):
                return meta["content"]
            og_locale = soup.find("meta", {"property": "og:locale"})
            if og_locale and og_locale.get("content") and "_" in og_locale["content"]:
                return og_locale["content"].split("_")[1]
        tld = domain.split(".")[-1]
        return COUNTRY_TLD_MAP.get(tld, "Unknown")
    except Exception as e:
        log(f"Error detecting country for {url}: {e}")
        return "Unknown"


def extract_publish_date_from_html(html):
    try:
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
    except Exception as e:
        log(f"Error extracting publish date: {e}")
        return None


def extract_topic_keywords(text):
    try:
        found_keywords = set()
        lower_text = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower_text):
                    found_keywords.add(kw)
        return list(found_keywords)
    except Exception as e:
        log(f"Error extracting topic keywords: {e}")
        return []


async def process_single_url(url):
    record = {"link": url}
    try:
        html = await fetch_url_async(url)
        if not html:
            record["error"] = "Failed to fetch HTML"
            return record

        markdown_title, markdown_text = extract_text_markdown(html)

        article = Article(url)
        article.download(input_html=html)
        article.parse()

        authors = article.authors if article.authors else []
        authors = filter_author_names(authors)
        publish_date = (
            article.publish_date.isoformat()
            if article.publish_date
            else extract_publish_date_from_html(html)
        )
        country = detect_country(url, html)

        src_type = detect_source_type(url)

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = strip_references(text)
        text = clean_text(text)

        if article.title:
            title = article.title
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        else:
            title = "Untitled"

        if title == 'Untitled':
            title = markdown_title

        words, tokens = compute_token_stats(text)
        topics_found = extract_topic_keywords(text)

        record.update({
            "title": title,
            "text": text,
            "markdown_title": markdown_title,
            "markdown_text": markdown_text,
            "source_type": src_type,
            "accessed_at": datetime.utcnow().isoformat() + "Z",
            "word_count": words,
            "token_count": tokens,
            "authors": authors,
            "publish_date": publish_date,
            "country": country,
            "topics": topics_found,
            # "date_processed": datetime.utcnow().isoformat() + "Z
        })
        log(f"Processed URL successfully: {url}")
        return record

    except Exception as e:
        record["error"] = str(e)
        log(f"Error processing URL {url}: {e}")
        return record

async def main(urls, out_name=OUT_FILE, method='W'):
    """Main scraping function."""
    # download_nltk_resources()
    if method == 'W' or method == 'w':
        if os.path.exists(out_name):
            os.remove(out_name)

    try:
        for url in urls:
            record = await process_single_url(url)
            save_record_to_file(record, out_name)
            update_total_global_metrics(record)

    except Exception as e:
        logger.error(f"Error running tasks: {e}")

    logger.info("GLOBAL METRICS")
    logger.info(f"Total URLs: {GLOBAL_METRICS['total_urls']}")
    logger.info(f"Successfully fetched: {GLOBAL_METRICS['successfully_fetched']}")
    logger.info(f"With authors: {GLOBAL_METRICS['with_authors']}")
    logger.info(f"With publish date: {GLOBAL_METRICS['with_publish_date']}")
    logger.info(f"With country: {GLOBAL_METRICS['with_country']}")
    logger.info(f"Country counts: {GLOBAL_METRICS['country_count']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Provide the URL List, output file name, "
            "restart or append option"
        )
    )
    parser.add_argument(
        "-u", "--url",
        nargs="+",
        type=str,
        default=[],
        required=True,
        help="List of URLs to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        default='raw_data.jsonl',
        help="Output file name"
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        required=True,
        default='W',
        help="W - for rewrite, A - for append to existing knowledgebase"
    )

    args = parser.parse_args()
    urls = [u.strip("[]") for u in args.url]
    output_file = args.output
    method = args.method
    if urls is None or urls == "":
        urls = []
    else:
        if method not in ["W", "A"]:
            raise ValueError(
                "Invalid method. Use 'W' for rewrite or 'A' for append."
            )
        asyncio.run(main(urls, output_file, method))
