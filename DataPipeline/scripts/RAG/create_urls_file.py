"""Create urls.txt in GCS."""
import os
from google.cloud import storage
from pathlib import Path

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path.home() / "gcp-service-account.json")

# All URLs
urls = [
    "https://www.cdc.gov/tb/treatment/index.html",
    "https://www.mayoclinic.org/diseases-conditions/lung-cancer/diagnosis-treatment/drc-20374627",
    "https://www.cancer.org/cancer/understanding-cancer/what-is-cancer.html",
    "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "https://ajronline.org/doi/full/10.2214/AJR.07.3896",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3876596/",
]

# Create local file
local_file = Path("urls.txt")
with open(local_file, 'w') as f:
    for url in urls:
        f.write(url + '\n')

# Upload to GCS
client = storage.Client(project="medscanai-476203")
bucket = client.bucket("medscan-rag-data")
blob = bucket.blob("RAG/config/urls.txt")
blob.upload_from_filename(str(local_file))

print(f"✅ Uploaded {len(urls)} URLs to GCS")
print(f"   gs://medscan-rag-data/RAG/config/urls.txt")

# Cleanup
local_file.unlink()