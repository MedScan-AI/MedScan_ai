"""
Creates urls.txt in GCS (Standalone Script)
"""
import os
from google.cloud import storage
from pathlib import Path

# Set credentials (local Mac path)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path.home() / "gcp-service-account.json")

# All URLs
urls = urls = [
    "https://www.cdc.gov/tb/treatment/index.html",
    "https://www.mayoclinic.org/diseases-conditions/lung-cancer/diagnosis-treatment/drc-20374627",
    "https://www.cancer.org/cancer/understanding-cancer/what-is-cancer.html",
    "https://ajronline.org/doi/full/10.2214/AJR.07.3896",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3876596/",
    "https://jphe.amegroups.org/article/view/3668/pdf",
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
    "https://www.medicalnewstoday.com/articles/323701",
    "https://go2.org/what-is-lung-cancer/types-of-lung-cancer/",
    "https://www.uptodate.com/contents/overview-of-the-initial-treatment-and-prognosis-of-lung-cancer",
    "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/tuberculosis",
    "https://www.cdc.gov/infection-control/hcp/core-practices/index.html",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6234945/",
    # new urls
    "https://www.cdc.gov/tb/about/index.html",
    "https://www.cdc.gov/tb/signs-symptoms/index.html",
    "https://www.cdc.gov/tb/treatment/active-tuberculosis-disease.html",
    "https://www.cdc.gov/tb/hcp/clinical-overview/index.html",
    "https://www.cdc.gov/tb/hcp/treatment/tuberculosis-disease.html",
    "https://www.cdc.gov/mmwr/volumes/69/rr/rr6901a1.htm", 

    "https://www.mayoclinic.org/diseases-conditions/tuberculosis/symptoms-causes/syc-20351250",
    "https://www.mayoclinic.org/diseases-conditions/tuberculosis/diagnosis-treatment/drc-20351256",
    "https://communityhealth.mayoclinic.org/featured-stories/tuberculosis"

    "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "https://www.who.int/health-topics/tuberculosis",
    "https://www.who.int/publications/i/item/9789240096196",
    "https://www.who.int/publications/i/item/9789240048126",
    "https://www.who.int/publications/i/item/9789240063129",
    "https://www.who.int/publications/i/item/9789240107243",
    "https://www.who.int/publications-detail-redirect/9789240007048",

    "https://jamanetwork.com/journals/jama/fullarticle/2804324",
    "https://jamanetwork.com/journals/jama/fullarticle/2800774",
    "https://jamanetwork.com/journals/jama/fullarticle/2804320",

    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6857485/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8176349/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8886963/",

    "https://www.maxhealthcare.in/blogs/understanding-tuberculosis"
]

# Create local file
local_file = Path("urls.txt")
with open(local_file, 'w') as f:
    for url in urls:
        f.write(url + '\n')

# Upload to GCS using UNIFIED bucket
client = storage.Client(project="medscanai-476203")
bucket = client.bucket("medscan-data") 
blob = bucket.blob("RAG/config/urls.txt")
blob.upload_from_filename(str(local_file))

print(f"Uploaded {len(urls)} URLs to GCS")
print(f"   gs://medscan-data/RAG/config/urls.txt")  

# Cleanup
local_file.unlink()