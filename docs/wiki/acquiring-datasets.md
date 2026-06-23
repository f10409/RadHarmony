# Acquiring Datasets

This page gives concrete download instructions for every dataset in RadHarmony, grouped by access method.

## Quick reference

| Dataset | Access | Auth | Approx. size |
|---------|--------|------|--------------|
| Montgomery County CXR | NLM (public) | None | ~130 MB |
| Shenzhen Hospital CXR | NLM (public) | None | ~3.8 GB |
| RAD-ChestCT | Zenodo (public) | None | ~500 GB |
| RSNA Pneumonia | RSNA website | None | ~7 GB |
| RSNA Pediatric Bone Age | RSNA website | None | ~12 GB |
| ChestX-ray14 | Kaggle | Account | ~45 GB |
| RANZCR-CLIP | Kaggle | Account + rules | ~11 GB |
| RSNA Pneumonia (Kaggle) | Kaggle | Account + rules | ~7 GB |
| RSNA PE Detection | Kaggle | Account + rules | ~780 GB |
| RSNA 2022 Cervical Spine | Kaggle | Account + rules | ~65 GB |
| RSNA 2023 Abdominal Trauma | Kaggle | Account + rules | ~350 GB |
| RSNA 2024 Lumbar Spine | Kaggle | Account + rules | ~170 GB |
| SIIM-ACR Pneumothorax | Kaggle | Account + rules | ~5 GB |
| SIIM COVID-19 | Kaggle | Account + rules | ~145 GB |
| BRAX | PhysioNet | Account + CITI + DUA | ~170 GB |
| MIMIC-CXR (DICOM) | PhysioNet | Account + CITI + DUA | ~4.5 TB |
| MIMIC-CXR-JPG | PhysioNet | Account + CITI + DUA | ~570 GB |
| VinDr-CXR | PhysioNet | Account + CITI + DUA | ~75 GB |
| CT-RATE | HuggingFace | Account | ~2 TB |
| TAIX-Ray | HuggingFace | Account | ~10 GB |
| CheXpert | Stanford AIMI | Registration | ~440 GB |
| CheXpert-Plus | Stanford AIMI | Registration | ~60 GB |

> Each dataset is released under its own license/terms — review and comply with them before downloading. RadHarmony ships **loader code only**; it never redistributes data.

---

## Public datasets (no account required)

### Montgomery County CXR

```bash
BASE="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set"

wget -q "${BASE}/MontgomerySet.zip" -O MontgomerySet.zip
unzip -q MontgomerySet.zip
# → MontgomerySet/  (set base_dir to this directory)
```

### Shenzhen Hospital CXR

The NLM server blocks directory listings on the root URL, so the dataset must be mirrored per-subdirectory. The following fetches images, clinical readings, top-level metadata, and annotations:

```bash
BASE="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set"
DEST="./Shenzhen-Hospital-CXR-Set"

# Images and clinical readings
wget --mirror --no-parent --no-host-directories --cut-dirs=3 \
     --reject='index.html*' -e robots=off -P "$DEST" \
     "${BASE}/CXR_png/" "${BASE}/ClinicalReadings/"

# Top-level metadata
wget -P "$DEST" \
     "${BASE}/NLM-ChinaCXRSet-ReadMe.docx" \
     "${BASE}/shenzhen_consensus_roi.csv"
```

For annotation files (polygon JSON + binary masks — only needed for segmentation research), use the Python crawler because index pages inside `Annotations/` must be parsed to discover file names:

```python
#!/usr/bin/env python3
"""Download Shenzhen supplemental annotation files."""
import os, re, time, urllib.request, urllib.error
from pathlib import Path

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set"
DEST = Path("./Shenzhen-Hospital-CXR-Set")
HEADERS = {"User-Agent": "Mozilla/5.0"}
CRAWL = ["Annotations/", "Annotations-2/"]

def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2)
    return None

def list_dir(url):
    data = fetch(url.rstrip("/") + "/index.html")
    if not data:
        return []
    links = re.findall(r"href='([^']+)'", data.decode("utf-8", errors="replace"))
    out = []
    for l in links:
        if l.startswith("?") or l.startswith("/"):
            continue
        if l.endswith("/index.html"):
            l = l[:-len("index.html")]
        out.append(l)
    return out

def crawl(rel):
    if rel.endswith("/"):
        for child in list_dir(f"{BASE}/{rel.rstrip('/')}"):
            crawl(rel + child)
    else:
        dest = DEST / rel
        if dest.exists() and dest.stat().st_size > 0:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = fetch(f"{BASE}/{rel}")
        if data:
            dest.write_bytes(data)
            print(rel)

for path in CRAWL:
    crawl(path)
```

### RAD-ChestCT

```bash
# Install Zenodo CLI or use wget directly
wget -O radchestct.zip "https://zenodo.org/records/6406114/files/CT_Scan_Metadata_Complete_35747.zip"
wget "https://zenodo.org/records/6406114/files/imgtrain_Abnormality_and_Location_Labels.csv"
# Images are distributed as individual .npz files — see the Zenodo record for the full file list.
```

### RSNA Pneumonia (official release)

Download from the [RSNA AI Image Challenge](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018) page.

### RSNA Pediatric Bone Age

Download from the [RSNA AI Image Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017) page.

---

## Kaggle datasets

### Setup

Install the Kaggle CLI and place your API credentials at `~/.kaggle/kaggle.json`:

```bash
pip install kaggle
# Create ~/.kaggle/kaggle.json from https://www.kaggle.com/settings → API → Create New Token
chmod 600 ~/.kaggle/kaggle.json
```

### ChestX-ray14

```bash
kaggle datasets download -d nih-chest-xrays/data -p ./CXR14
unzip ./CXR14/data.zip -d ./CXR14
```

### RANZCR-CLIP

```bash
kaggle competitions download -c ranzcr-clip-catheter-line-classification -p ./ranzcr-clip
unzip ./ranzcr-clip/ranzcr-clip-catheter-line-classification.zip -d ./ranzcr-clip
```

### RSNA Pneumonia (Kaggle release)

```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge -p ./rsna-pneumonia-kaggle
unzip ./rsna-pneumonia-kaggle/rsna-pneumonia-detection-challenge.zip -d ./rsna-pneumonia-kaggle
```

### RSNA PE Detection

```bash
kaggle competitions download -c rsna-str-pulmonary-embolism-detection -p ./rsna-pe
unzip ./rsna-pe/rsna-str-pulmonary-embolism-detection.zip -d ./rsna-pe
```

### RSNA 2022 Cervical Spine

```bash
kaggle competitions download -c rsna-2022-cervical-spine-fracture-detection -p ./rsna-cervical-spine
unzip ./rsna-cervical-spine/rsna-2022-cervical-spine-fracture-detection.zip -d ./rsna-cervical-spine
```

### RSNA 2023 Abdominal Trauma

```bash
kaggle competitions download -c rsna-2023-abdominal-trauma-detection -p ./rsna-abdominal-trauma
unzip ./rsna-abdominal-trauma/rsna-2023-abdominal-trauma-detection.zip -d ./rsna-abdominal-trauma
```

### RSNA 2024 Lumbar Spine

```bash
kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification -p ./rsna-lumbar-spine
unzip ./rsna-lumbar-spine/rsna-2024-lumbar-spine-degenerative-classification.zip -d ./rsna-lumbar-spine
```

### SIIM-ACR Pneumothorax

```bash
kaggle competitions download -c siim-acr-pneumothorax-segmentation -p ./siim-acr-ptx
unzip ./siim-acr-ptx/siim-acr-pneumothorax-segmentation.zip -d ./siim-acr-ptx
```

### SIIM COVID-19

```bash
kaggle competitions download -c siim-covid19-detection -p ./siim-covid19
unzip ./siim-covid19/siim-covid19-detection.zip -d ./siim-covid19
```

---

## PhysioNet datasets

### Setup

1. Create a free account at [physionet.org](https://physionet.org).
2. Complete the required CITI training course.
3. Sign the Data Use Agreement for each dataset on its PhysioNet page.
4. Use your PhysioNet username and password with `wget`:

```bash
wget -r -N -c -np \
     --user <physionet-username> --ask-password \
     https://physionet.org/files/<dataset>/<version>/
```

Or install the PhysioNet client:

```bash
pip install physionet-client
physionet-client download -p <dataset>/<version>
```

### BRAX

```bash
wget -r -N -c -np --user <username> --ask-password \
     https://physionet.org/files/brax/1.1.0/
```

### MIMIC-CXR (DICOM)

```bash
wget -r -N -c -np --user <username> --ask-password \
     https://physionet.org/files/mimic-cxr/2.1.0/
```

### MIMIC-CXR-JPG

```bash
wget -r -N -c -np --user <username> --ask-password \
     https://physionet.org/files/mimic-cxr-jpg/2.0.0/
```

### VinDr-CXR

```bash
wget -r -N -c -np --user <username> --ask-password \
     https://physionet.org/files/vindr-cxr/1.0.0/
```

---

## HuggingFace datasets

### Setup

```bash
pip install huggingface_hub
huggingface-cli login   # enter your access token from hf.co/settings/tokens
```

### CT-RATE

```bash
huggingface-cli download ibrahimhamamci/CT-RATE --repo-type dataset --local-dir ./CT-RATE
```

### TAIX-Ray

```bash
huggingface-cli download TLAIM/TAIX-Ray --repo-type dataset --local-dir ./TAIX-Ray
```

---

## Stanford AIMI datasets

Both CheXpert datasets are available through the [Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/) portal. Register for a free account, then request access to the specific dataset. Download links are provided after approval.

### CheXpert

Request access at the [CheXpert dataset page](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2). After approval, download and extract the archive:

```bash
# Use the download link provided after approval
tar -xzf CheXpert-v1.0.tar.gz
# → CheXpert-v1.0/train/ and CheXpert-v1.0/valid/
```

### CheXpert-Plus

Request access at the [CheXpert-Plus dataset page](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1). Download the DICOM tarball and extract:

```bash
tar -xzf chexpertplus_dcm.tar.gz
# → chexpertplus/DICOM/Uncompressed/train/ and .../valid/
```
