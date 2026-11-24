# Get the data

<!-- blocks: run=skip reason="downloads data or installs a download tool" -->

This page shows how to download each dataset of RadHarmony. The sections group the datasets by download
site: public servers, Kaggle, PhysioNet, Hugging Face, BIMCV, Stanford AIMI and other places. The last
section gives a link for each other dataset. After the download, the dataset page gives the folder for
`base_image_dir`.

## Quick reference

| Dataset | Source | Access | Size (about) |
|---|---|---|---|
| Montgomery County CXR | NLM (public) | None | ~600 MB |
| Shenzhen Hospital CXR | NLM (public) | None | ~3.8 GB |
| OpenI IU CXR | NLM Open-i (public) | None | ~1.5 GB (PNG) |
| RAD-ChestCT | Zenodo (public) | None | ~500 GB |
| RSNA Pneumonia | RSNA website | None | ~7 GB |
| RSNA Pediatric Bone Age | RSNA website | None | ~12 GB |
| ChestX-ray14 | Kaggle | Account | ~45 GB |
| RANZCR CLiP | Kaggle | Account + rules | ~11 GB |
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
| CT-RATE | Hugging Face | Account | ~2 TB |
| TAIX-Ray | Hugging Face | Account | ~10 GB |
| ReXGradient-160K | Hugging Face | Account + agreement | ~150 GB |
| PadChest | BIMCV (b2drop) | Registration | ~1.02 TB |
| PadChest-GR | BIMCV (b2drop) | Public share | ~48 GB |
| CheXpert | Stanford AIMI | Registration | ~440 GB |
| CheXpert Plus | Stanford AIMI | Registration | ~60 GB |
| CheXlocalize | Stanford AIMI (GitHub links) | Registration | see page |
| CheXphoto | Stanford AIMI (Redivis) | Account + agreement | see page |
| ChestX-Det10 | GitHub + Deepwise server | None | ~1.4 GB |
| NODE21 | Zenodo (public) | None | see page |
| VinDr-PCXR | PhysioNet | Account + DUA (restricted access, no CITI) | see page |
| Chest ImaGenome | PhysioNet | Account + CITI + DUA | see page |
| MS-CXR | PhysioNet | Account + CITI + DUA | small (CSV) |
| MS-CXR-T | PhysioNet | Account + CITI + DUA | small (CSV) |
| QaTa-COV19 v2 | Kaggle | Account | ~5 GB |
| MedSeg COVID-19 CT | figshare | None | small |
| VQA-RAD | OSF | None | small |
| CheXinstruct VQA | Several sources (no single download) | Mixed | see page |
| GEMeX-VQA | Hugging Face (+ MIMIC-CXR-JPG) | Account for MIMIC | small |
| MIMIC-Ext-CXR-QBA | PhysioNet | Account + CITI + DUA | see page |
| ROCO | GitHub fetch script (PubMed) | None | see page |
| ROCOv2 | Zenodo / Hugging Face | None | see page |
| MedPMC | Hugging Face | None | very large (11M pairs) |

"see page" means: read the dataset page. CITI is the human research training course, and
DUA is the data use agreement.

> Each dataset has its own license and terms. Read them and follow them before you download. RadHarmony
> gives only the code that loads the data. It does not share any data.

---

## Public datasets (no account required)

### Montgomery County CXR

The server has no zip file. These commands copy the `MontgomerySet/` folder (about 600 MB) to the current
folder.

```bash
BASE="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set"
wget -q -r -np -nH --cut-dirs=3 -e robots=off --reject "index.html*" "${BASE}/MontgomerySet/index.html"
ls MontgomerySet/CXR_png | wc -l  # 138
# set base_image_dir to MontgomerySet/CXR_png/
```

### Shenzhen Hospital CXR

The NLM server does not list the files of the root folder. So copy each subfolder on its own. These
commands get the images, the clinical readings and the top-level files:

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

The annotation files (polygon JSON and binary masks) are only for segmentation. To get them, use this
Python script. It reads the index pages in `Annotations/` to find the file names.

<!-- block: run=skip reason="downloads the Shenzhen annotation files from the NLM server" -->
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
wget -O radchestct.zip "https://zenodo.org/records/6406114/files/CT_Scan_Metadata_Complete_35747.zip"
wget "https://zenodo.org/records/6406114/files/imgtrain_Abnormality_and_Location_Labels.csv"
# Each image is a separate .npz file. The Zenodo record lists all files.
```

### RSNA Pneumonia (official release)

Download the data from the page of the
[RSNA AI Image Challenge](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018).

### RSNA Pediatric Bone Age

Download the data from the page of the
[RSNA AI Image Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017).

### OpenI IU CXR

```bash
BASE="https://openi.nlm.nih.gov/imgs/collections"
DEST="./OpenI-IU-CXR"

wget -P "$DEST" "${BASE}/NLMCXR_png.tgz"        # PNG images
tar -xzf "$DEST/NLMCXR_png.tgz" -C "$DEST"
mkdir -p "$DEST/images" && mv "$DEST"/*.png "$DEST/images/"   # the class reads images/<id>.png
wget -P "$DEST" "${BASE}/NLMCXR_reports.tgz"     # XML reports, in ecgen-radiology/
tar -xzf "$DEST/NLMCXR_reports.tgz" -C "$DEST"
# Optional: the DICOM files (one <n>/ folder for each report). The harmonizer reads only their
# ViewPosition tag, to correct the view. The images always come from the PNG files.
wget -P "$DEST" "${BASE}/NLMCXR_dcm.tgz"
tar -xzf "$DEST/NLMCXR_dcm.tgz" -C "$DEST"
```

---

## Kaggle datasets

### Setup

Install the Kaggle command-line tool. Put your API key file at `~/.kaggle/kaggle.json`. For a
competition, accept its rules on the Kaggle page first.

```bash
uv pip install kaggle
# Create ~/.kaggle/kaggle.json from https://www.kaggle.com/settings → API → Create New Token
chmod 600 ~/.kaggle/kaggle.json
```

### ChestX-ray14

```bash
kaggle datasets download -d nih-chest-xrays/data -p ./CXR14
unzip ./CXR14/data.zip -d ./CXR14
```

### RANZCR CLiP

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

1. Make a free account at [physionet.org](https://physionet.org).
2. Complete the CITI training course.
3. Sign the data use agreement of each dataset on its PhysioNet page.
4. Give your PhysioNet user name and password to `wget`:

```bash
wget -r -N -c -np \
     --user <physionet-username> --ask-password \
     https://physionet.org/files/<dataset>/<version>/
```

You can also use the PhysioNet client:

```bash
uv pip install physionet-client
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

## Hugging Face datasets

### Setup

```bash
uv pip install huggingface_hub
hf auth login   # enter your access token from hf.co/settings/tokens
```

### CT-RATE

```bash
hf download ibrahimhamamci/CT-RATE --repo-type dataset --local-dir ./CT-RATE
```

### TAIX-Ray

```bash
hf download TLAIM/TAIX-Ray --repo-type dataset --local-dir ./TAIX-Ray
```

### ReXGradient-160K

Access needs the ReXGradient-160K Non-Commercial Data Access and Use Agreement of Harvard Medical School
and Gradient Health. Use the data for non-commercial research only. Accept the agreement on the dataset
page, then run:

```bash
hf download rajpurkarlab/ReXGradient-160K --repo-type dataset --local-dir ./ReXGradient-160K/download
# The PNG files come in 10 zstd tar parts. Join and unpack them:
cd ./ReXGradient-160K
cat download/deid_png.part* | tar --zstd -xvf - -C .
# → deid_png/<PatientID>/<AccessionNumber>/studies/<StudyUID>/...
```

---

## BIMCV datasets

### PadChest

PadChest is on the [BIMCV b2drop share](https://b2drop.bsc.es/index.php/s/BIMCV-PadChest-FULL). You must
register. The license is for non-commercial research. The data is a set of numbered zip files of 16-bit PNG
images, and one CSV file with the labels and the reports.

Download each zip file and the CSV. Unzip `N.zip` into `images/N/`. The [PadChest
page](../reference/datasets/cxr-labels/padchest.md) gives the layout.

### PadChest-GR

PadChest-GR (grounded reports on PadChest images) is on the
[BIMCV b2drop share](https://b2drop.bsc.es/nextcloud/public.php/dav/files/PadChest-GR/?accept=zip).

- The zip of the whole folder is about 47.5 GB. It is a streamed Zip64 file, and a stopped download does
  not resume.
- It unzips to a folder named `BIMCV-Padchest-GR `, with a space at the end of the name.
- The images in it are a zip of 37 parts. `unzip` cannot read it. Run `7z x PadChest_GR.zip.001` in
  `Padchest_GR_files/`.
- The release has no license file. The terms are the BIMCV terms, so read them on the download page.

The [PadChest-GR page](../reference/datasets/cxr-boxes-masks/padchest_gr.md) gives the layout and the overlap
with PadChest.

---

## Stanford AIMI datasets

Both CheXpert datasets are on the [Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/)
site. Make a free account, and ask for access to the dataset. You get the download links after the
approval.

### CheXpert

Ask for access on the
[CheXpert dataset page](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2).
After the approval, download the archive and unpack it:

```bash
# Use the download link that you get after the approval
tar -xzf CheXpert-v1.0.tar.gz
# → CheXpert-v1.0/train/ and CheXpert-v1.0/valid/
```

### CheXpert Plus

Ask for access on the
[CheXpert-Plus dataset page](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1).
Download the DICOM archive and unpack it:

```bash
tar -xzf chexpertplus_dcm.tar.gz
# → chexpertplus/DICOM/Uncompressed/train/ and .../valid/
```

---

## Other datasets

Each item links to the dataset page. The dataset page gives the layout and the other details.

- **CheXlocalize.** [GitHub repo](https://github.com/rajpurkarlab/cheXlocalize), which links to the Stanford
  AIMI download. See [chexlocalize](../reference/datasets/cxr-boxes-masks/chexlocalize.md).
- **CheXphoto.** [Redivis `2qwg-aav2k243j`](https://stanford.redivis.com/datasets/2qwg-aav2k243j). Accept the
  Stanford research use agreement. See [chexphoto](../reference/datasets/cxr-labels/chexphoto.md).
- **ChestX-Det10.** Annotations at [GitHub](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset). Images:
  `http://resource.deepwise.com/xraychallenge/train_data.zip` and `test_data.zip`. See
  [chestxdet10](../reference/datasets/cxr-boxes-masks/chestxdet10.md).
- **NODE21.** Public Zenodo release, [DOI 10.5281/zenodo.5548363](https://doi.org/10.5281/zenodo.5548363). The
  challenge site is [node21.grand-challenge.org](https://node21.grand-challenge.org/). The license is CC
  BY-NC-ND 4.0. The release has the folders `proccessed_data` (two c's) and `original_data`. See
  [node21](../reference/datasets/cxr-boxes-masks/node21.md).
- **VinDr-PCXR.** [PhysioNet](https://physionet.org/content/vindr-pcxr/). See
  [vindr_pcxr](../reference/datasets/cxr-boxes-masks/vindr_pcxr.md).
- **Chest ImaGenome.** [PhysioNet](https://physionet.org/content/chest-imagenome/1.0.0/). The images come from
  MIMIC-CXR. See [chest_imagenome](../reference/datasets/cxr-boxes-masks/chest_imagenome.md).
- **MS-CXR and MS-CXR-T.** [MS-CXR](https://physionet.org/content/ms-cxr/) and
  [MS-CXR-T](https://physionet.org/content/ms-cxr-t/). Both need MIMIC-CXR for the images. See
  [ms_cxr](../reference/datasets/cxr-boxes-masks/ms_cxr.md) and
  [ms_cxr_t](../reference/datasets/cxr-labels/ms_cxr_t.md).
- **QaTa-COV19 v2.** Kaggle dataset `aysendegerli/qatacov19-dataset`. It needs a Kaggle
  account, as the other Kaggle downloads do. See
  [qata_cov19](../reference/datasets/cxr-boxes-masks/qata_cov19.md).
- **MedSeg COVID-19 CT.** [figshare author page](https://figshare.com/authors/MedSeg/9940190). See
  [medseg_covid_ct](../reference/datasets/ct-mri-other/medseg_covid_ct.md).
- **VQA-RAD.** [OSF 89kps](https://osf.io/89kps/). CC0 license. See
  [vqa_rad](../reference/datasets/vqa-text/vqa_rad.md).
- **CheXinstruct VQA.** There is no single download. You compile it from several sources. See
  [chexinstruct_vqa](../reference/datasets/vqa-text/chexinstruct_vqa.md).
- **GEMeX-VQA.** [Hugging Face `BoKelvin/GEMeX-VQA`](https://huggingface.co/datasets/BoKelvin/GEMeX-VQA). The
  images come from MIMIC-CXR-JPG. See [gemex_vqa](../reference/datasets/vqa-text/gemex_vqa.md).
- **MIMIC-Ext-CXR-QBA.** [PhysioNet](https://physionet.org/content/mimic-ext-cxr-qba/1.0.0/). See
  [mimic_ext_cxr_qba](../reference/datasets/vqa-text/mimic_ext_cxr_qba.md).
- **ROCO.** Clone [razorx89/roco-dataset](https://github.com/razorx89/roco-dataset) and run
  `scripts/fetch.py`. See [roco](../reference/datasets/vqa-text/roco.md).
- **ROCOv2.** [Zenodo record 8333645](https://zenodo.org/records/8333645). Hugging Face copy:
  `eltorio/ROCOv2-radiology`. See [roco](../reference/datasets/vqa-text/roco.md).
- **MedPMC.** [Hugging Face
  `Yale-BIDS-Chen/medpmc-11m-dataset_jun24_baseline`](https://huggingface.co/datasets/Yale-BIDS-Chen/medpmc-11m-dataset_jun24_baseline).
  The license is CC BY-NC-SA 4.0. See [MedPMC](../reference/datasets/vqa-text/medpmc.md).

## Check

`ls <base_image_dir> | head` shows image files. The dataset page says which folder is `base_image_dir`.

## See also

- [Dataset catalog](../reference/datasets/index.md): the class, the outputs and the page of each dataset.
- [Load a dataset and pick outputs](load-a-dataset.md)
