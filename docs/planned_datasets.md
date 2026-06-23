# Planned datasets

Dataset integrations that are scoped but not yet implemented. Each entry
captures the key facts needed to start scaffolding (per the add-dataset skill
in `docs/add_dataset_skill.md`).

The CXR datasets below are the unintegrated entries from the chest-radiograph
shortlist that motivated the `integration-cxr` branch. RANZCR CLiP (Royal
Australian and New Zealand College of Radiologists catheter & line challenge)
was the first to land — see `docs/dataset_guide.md`.

## CXR shortlist — not yet integrated

### PadChest

- **Source:** BIMCV — <http://bimcv.cipf.es/bimcv-projects/padchest/> (registration required).
- **Reference:** Bustos et al., *PadChest: A large chest X-ray image dataset with multi-label annotated reports.* Medical Image Analysis 2020.
- **Modality:** CXR (PNG, 16-bit grayscale).
- **Samples:** ~160k images, ~67k patients, 174 radiographic findings + 19 differential diagnoses.
- **License:** non-commercial research.
- **Notes:** Hierarchical label vocabulary (UMLS-mapped); a pragmatic v1 should select the ~18-label subset that overlaps with CheXpert / NIH-14. Spanish reports are available — wire `output_report=True` if a translation pipeline is in scope.

### BIMCV-COVID-19+

- **Source:** BIMCV — <http://bimcv.cipf.es/bimcv-projects/bimcv-covid19/> (registration + DUA).
- **Reference:** Vayá et al., *BIMCV COVID-19+: a large annotated dataset of RX and CT images of COVID-19 patients.* arXiv:2006.01174.
- **Modality:** CXR (DICOM) + CT (DICOM); start with the CXR subset.
- **Samples:** Iteration-2 release: ~23k CXR studies, ~9k patients, with structured radiology reports.
- **License:** CC BY 4.0 (data) but DUA-gated.
- **Notes:** Distinct from the existing `siim_covid19` integration (which is the Kaggle-released SIIM-FISABIO-RSNA *derivative* of BIMCV). The original BIMCV release ships the full DICOM tree, full reports, and a much larger imaging set.

### BRAX (Brazilian Chest X-Ray)

- **Source:** PhysioNet — <https://brax.ai/> redirects to <https://physionet.org/content/brax/1.1.0/>.
- **Reference:** Reis et al., *BRAX: a publicly available Brazilian chest X-ray dataset.* Scientific Data 2022.
- **Modality:** CXR (DICOM, 16-bit). Includes lateral views.
- **Samples:** ~24,959 studies / ~40,967 images / 19,351 patients.
- **License:** PhysioNet Credentialed Health Data License 1.5.0.
- **Notes:** Labels are derived from CheXpert NLP labeller against Portuguese reports — schema matches CheXpert's 14-class one-hot. Make sure to honour the CheXpert uncertain-handling policy (`-1` → NaN by default).

### RANZCR (already integrated)

Done in the `integration-cxr` branch as `RANZCRClipDataset` (registry key `ranzcr_clip`). See `docs/dataset_guide.md#ranzcr-clip`.

### CANDIX — Chest X-Ray Anonymised New Zealand Dataset, Dunedin

- **Source:** <https://candix.ai/>. Dataset is gated behind an institutional access agreement.
- **Modality:** CXR (DICOM, anonymised).
- **Samples:** TBD — confirm at access-grant time.
- **Notes:** Access is the gating issue; defer scoping until a credentialed sample is in hand.

### VinDr-RibCXR

- **Source:** <https://vindr.ai/ribcxr>. Distributed **directly from VinDr.ai** behind a signed Data Use Agreement; no PhysioNet or Kaggle mirror. (Verified 2026-04-29 via WebFetch of the dataset page.)
- **Access:** DUA must be signed and sent to Ha Nguyen at the VinDr lab to obtain a download link.
- **Reference:** Nguyen, Le, Pham, Nguyen — *VinDr-RibCXR: A Benchmark Dataset for Automatic Segmentation and Labeling of Individual Ribs on Chest X-rays.* MIDL 2021.
- **Modality:** CXR (DICOM, sourced from VinDr-CXR) with **per-rib segmentation masks** (20 rib classes, left/right).
- **Samples:** 245 CXRs total — **196 train / 49 val** per the dataset page. No separate test split.
- **Task:** primarily semantic segmentation, secondarily anatomy localization.
- **Annotation format:** JSON, described as "masks of ribs" — likely polygon vertices or a serialized mask, but exact schema is not visible on the public page (DUA-gated). Confirm at integration time by inspecting the actual download.
- **Notes:** Decision point — output as a 20-channel mask (one per rib instance) or a single multi-class label image (pixel value = rib index)? `BaseRadiologicalDataset.output_mask` currently assumes a single binary mask; the multi-class label image fits the existing pipeline with the smallest base-class change.

### Object-CXR

- **Source:** <https://object-cxr.github.io/> — JF Healthcare, MIDL 2020 challenge.
- **Modality:** CXR (JPEG).
- **Samples:** 10k frontal CXRs (8k train / 1k dev / 1k test) annotated for foreign-object detection.
- **Task:** binary classification + bounding-box localization for foreign objects (jewelry, buttons, electrodes, etc.).
- **Notes:** Annotations include both bbox and polygon. v1 should expose `output_cls` + `output_bbox`; polygon support can be deferred. The dataset host is no longer maintained; download mirrors are listed in the GitHub README.

### ChestX-Det

- **Source:** <https://github.com/Deepwise-AILab/ChestX-Det-Dataset>.
- **Reference:** Liu et al., *ChestX-Det10/ChestX-Det.* (multiple releases; ChestX-Det extends ChestX-Det10 from 10 to 13 categories).
- **Modality:** CXR (PNG); a re-annotated subset of NIH ChestX-ray14.
- **Samples:** 3,578 CXRs with **instance-level polygon masks** for 13 abnormality classes.
- **Task:** segmentation + multi-label classification.
- **Notes:** Polygon → mask rasterization mirrors the RANZCR pattern; reuse the `_decode_mask` polyline helper. Class taxonomy overlaps with ChestX-ray14 — flag whether to merge or keep distinct.

### RALO — Radiographic Assessment of Lung Opacity

- **Source:** <https://stanfordaimi.azurewebsites.net/datasets/0bef0c12-6cf3-4d3c-a2c9-a9b7f4e9f2cb> (Stanford AIMI catalogue; the `ralo-dataset.org` URL no longer resolves).
- **Reference:** Cohen et al., *Radiographic Assessment of Lung Opacity Score Dataset.* Stanford AIMI 2022.
- **Modality:** CXR (DICOM).
- **Samples:** 2,373 portable CXRs with paired RALO scores (geographic extent + opacity density, each 0–4).
- **Task:** ordinal regression on two correlated targets — natural fit for the new `output_reg=True` plumbing landed for RSNA Bone Age.
- **Notes:** Dual ordinal targets — extend `REG_COLS` to `["geographic_extent", "opacity_density"]`. Confirm DICOM access through Stanford AIMI's research-data portal.

### MedFMC

- **Source:** <https://medfmc.github.io/> — MICCAI 2023 few-shot learning challenge.
- **Modality:** mixed (CXR + endoscopy + dermatology + retinal). The CXR subset is a re-released slice of ChestDR.
- **Samples:** thousands of images per task; few-shot prompt format (5 / 10 / 50 labelled per class).
- **Task:** few-shot multi-label classification benchmark.
- **Notes:** Lower priority — this is a *benchmark* layered over existing data, not a new image source. Wire only after the underlying ChestDR / ChestX-Det data is in.

## Pipeline-level decisions still pending

- **Polygon / instance masks.** Object-CXR, ChestX-Det, and VinDr-RibCXR all need either polygon→raster decoding (extend the polyline helper from RANZCR) or per-instance mask channels. The base class currently assumes a single binary mask per row.
- **Dual / multi-target regression.** RALO needs two correlated ordinal scores. `REG_COLS` already supports a list — confirm the transform pipeline serialises a length-`len(REG_COLS)` tensor under `reg`.
- **DUA-gated downloads.** PadChest, BIMCV-COVID-19+, BRAX, and CANDIX all require credentialed access. Add a one-page `docs/access_notes.md` (or extend `remote_access.md`) capturing where each dataset's access agreement lives once a credentialed copy is on the lab NAS.
