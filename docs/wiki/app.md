# App Guide

The RadHarmony Gradio app lets you explore any supported dataset visually — browse samples, overlay bounding boxes and masks, and generate the transform code snippet for your training pipeline.

## Launch

```bash
# Local
python app.py

# With uv
uv run python app.py
```

The app starts at **http://localhost:7860** and listens on `0.0.0.0:7860` (all interfaces), so it is reachable from other machines on the same network.

---

## UI walkthrough

### 1. Modality tabs

The top-level tabs group datasets by imaging modality: **CXR**, **Radiograph** <sup class="beta">beta</sup>, **CT** <sup class="beta">beta</sup>, **MRI** <sup class="beta">beta</sup>, **VQA** <sup class="beta">beta</sup> (vision-question-answer), **Other**. Select the tab for the modality you want to explore.

### 2. Dataset selector

Within each tab, a dropdown lists the available datasets. Selecting one reveals its input panel.

### 3. Input panel

Each dataset has up to four input fields:

| Field | Description |
|-------|-------------|
| **Base directory** | Root path to the images (required) |
| **CSV path** | Primary metadata CSV — leave blank for auto-discovery |
| **Extra field** | Dataset-specific (e.g. BBox CSV, Label JSON, Series description CSV) |
| **Extra field 2** | Some datasets have a second extra field (e.g. MIMIC-CXR report CSV) |

Some datasets also show a **dropdown** for additional filtering (e.g. RSNA Lumbar Spine series type, TAIX-Ray label mode, RSNA Bone Age split).

Click **Load dataset** after filling in the fields.

### 4. Output flags

Checkboxes control which outputs are requested:

- **Classification labels** (`output_cls`)
- **Bounding boxes** (`output_bbox`)
- **Segmentation mask** (`output_mask`)
- **Radiology report** (`output_report`)

Only flags supported by the selected dataset are shown.

### 5. Sample browser

After loading, use **Previous** / **Next** to step through samples. Each sample shows:

- The image (with bbox overlay if bounding boxes are loaded)
- Mask overlay (if masks are loaded)
- Classification label list with values
- Report text (if reports are loaded)

### 6. Transform builder

The sidebar transform builder lets you toggle augmentations interactively:

- Flip, Affine, Intensity jitter, Gaussian noise, Gaussian smooth, Elastic deformation
- Adjust probability and range sliders
- Click **Copy code** to get the Python snippet for your training script

---

## Remote access

If the app runs on a remote server and you want to access it from your laptop:

**On the server:**

```bash
python app.py   # already listening on 0.0.0.0:7860
```

**On your local machine:**

```bash
ssh -N -L 17860:localhost:7860 user@server-hostname
```

Then open **http://localhost:17860** in your browser.

The `-N` flag keeps the tunnel open without starting a shell. To run it in the background, add `-f`:

```bash
ssh -f -N -L 17860:localhost:7860 user@server-hostname
```

### VPN / jump host

If the server is behind a VPN or jump host:

```bash
ssh -N -L 17860:localhost:7860 -J jumphost user@server-hostname
```

### Persistent tunnel with autossh

```bash
autossh -M 0 -f -N -L 17860:localhost:7860 user@server-hostname
```

`autossh` automatically restarts the tunnel if the connection drops.
