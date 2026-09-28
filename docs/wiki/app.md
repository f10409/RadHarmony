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

The **Labels** box shows every label whose `cls` value is positive (`> 0.5`).
For datasets with ordinal severity grades (e.g. TAIX-Ray with
`label_mode="ordinal"`), values `≥ 2` are surfaced inline as `name (N)` — e.g.
`pleural_effusion_left (3)`. Plain `name` (no parens) means severity 1, identical
to a binary label being on. Standard binary datasets (CheXpert, MIMIC, etc.) are
unaffected since their `cls` values never exceed 1.

### 6. Transform builder

The sidebar transform builder lets you toggle augmentations interactively:

- Flip, Affine, Intensity jitter, Gaussian noise, Gaussian smooth, Elastic deformation
- Adjust probability and range sliders
- Click **Copy code** to get the Python snippet for your training script

---

## Remote access (SSH tunnel)

When the app runs on a remote machine and a network between you and the host
blocks high ports — typical for VPN-behind-corporate-firewall setups where SSH
(22) is allowed but `7860` is not — forward the port over SSH.

If your laptop and the host are on the same flat network and you can already
open `http://<remote-host>:7860` in a browser, you do not need any of this —
just run `python app.py` and visit that URL.

### TL;DR

On the remote machine:

```bash
cd ~/path/to/RadHarmony
uv pip install -e ".[all]"       # only when pyproject.toml changed
.venv/bin/python app.py          # binds 0.0.0.0:7860
```

On your laptop (separate terminal):

```bash
ssh -v -N -L 17860:localhost:7860 <user>@<remote-host>
```

Open **`http://localhost:17860`** in your browser. Stop the tunnel with `Ctrl+C`.

### Check for stale gradio servers first

Gradio binds port 7860 by default. If a previous session is still running, the
new launch will fail (or quietly land on 7861). List anything currently bound:

```bash
ss -lntp | grep -E ':(7860|7861|7862)'
```

Each line shows `pid=NNNN` for the owner. `kill <pid>` to stop one you own — if
you do not own the process, pick a different port instead by passing a different
`server_port` to `demo.launch(...)` in `app.py`.

### The SSH tunnel, explained

```bash
ssh -v -N -L 17860:localhost:7860 <user>@<remote-host>
```

| Flag | Why |
|---|---|
| `-N` | Hold the tunnel without opening an interactive shell. |
| `-v` | Print the bind result so you can confirm the tunnel is real (see below). |
| `-L 17860:localhost:7860` | Forward laptop port 17860 → server-side `localhost:7860`. |

**Why port 17860 and not 7860?** If your laptop already has a local Gradio on
7860 (common during dev), `ssh -L 7860:...` fails to bind the local side and
prints `bind: Address already in use` — but the SSH session stays up regardless.
Without `-v` you cannot tell a working tunnel from a silently-collided one, and
your browser then hits your *local* Gradio while you think you're on the server.
A distinct local port (17860, 18000, anything free) removes the ambiguity.

**Why "no acceptance message" is normal.** `-N` means "do not run a remote
command, just hold the tunnel" — no shell prompt, no banner. The hang **is** the
success state. With `-v` you get positive confirmation: a line like
`Local connections to LOCALHOST:17860 forwarded to remote address localhost:7860`.
If it instead says `bind [127.0.0.1]:17860: Address already in use`, the tunnel
did not bind — pick a different local port.

To run the tunnel in the background, add `-f`:

```bash
ssh -f -N -L 17860:localhost:7860 <user>@<remote-host>
```

### VPN / jump host

If the server is reachable only through a jump host:

```bash
ssh -N -L 17860:localhost:7860 -J jumphost <user>@<remote-host>
```

### Persistent tunnel with autossh

```bash
autossh -M 0 -f -N -L 17860:localhost:7860 <user>@<remote-host>
```

`autossh` automatically restarts the tunnel if the connection drops.

### Alternative: `share=True` (gradio.live tunnel)

`demo.launch(server_name="0.0.0.0", share=True)` opens a public `*.gradio.live`
HTTPS tunnel that bypasses the firewall entirely. Use it only when SSH
forwarding is not an option, because the URL is public (anyone with the link
reaches your app and the filesystem paths shown in the UI), it expires after
~72 h, and it is printed only once at startup. Revert before committing.

### Stopping everything

| What | How |
|---|---|
| Tunnel | `Ctrl+C` in the `ssh -N` terminal. |
| App | `Ctrl+C` in the `uv run python app.py` terminal. If backgrounded, find the PID with `ss -lntp \| grep 7860` and `kill <pid>`. |
| Stray gradio processes | `pkill -f 'python app.py'` (review with `pgrep -af 'python app.py'` first — it kills every match). |
