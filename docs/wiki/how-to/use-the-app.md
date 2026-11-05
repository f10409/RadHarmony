# Use the app

<!-- blocks: run=skip reason="starts the app server or an SSH tunnel" -->

The RadHarmony app is a Gradio web page. Use it to look at the samples of a dataset, with the boxes and
masks on the image. You can also try the transform steps before you train, and copy their Python code.

Before you start, install the `app` extra in your venv: `uv pip install -e ".[app]"`.

## 1. Start the app

```bash
python app.py
```

The app listens on `0.0.0.0:7860` (all network interfaces). Open `http://localhost:7860` on the same
machine. Other machines on the same network can open `http://<host>:7860`. For another port, run
`python app.py --server-port 7861`.

## 2. Load a dataset

1. Pick the tab of the modality: **CXR**, **Radiograph**, **CT**, **CT (2D)**, **MRI** or **VQA**.
2. Pick the **Dataset family**. A family with more than one class also shows a **Variant** list, for
   example the train and test split.
3. Type the **Base image directory**. The dataset page says which folder it is.
4. If the dataset needs other files, type their paths in **CSV path** and in up to 2 extra fields. An
   extra field holds, for example, a box CSV or a report CSV. Some datasets also show a drop-down list,
   for example the label mode of TAIX-Ray.
5. Type a **Cache dir**. Leave it blank for no cache.
6. Pick the **Output fields**: **Labels** (`output_cls`), **Mask** (`output_mask`), **Report**
   (`output_report`), **BBox** (`output_bbox`) and **Reg** (`output_reg`). **BBox** is on by default
   for NODE21, ChestX-Det10, PadChest-GR and Chest ImaGenome (D-11).
7. Click **Load Dataset**. The **Status** box shows the number of samples and the keys, for example
   `Loaded 138 samples.  Keys: ['cls', 'img']`.

## 3. Set the steps and run

1. Under **Preprocessing & Display**, set the image size and the pad. A 3-D tab also has the voxel
   spacing. The CT tab also has the HU window.
2. Under **Augmentations**, turn on the steps that you want to see: Flip, Intensity Jitter, Gaussian Noise,
   Gaussian Smooth, Elastic Deformation, Affine and Transpose. In the app, each step that is on always runs
   (probability 1.0).
3. Click **Run on random sample**. **Run on same sample** runs the steps again on the last sample.

The gallery shows the **Original** image first, then each augmentation alone. With 2 or more steps on, the
last image (**Compound**) has all of them. A 3-D tab shows the middle axial, coronal and sagittal slices.

Under the gallery:

- **Labels** shows each label with a `cls` value above 0.5. A value of 2 or more shows as `name (N)`, for
  example `pleural_effusion_left (3)` for TAIX-Ray with `label_mode="ordinal"`. `name` alone means 1.
  A regression value shows as `name=value`.
- **Report** and **Q & A** show the text of the sample.
- **Transform code snippet** gives the `RadiologyTransform2D` or `RadiologyTransform3D` code of your
  settings. In the code, each step has its normal probability, for example `prob=0.5`.

<a id="remote-access-ssh-tunnel"></a>

## 4. Open the app from another machine

Do this step when the app runs on a remote machine and a firewall or VPN blocks port 7860, but allows SSH
(port 22). If your browser can open `http://<remote-host>:7860`, you do not need it.

1. On the remote machine, start the app: `python app.py`.
2. On your laptop, in a second terminal, open an SSH tunnel:

   ```bash
   ssh -v -N -L 17860:localhost:7860 <user>@<remote-host>
   ```

3. Open `http://localhost:17860` in your browser.
4. To stop the tunnel, press `Ctrl+C` in its terminal. To stop the app, press `Ctrl+C` in the terminal of
   the app.

Notes:

- `-L 17860:localhost:7860` sends laptop port 17860 to port 7860 of the server. Use a free local port.
  A local app on port 7860 of your laptop blocks the local side of a `-L 7860:...` tunnel, and your
  browser then shows the local app.
- `-N` opens no remote shell, so the terminal stays silent while the tunnel works. With `-v`, SSH prints
  `Local connections to LOCALHOST:17860 forwarded to remote address localhost:7860`. If it prints
  `bind [127.0.0.1]:17860: Address already in use`, pick another local port.
- An old app on port 7860 stops the new one, or the new one starts on port 7861.
  `ss -lntp | grep -E ':(7860|7861|7862)'` shows the process IDs. Stop your own with `kill <pid>`, or
  start the app with `--server-port`.
- Add `-f` to run the tunnel in the background. Through a jump host, add `-J jumphost`. For a tunnel that
  starts again after a lost connection, use
  `autossh -M 0 -f -N -L 17860:localhost:7860 <user>@<remote-host>`.
- `demo.launch(server_name="0.0.0.0", share=True)` in `app.py` gives a public `*.gradio.live` link that
  the firewall does not block. Anyone with the link can use your app and see its file paths. The link
  stops after about 72 hours. Use it only when SSH is not possible, and undo the change before you
  commit.

## Check

After **Load Dataset**, the **Status** box shows `Loaded <N> samples`. After **Run on random sample**, the
gallery shows the **Original** image.

## See also

- [Load a dataset and pick outputs](load-a-dataset.md): the same steps in Python.
- [Transforms](../reference/transforms.md): the steps of the transform code snippet.
