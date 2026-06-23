# RadHarmony — Remote Access (SSH Tunnel)

How to bring up the Gradio app on a remote machine and reach it from your laptop when a network sits between you and the host that blocks high ports — typical for VPN-behind-corporate-firewall setups where SSH (22) is allowed but `7860` is not.

If your laptop and the host are on the same flat network and you can already open `http://<remote-host>:7860` in a browser, you do not need this document — just run `python app.py` and visit that URL.

---

## TL;DR

On the remote machine:

```bash
cd ~/path/to/RadHarmony
uv sync                          # only when pyproject.toml changed
uv run python app.py             # binds 0.0.0.0:7860
```

On your laptop (separate terminal):

```bash
ssh -v -N -L 17860:localhost:7860 <user>@<remote-host>
```

Open **`http://localhost:17860`** in your browser. Stop the tunnel with `Ctrl+C`.

---

## 1. Bring up the server

### Check for stale gradio servers first

Gradio binds port 7860 by default. If a previous session is still running, the new launch will fail (or quietly land on 7861). List anything currently bound:

```bash
ss -lntp | grep -E ':(7860|7861|7862)'
```

Each line shows `pid=NNNN` for the owner. To kill a specific one:

```bash
kill <pid>
```

If you do not own the process, do not kill it — pick a different port instead by editing `app.py:demo.launch(...)` to take a different `server_port`.

### Sync the venv (if dependencies changed)

`uv sync` is a no-op when `uv.lock` already matches `pyproject.toml`, so it is safe to run every time:

```bash
uv sync
```

### Launch

```bash
uv run python app.py
```

You should see Gradio print `Running on local URL: http://0.0.0.0:7860` within ~10 s. Leave the terminal open — when you `Ctrl+C` it, the app stops.

For a longer-lived session, run it inside `tmux` or `screen` so it survives SSH disconnects.

---

## 2. SSH tunnel from your laptop

When the network blocks port 7860 from your laptop to the remote host (test with `curl -m 3 http://<remote-host>:7860` — "connection refused" or hanging means blocked), but SSH on 22 is allowed, forward 7860 over the SSH session:

```bash
ssh -v -N -L 17860:localhost:7860 <user>@<remote-host>
```

Flag breakdown:

| Flag | Why |
|---|---|
| `-N` | Hold the tunnel without opening an interactive shell. |
| `-v` | Print the bind result so you can see "Local connections to LOCALHOST:17860 forwarded to remote address localhost:7860" — confirms the tunnel is real. |
| `-L 17860:localhost:7860` | Forward laptop port 17860 → server-side `localhost:7860`. |

Open **`http://localhost:17860`** in your browser.

### Why port 17860 and not 7860?

If your laptop already has a local Gradio on 7860 (very common during dev), `ssh -L 7860:...` will fail to bind the local side and print `bind: Address already in use` — but **the SSH session stays up regardless**. Without `-v` you cannot tell the difference between a working tunnel and a silently-collided one. Your browser then hits your local Gradio and you think you are looking at the server.

Using a distinct local port (17860, 18000, anything free) eliminates the ambiguity.

### Why "no acceptance message" after entering your password is normal

`-N` means "do not run a remote command, just hold the tunnel." There is no shell prompt, no banner, no "Welcome" line. The hang **is** the success state. With `-v` added you do get progress output ("Authenticated to ...", "Entering interactive session"), which is the easiest positive confirmation.

---

## 3. Verify you are hitting the server, not your local instance

Two quick checks:

1. **`-v` ssh output** must contain a line like `Local connections to LOCALHOST:17860 forwarded to remote address localhost:7860`. If it instead says `bind [127.0.0.1]:17860: Address already in use`, the tunnel did not bind — pick a different local port.
2. **Look for a recent change in the UI** that exists only on the server. For example, after merging a new dataset, its entry under the right modality tab is visible only in the server build until your local checkout pulls.

---

## Alternative: `share=True` (gradio.live tunnel)

`gradio.Blocks.launch(share=True)` opens a public `*.gradio.live` HTTPS tunnel that bypasses the firewall entirely. Use it only when SSH forwarding is not an option, because:

- The URL is public — anyone with the link reaches your app and the absolute filesystem paths in the UI become visible.
- The link expires after ~72 h.
- The URL is printed once at startup; if you backgrounded the process without capturing stdout, you have lost it and need to relaunch.

To enable temporarily, edit `app.py`:

```python
demo.launch(server_name="0.0.0.0", share=True)
```

Revert before committing.

---

## Stopping everything

| What | How |
|---|---|
| Tunnel | `Ctrl+C` in the `ssh -N` terminal. |
| App | `Ctrl+C` in the `uv run python app.py` terminal. If it was backgrounded, find the PID with `ss -lntp \| grep 7860` and `kill <pid>`. |
| Stray gradio processes | `pkill -f 'python app.py'` (kills every match — review with `pgrep -af 'python app.py'` first). |

---

## Personal cheatsheet

If this repository is your daily driver, copy `docs/remote_access.md` to `docs/local/remote_access.md` and replace the placeholders with your concrete `<user>`, `<remote-host>`, and preferred local port. `docs/local/` is in `.gitignore`, so the personal version stays on your machine and never leaks into commits.
