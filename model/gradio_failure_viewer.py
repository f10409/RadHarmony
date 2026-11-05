"""Gradio app for exploring TAIX-Ray linear probe failure cases.

Shows test samples where all three backbone models (CheXagent, MedSigLIP,
BiomedCLIP) failed, with filters for label, failure type (FP/FN), sorting
by case difficulty, and per-image comments saved to a JSON file.

Usage
-----
  cd model
  RADHARMONY_OUTPUT_DIR=/path/to/outputs .venv/bin/python gradio_failure_viewer.py [--port 7860] [--share]

It reads the output of analyze_failures.py in <RADHARMONY_OUTPUT_DIR>/taix_failure_analysis
(default: ./outputs/taix_failure_analysis).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image

# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.environ.get("RADHARMONY_OUTPUT_DIR") or "./outputs", "taix_failure_analysis")
FULL_PARQUET  = os.path.join(DATA_DIR, "taix_lp_predictions.parquet")
COMMENTS_FILE = os.path.join(DATA_DIR, "reviewer_comments.json")

LABELS = [
    "atelectasis_left", "atelectasis_right", "heart_size",
    "pleural_effusion_left", "pleural_effusion_right",
    "pulmonary_congestion", "pulmonary_opacities_left", "pulmonary_opacities_right",
]
BACKBONES = ["chexagent", "medsiglip", "biomedclip"]

TAGS = ["label error", "genuinely hard", "image quality", "interesting pattern", "other"]

# ── data loading ─────────────────────────────────────────────────────────────
print("Loading predictions parquet …")
DF_FULL = pd.read_parquet(FULL_PARQUET)
print(f"  {len(DF_FULL):,} test samples loaded")

# ── comment store ─────────────────────────────────────────────────────────────
_comments_lock = threading.Lock()

def _load_comments() -> dict:
    if os.path.exists(COMMENTS_FILE):
        with open(COMMENTS_FILE) as f:
            return json.load(f)
    return {}

def _save_comments(store: dict) -> None:
    with open(COMMENTS_FILE, "w") as f:
        json.dump(store, f, indent=2)

def get_comment(study_id: str) -> tuple[str, list[str]]:
    store = _load_comments()
    entry = store.get(study_id, {})
    return entry.get("text", ""), entry.get("tags", [])

def is_reviewed(study_id: str) -> bool:
    store = _load_comments()
    return store.get(study_id, {}).get("reviewed", False)

def save_comment(study_id: str, text: str, tags: list[str], focus_label: str) -> str:
    with _comments_lock:
        store = _load_comments()
        entry = store.get(study_id, {})
        entry.update({
            "text": text,
            "tags": tags,
            "focus_label": focus_label,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        store[study_id] = entry
        _save_comments(store)
    n_total = sum(1 for v in store.values() if v.get("text") or v.get("tags"))
    return f"✓ Saved ({n_total} total comments)"

def mark_reviewed(study_id: str) -> None:
    with _comments_lock:
        store = _load_comments()
        entry = store.get(study_id, {})
        entry["reviewed"] = True
        entry["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        store[study_id] = entry
        _save_comments(store)

def export_comments_df() -> pd.DataFrame:
    store = _load_comments()
    if not store:
        return pd.DataFrame(columns=["study_id", "reviewed", "tags", "text", "focus_label", "updated_at"])
    rows = [{"study_id": sid, **v} for sid, v in store.items()]
    df = pd.DataFrame(rows)
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(lambda t: ", ".join(t) if isinstance(t, list) else t)
    return df.sort_values("updated_at", ascending=False, na_position="last")


# ── helpers ──────────────────────────────────────────────────────────────────

def get_filtered_df(focus_label, failure_type, view_subset, sort_col, hide_reviewed) -> pd.DataFrame:
    df = DF_FULL.copy()

    if hide_reviewed:
        store = _load_comments()
        reviewed_ids = {sid for sid, v in store.items() if v.get("reviewed", False)}
        if reviewed_ids:
            df = df[~df["study_id"].isin(reviewed_ids)]

    if view_subset == "All models failed on label":
        df = df[df[f"all_fail_{focus_label}"] == 1]
    elif view_subset == "Any model failed":
        df = df[sum(df[f"err_{b}_{focus_label}"] for b in BACKBONES) > 0]
    elif view_subset == "All correct":
        df = df[sum(df[f"err_{b}_{focus_label}"] for b in BACKBONES) == 0]

    if failure_type == "FP (false alarm)":
        df = df[df[f"true_{focus_label}"] == 0]
    elif failure_type == "FN (missed finding)":
        df = df[df[f"true_{focus_label}"] == 1]

    if sort_col == "Hardest first (most labels failed)":
        df = df.sort_values("n_labels_all_fail", ascending=False)
    elif sort_col == "Commented first":
        store = _load_comments()
        df["_has_comment"] = df["study_id"].isin(store).astype(int)
        df = df.sort_values(["_has_comment", "n_labels_all_fail"], ascending=[False, False])
        df = df.drop(columns=["_has_comment"])
    else:
        df = df.sample(frac=1, random_state=42)

    return df.reset_index(drop=True)


def load_raw_image(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        return np.array(Image.open(path)).astype(np.float32)
    except Exception:
        return None


def apply_window(arr: np.ndarray | None, lo_pct: float, hi_pct: float) -> Image.Image | None:
    if arr is None:
        return None
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    windowed = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    return Image.fromarray((windowed * 255).astype(np.uint8)).convert("RGB")


def make_pred_table(row: pd.Series, focus_label: str) -> pd.DataFrame:
    records = []
    for lbl in LABELS:
        true_val = int(row[f"true_{lbl}"])
        errs = {b: bool(row[f"err_{b}_{lbl}"]) for b in BACKBONES}

        def fmt(b, errs=errs):
            mark = "✗" if errs[b] else "✓"
            pred = "pos" if int(row[f"pred_{b}_{lbl}"]) else "neg"
            prob = float(row[f"prob_{b}_{lbl}"])
            return f"{mark} {pred} ({prob:.2f})"

        records.append({
            "Label": ("► " if lbl == focus_label else "  ") + lbl.replace("_", " "),
            "True": "pos" if true_val else "neg",
            "CheXagent": fmt("chexagent"),
            "MedSigLIP": fmt("medsiglip"),
            "BiomedCLIP": fmt("biomedclip"),
            "All fail?": "❌" if int(row[f"all_fail_{lbl}"]) else "",
        })
    return pd.DataFrame(records)


def make_summary_stats() -> pd.DataFrame:
    n = len(DF_FULL)
    records = []
    for lbl in LABELS:
        n_fail = int(DF_FULL[f"all_fail_{lbl}"].sum())
        fp = int(((DF_FULL[f"true_{lbl}"] == 0) & (DF_FULL[f"all_fail_{lbl}"] == 1)).sum())
        fn = int(((DF_FULL[f"true_{lbl}"] == 1) & (DF_FULL[f"all_fail_{lbl}"] == 1)).sum())
        records.append({
            "Label": lbl.replace("_", " "),
            "All-fail %": f"{n_fail/n*100:.1f}%",
            "FP": fp, "FN": fn,
            "Prev %": f"{DF_FULL[f'true_{lbl}'].mean()*100:.1f}%",
        })
    return pd.DataFrame(records)


SUMMARY_STATS = make_summary_stats()


# ── render ────────────────────────────────────────────────────────────────────

def render_case(filtered_df, idx, focus_label, lo_pct=1.0, hi_pct=99.0):
    """Return all UI values for one case."""
    if filtered_df is None or len(filtered_df) == 0:
        empty = (None, None, pd.DataFrame(), "0 / 0", "*No cases match.*",
                 "", [], "", gr.update(interactive=False), gr.update(interactive=False))
        return empty

    idx = max(0, min(idx, len(filtered_df) - 1))
    row = filtered_df.iloc[idx]
    sid = row["study_id"]

    raw   = load_raw_image(row["image_path"])
    img   = apply_window(raw, lo_pct, hi_pct)
    table = make_pred_table(row, focus_label)
    store = _load_comments()
    n_commented = sum(1 for v in store.values() if v.get("text") or v.get("tags"))
    n_reviewed  = sum(1 for v in store.values() if v.get("reviewed", False))

    reviewed_badge = "  ✅ **Reviewed**" if store.get(sid, {}).get("reviewed", False) else ""
    counter = f"{idx + 1} / {len(filtered_df)}  ({n_commented} commented · {n_reviewed} reviewed)"
    meta = (
        f"**Study:** `{sid}`  "
        f"| **Age:** {int(row['age']) // 365}y  "
        f"| **Sex:** {row['sex']}  "
        f"| **Labels failing all models:** {int(row['n_labels_all_fail'])}/8"
        + reviewed_badge
    )

    saved_text, saved_tags = get_comment(sid)

    return (
        img, raw, table, counter, meta,
        saved_text, saved_tags,
        "",                                      # status cleared on navigation
        gr.update(interactive=idx > 0),
        gr.update(interactive=idx < len(filtered_df) - 1),
    )


# ── app ───────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="TAIX Failure Viewer") as demo:
        gr.Markdown("# TAIX-Ray Linear Probe — Failure Case Viewer")
        gr.Markdown(
            "Cases where **all three models** (CheXagent, MedSigLIP, BiomedCLIP) "
            "predicted incorrectly. ✗ = wrong, ✓ = correct."
        )

        filtered_state  = gr.State(value=None)
        idx_state       = gr.State(value=0)
        raw_image_state = gr.State(value=None)

        with gr.Row():
            # ── left column ──────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Filters")
                focus_label_dd = gr.Dropdown(LABELS, value="pulmonary_congestion", label="Focus label")
                view_subset_dd = gr.Dropdown(
                    ["All models failed on label", "Any model failed", "All correct", "No filter"],
                    value="All models failed on label", label="Show cases",
                )
                failure_type_dd = gr.Dropdown(
                    ["Both", "FP (false alarm)", "FN (missed finding)"],
                    value="Both", label="Failure direction",
                )
                sort_dd = gr.Dropdown(
                    ["Hardest first (most labels failed)", "Commented first", "Random"],
                    value="Hardest first (most labels failed)", label="Sort by",
                )
                hide_reviewed_cb = gr.Checkbox(
                    value=True, label="Hide reviewed cases"
                )
                apply_btn = gr.Button("Apply", variant="primary")

                gr.Markdown("### Failure rates")
                stats_table = gr.Dataframe(value=SUMMARY_STATS, interactive=False, wrap=True)

                gr.Markdown("### Comment export")
                export_btn = gr.Button("Export all comments")
                export_df  = gr.Dataframe(label="Comments", interactive=False, wrap=True)

            # ── right column ─────────────────────────────────────────────────
            with gr.Column(scale=2):
                meta_md    = gr.Markdown("*Apply filters to start browsing.*")
                counter_md = gr.Markdown("")

                with gr.Row():
                    prev_btn = gr.Button("← Prev", interactive=False)
                    next_btn = gr.Button("Next →", interactive=False)

                image_display = gr.Image(label="Chest X-Ray", type="pil", height=800)

                with gr.Row():
                    lo_pct_slider = gr.Slider(0, 20, value=1, step=0.5, label="Dark clip %", scale=1)
                    hi_pct_slider = gr.Slider(80, 100, value=99, step=0.5, label="Bright clip %", scale=1)

                gr.Markdown("### Notes")
                comment_box = gr.Textbox(
                    label="Comment",
                    placeholder="Type your observation here…",
                    lines=3,
                )
                tag_boxes = gr.CheckboxGroup(choices=TAGS, label="Tags")
                with gr.Row():
                    save_btn          = gr.Button("Save comment", variant="primary")
                    mark_reviewed_btn = gr.Button("✅ Mark reviewed & next", variant="secondary")
                    clear_btn         = gr.Button("Clear")
                save_status = gr.Markdown("")

                pred_table_display = gr.Dataframe(
                    label="Predictions (► = focus label)",
                    interactive=False, wrap=False,
                )

        # ── outputs list shared by nav buttons ───────────────────────────────
        nav_outputs = [
            image_display, raw_image_state, pred_table_display, counter_md, meta_md,
            comment_box, tag_boxes, save_status,
            prev_btn, next_btn,
        ]

        # ── handlers ─────────────────────────────────────────────────────────

        def on_apply(focus_label, view_subset, failure_type, sort_col, hide_reviewed, lo_pct, hi_pct):
            fdf = get_filtered_df(focus_label, failure_type, view_subset, sort_col, hide_reviewed)
            return (fdf, 0) + render_case(fdf, 0, focus_label, lo_pct, hi_pct)

        def on_prev(fdf, idx, focus_label, lo_pct, hi_pct):
            new_idx = max(0, idx - 1)
            return (new_idx,) + render_case(fdf, new_idx, focus_label, lo_pct, hi_pct)

        def on_next(fdf, idx, focus_label, lo_pct, hi_pct):
            new_idx = idx + 1
            return (new_idx,) + render_case(fdf, new_idx, focus_label, lo_pct, hi_pct)

        def on_save(fdf, idx, focus_label, text, tags):
            if fdf is None or len(fdf) == 0:
                return "No case loaded."
            sid = fdf.iloc[idx]["study_id"]
            return save_comment(sid, text, tags, focus_label)

        def on_mark_reviewed(fdf, idx, focus_label, text, tags, lo_pct, hi_pct):
            if fdf is None or len(fdf) == 0:
                return (idx,) + render_case(fdf, idx, focus_label, lo_pct, hi_pct)
            sid = fdf.iloc[idx]["study_id"]
            if text or tags:
                save_comment(sid, text, tags, focus_label)
            mark_reviewed(sid)
            new_idx = min(idx + 1, len(fdf) - 1)
            result = list(render_case(fdf, new_idx, focus_label, lo_pct, hi_pct))
            result[7] = "✅ Marked as reviewed — re-apply filters to hide it"
            return (new_idx,) + tuple(result)

        def on_window(raw_arr, lo_pct, hi_pct):
            return apply_window(raw_arr, lo_pct, hi_pct)

        def on_clear():
            return "", [], ""

        def on_export():
            return export_comments_df()

        apply_btn.click(on_apply,
            inputs=[focus_label_dd, view_subset_dd, failure_type_dd, sort_dd, hide_reviewed_cb,
                    lo_pct_slider, hi_pct_slider],
            outputs=[filtered_state, idx_state] + nav_outputs)

        prev_btn.click(on_prev,
            inputs=[filtered_state, idx_state, focus_label_dd, lo_pct_slider, hi_pct_slider],
            outputs=[idx_state] + nav_outputs)

        next_btn.click(on_next,
            inputs=[filtered_state, idx_state, focus_label_dd, lo_pct_slider, hi_pct_slider],
            outputs=[idx_state] + nav_outputs)

        save_btn.click(on_save,
            inputs=[filtered_state, idx_state, focus_label_dd, comment_box, tag_boxes],
            outputs=[save_status])

        mark_reviewed_btn.click(on_mark_reviewed,
            inputs=[filtered_state, idx_state, focus_label_dd, comment_box, tag_boxes,
                    lo_pct_slider, hi_pct_slider],
            outputs=[idx_state] + nav_outputs)

        lo_pct_slider.change(on_window,
            inputs=[raw_image_state, lo_pct_slider, hi_pct_slider],
            outputs=[image_display])
        hi_pct_slider.change(on_window,
            inputs=[raw_image_state, lo_pct_slider, hi_pct_slider],
            outputs=[image_display])

        clear_btn.click(on_clear, outputs=[comment_box, tag_boxes, save_status])

        export_btn.click(on_export, outputs=[export_df])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    build_app().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
