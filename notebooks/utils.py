import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

##############################################################################################################
### function for loading the provided data from the npz file
# used for reconstructing the nested structure from the flattened npz file

def load_data(path="../data"):
    raw = dict(np.load(Path(path) / "visual_coding_data.npz", allow_pickle=True))
    # Initialize the hierarchical dictionary.
    # matched_cell_ids tracks longitudinally recorded ROIs across sessions.
    data = {
        "matched_cell_ids": raw["matched_cell_ids"],
        "templates": {},
        "sessions": {},
    }
    # Helper function to initialize or retrieve a session sub-dictionary.
    def session(L):
        return data["sessions"].setdefault(L, {"stim_tables": {}})
    # Iterate through the flat dictionary. The keys encode the hierarchy using '__'.
    for key, val in raw.items():
        if key == "matched_cell_ids":
            continue # Already handled during initialization.
        parts = key.split("__")
        # Route template data (e.g., spatial footprints or stimulus templates).
        if parts[0] == "tmpl":
            data["templates"][parts[1]] = val
            continue
        # L represents the session identifier.
        L = parts[0]
        s = session(L)
        # Reconstruct stimulus tables.
        # Example key: SessionID__stim__drifting_gratings__values
        if parts[1] == "stim" and parts[3] == "values":
            stim = parts[2]
            # Fetch the corresponding column names for this specific stimulus array.
            cols = list(raw[f"{L}__stim__{stim}__cols"])
            s["stim_tables"][stim] = pd.DataFrame(val, columns=cols)
        # Reconstruct epoch tables (blocks of time with continuous stimulus types).
        elif parts[1] == "epoch" and parts[2] == "values":
            cols = list(raw[f"{L}__epoch__cols"])
            s["stim_epoch_table"] = pd.DataFrame(val, columns=cols)
        # Extract scalar metadata.
        elif parts[1] in ("session_type",):
            s["session_type"] = val.item() if hasattr(val, "item") else val
        # Route core timeseries and spatial matrices directly.
        # t: time vector, dff: deltaF/F calcium traces, roi_masks: spatial footprints.
        elif parts[1] in ("t", "dff", "roi_masks", "max_projection", "running_speed"):
            s[parts[1]] = val
    return data

##############################################################################################################
### function for printing some data information
# for verification of dimensionality, and integrity of the data structure

def print_info(data):
    print(f"matched cells: {len(data['matched_cell_ids'])}")
    print(f"templates: {list(data['templates'])}")
    for L, s in sorted(data["sessions"].items()):
        print(f"\nsession {L} ({s.get('session_type')})")
        print(
            f"  t: {s['t'].shape}, dff: {s['dff'].shape}, roi_masks: {s['roi_masks'].shape}"
        )
        for name, df in s["stim_tables"].items():
            print(f"  stim '{name}': {df.shape} cols={list(df.columns)}")

##############################################################################################################
##############################################################################################################
### For visualization of the data

def plot_field_of_view(data, session="A", ax=None):
    s = data["sessions"][session]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(s["max_projection"], cmap="gray")
    ax.imshow(
        np.ma.masked_where(s["roi_masks"].sum(0) == 0, s["roi_masks"].sum(0)),
        cmap="autumn",
        alpha=0.5,
    )
    ax.set_title(f"session {session}: field of view + ROIs")
    ax.axis("off")
    return ax


def plot_traces(data, session="A", cells=(0, 1, 2)):
    s = data["sessions"][session]
    fig, axes = plt.subplots(
        len(cells),
        1,
        figsize=(10, 1.6 * len(cells)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cells):
        ax.plot(s["t"], s["dff"][c], lw=0.5)
        ax.set_ylabel(f"cell {c}\nΔF/F")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"session {session}: example traces")
    return fig


def plot_running_speed(data, session="A", ax=None):
    s = data["sessions"][session]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2))
    ax.plot(s["running_speed"][1], s["running_speed"][0], lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("running speed\n(cm/s)")
    ax.set_title(f"session {session}: running speed")
    return ax


def show_stimulus_examples(data):
    templates = data["templates"]
    fig, axes = plt.subplots(
        1, len(templates), figsize=(4 * len(templates), 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, (name, tmpl) in zip(axes, templates.items()):
        ax.imshow(tmpl[0], cmap="gray")
        ax.set_title(f"{name}\n(frame 0, template {tmpl.shape})")
        ax.axis("off")
    return fig









