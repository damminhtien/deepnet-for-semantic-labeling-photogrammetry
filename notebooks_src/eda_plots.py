"""Render EDA charts from aggregate tables, without opening rasters."""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  # Registers Matplotlib styles.

STYLE = ("science", "no-latex")


@plt.style.context(STYLE)
def asset_coverage(asset_summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for axis, (display_name, frame) in zip(
        axes, asset_summary.groupby("dataset"), strict=True
    ):
        pivot = frame.set_index("asset")["scenes"].sort_values()
        axis.barh(pivot.index, pivot.values, color="#4472C4")
        axis.set_title(f"{display_name}: available assets in DVC catalog")
        axis.set_xlabel("scene count")
        axis.grid(axis="x", alpha=0.25)
    return fig


@plt.style.context(STYLE)
def pooled_class_balance(pooled_balance, datasets, class_columns):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, display_name in zip(axes, datasets, strict=True):
        subset = pooled_balance[pooled_balance["dataset"] == display_name].set_index(
            "role"
        )
        subset[class_columns].plot.barh(
            stacked=True,
            ax=axis,
            color=["#8C8C8C", "#2F5597", "#5BC0DE", "#70AD47", "#FFC000", "#C00000"],
        )
        axis.set_title(f"{display_name}: semantic label mix")
        axis.set_xlabel("share of six semantic classes")
        axis.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
        axis.grid(axis="x", alpha=0.2)
    return fig


@plt.style.context(STYLE)
def per_scene_class_balance(label_counts, datasets, class_columns):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for axis, display_name in zip(axes, datasets, strict=True):
        subset = label_counts[label_counts["dataset"] == display_name].copy()
        subset["split_scene"] = subset["role"] + ":" + subset["scene_id"]
        matrix = subset[[f"share_{name}" for name in class_columns]].to_numpy()
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=0.6, cmap="magma")
        axis.set_title(f"{display_name}: per-scene class share")
        axis.set_yticks(np.arange(len(subset)), subset["split_scene"], fontsize=7)
        axis.set_xticks(np.arange(6), class_columns, rotation=40, ha="right")
        axis.set_ylabel("scene role")
        fig.colorbar(image, ax=axis, label="fraction of semantic pixels")
    return fig


@plt.style.context(STYLE)
def irrg_histograms(irrg_full_histograms, datasets):
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8), sharex=True, constrained_layout=True
    )
    for row_index, display_name in enumerate(datasets):
        for column_index, band_name in enumerate(("ir", "red", "green")):
            axis = axes[row_index, column_index]
            for role, color in (("train", "#4472C4"), ("validation", "#ED7D31")):
                rows = irrg_full_histograms.loc[
                    (irrg_full_histograms["dataset"] == display_name)
                    & (irrg_full_histograms["role"] == role)
                    & (irrg_full_histograms["band"] == band_name)
                ]
                histogram = rows.groupby("digital_number")["pixel_count"].sum()
                axis.plot(
                    histogram.index,
                    histogram / max(int(histogram.sum()), 1),
                    label=role,
                    color=color,
                )
            axis.set_title(f"{display_name} · {band_name} · all pixels")
            axis.set_xlabel("source digital number")
            axis.set_ylabel("pixel fraction")
            axis.grid(alpha=0.2)
            axis.legend()
    return fig


@plt.style.context(STYLE)
def irrg_percentiles(scene_band_statistics, datasets):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row_index, display_name in enumerate(datasets):
        for column_index, band_name in enumerate(("ir", "red", "green")):
            axis = axes[row_index, column_index]
            subset = scene_band_statistics[
                (scene_band_statistics["dataset"] == display_name)
                & (scene_band_statistics["band"] == band_name)
            ]
            for role, marker, color in (
                ("train", "o", "#4472C4"),
                ("validation", "s", "#ED7D31"),
            ):
                role_rows = subset[subset["role"] == role].reset_index(drop=True)
                positions = np.arange(len(role_rows)) + (0 if role == "train" else 0.15)
                axis.errorbar(
                    positions,
                    role_rows["p50"],
                    yerr=np.vstack(
                        [
                            role_rows["p50"] - role_rows["p05"],
                            role_rows["p95"] - role_rows["p50"],
                        ]
                    ),
                    fmt=marker,
                    markersize=3,
                    capsize=1,
                    color=color,
                    label=role,
                )
            axis.set_title(f"{display_name} · {band_name} · scene percentiles")
            axis.set_ylabel("digital number (p05–p95; center=p50)")
            axis.grid(axis="y", alpha=0.2)
            if row_index == len(datasets) - 1:
                axis.set_xlabel("Scene index within role")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2)
    return fig


@plt.style.context(STYLE)
def elevation_profiles(auxiliary_statistics, auxiliary_histograms):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    dsm_rows = auxiliary_statistics[auxiliary_statistics["asset"] == "dsm"]
    for role, marker, color in (
        ("train", "o", "#4472C4"),
        ("validation", "s", "#ED7D31"),
    ):
        role_rows = dsm_rows[dsm_rows["role"] == role].reset_index(drop=True)
        positions = np.arange(len(role_rows))
        axes[0].errorbar(
            positions,
            role_rows["p50"],
            yerr=np.vstack(
                [
                    role_rows["p50"] - role_rows["p05"],
                    role_rows["p95"] - role_rows["p50"],
                ]
            ),
            fmt=marker,
            markersize=3,
            capsize=1,
            color=color,
            label=role,
        )
    axes[0].set_title("Potsdam DSM · per-scene p05–p95")
    axes[0].set_ylabel("source value (configured as metres)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    for role, color in (("train", "#4472C4"), ("validation", "#ED7D31")):
        role_histogram = (
            auxiliary_histograms[
                (auxiliary_histograms["asset"] == "ndsm")
                & (auxiliary_histograms["role"] == role)
            ]
            .groupby("digital_number")["pixel_count"]
            .sum()
        )
        if not role_histogram.empty:
            axes[1].plot(
                role_histogram.index,
                role_histogram / role_histogram.sum(),
                label=role,
                color=color,
            )
    axes[1].set_title("Potsdam nDSM JPEG · encoded DN distribution")
    axes[1].set_xlabel("encoded digital number")
    axes[1].set_ylabel("pixel fraction")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    return fig


@plt.style.context(STYLE)
def augmentation_impact(augmentation_audit, datasets, optical_channels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    valid_data = [
        augmentation_audit.loc[
            augmentation_audit["dataset"] == name, "geometry_valid_fraction"
        ]
        for name in datasets
    ]
    datasets = tuple(datasets)
    axes[0].boxplot(valid_data, showfliers=False)
    axes[0].set_xticks(np.arange(1, len(datasets) + 1), datasets)
    axes[0].set_title("Valid pixels after geometry")
    axes[0].set_ylabel("fraction of crop")
    axes[0].grid(axis="y", alpha=0.2)

    for dataset_index, display_name in enumerate(datasets):
        subset = augmentation_audit[augmentation_audit["dataset"] == display_name]
        values = subset["class_share_abs_delta_mean"] * 100
        axes[1].scatter(
            np.full(len(values), dataset_index) + np.linspace(-0.12, 0.12, len(values)),
            values,
            s=12,
            alpha=0.55,
            label=display_name,
        )
    axes[1].set_xticks(np.arange(len(datasets)), datasets)
    axes[1].set_title("Mean absolute class-share shift")
    axes[1].set_ylabel("percentage points per class")
    axes[1].grid(axis="y", alpha=0.2)

    band_labels = []
    band_values = []
    for display_name in datasets:
        subset = augmentation_audit[augmentation_audit["dataset"] == display_name]
        for band_name in optical_channels:
            band_labels.append(f"{display_name}\n{band_name}")
            delta_column = f"band_{band_name}_mean_abs_radiometric_delta"
            changed = subset[delta_column] > 1e-8
            band_values.append(subset.loc[changed, delta_column].median())
    axes[2].bar(np.arange(len(band_values)), band_values, color="#70AD47")
    axes[2].set_xticks(np.arange(len(band_values)), band_labels, fontsize=8)
    axes[2].set_title("Optical change when active")
    axes[2].set_ylabel("median absolute DN change")
    axes[2].grid(axis="y", alpha=0.2)
    return fig
