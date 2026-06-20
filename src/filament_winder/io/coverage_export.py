"""Coverage map exporters."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from filament_winder.core.coverage import CoverageMap


def export_coverage_csv(coverage_map: CoverageMap, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["z_mm", "theta_rad", "theta_deg", "coverage_count", "covered"],
        )
        writer.writeheader()
        for z_index, z_mm in enumerate(coverage_map.z_mm):
            for theta_index, theta_rad in enumerate(coverage_map.theta_rad):
                count = int(coverage_map.coverage_count[z_index, theta_index])
                writer.writerow(
                    {
                        "z_mm": float(z_mm),
                        "theta_rad": float(theta_rad),
                        "theta_deg": float(np.rad2deg(theta_rad)),
                        "coverage_count": count,
                        "covered": count > 0,
                    }
                )
    return path


def export_coverage_summary_csv(coverage_map: CoverageMap, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = coverage_map.summary()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "covered_percent",
                "gap_percent",
                "overlap_percent",
                "max_coverage_count",
                "mean_coverage_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "covered_percent": summary.covered_percent,
                "gap_percent": summary.gap_percent,
                "overlap_percent": summary.overlap_percent,
                "max_coverage_count": summary.max_coverage_count,
                "mean_coverage_count": summary.mean_coverage_count,
            }
        )
    return path
