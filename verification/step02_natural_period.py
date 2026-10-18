"""Step 02: natural-period verification of a one-story portal frame.

The two roof-node horizontal DOFs are tied to form one lateral floor DOF.  A
specified floor weight is converted to mass using M = W/g and divided equally
between the roof nodes.  The OpenSees eigenperiod is compared with

    T = 2 pi sqrt(M / k),

where k is obtained from the Step 01 static analysis of the same frame.

Units: kN, m, s.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import openseespy.opensees as ops

if __package__ in (None, ""):
    # VS Code's "Run Python File" adds this file's directory, not the project
    # root, to sys.path. Support both direct and module execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.step01_static_stiffness import (
    PROJECT_ROOT,
    PortalFrameParameters,
    build_portal_frame_model,
    run_static_case,
    theoretical_portal_stiffness,
    relative_beam_column_stiffness,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "verification" / "step02_natural_period"


@dataclass(frozen=True)
class NaturalPeriodParameters:
    floor_weight: float = 200.0
    gravity: float = 9.80665
    beam_inertia_factor: float = 1.0
    axial_area_factor: float = 1.0

    @property
    def floor_mass(self) -> float:
        return self.floor_weight / self.gravity


@dataclass(frozen=True)
class NaturalPeriodResult:
    floor_weight: float
    gravity: float
    floor_mass: float
    static_stiffness: float
    textbook_bending_stiffness: float
    sdof_angular_frequency: float
    sdof_period: float
    opensees_eigenvalue: float
    opensees_angular_frequency: float
    opensees_period: float
    period_error_percent: float
    textbook_period: float
    textbook_period_difference_percent: float
    total_free_mass_x: float
    mass_error_percent: float
    left_mode_component: float
    right_mode_component: float
    mode_shape_difference: float


def _build_dynamic_model(
    frame: PortalFrameParameters,
    dynamic: NaturalPeriodParameters,
) -> None:
    build_portal_frame_model(
        frame,
        beam_inertia_factor=dynamic.beam_inertia_factor,
        axial_area_factor=dynamic.axial_area_factor,
        lateral_load=None,
    )

    half_mass = dynamic.floor_mass / 2.0
    ops.mass(3, half_mass, 0.0, 0.0)
    ops.mass(4, half_mass, 0.0, 0.0)

    # One lateral floor DOF: the two roof nodes translate together in X.
    # Vertical translations and joint rotations remain unconstrained and are
    # statically condensed by the eigenvalue solution because they are massless.
    ops.equalDOF(3, 4, 1)


def run_natural_period_verification(
    frame: PortalFrameParameters | None = None,
    dynamic: NaturalPeriodParameters | None = None,
) -> NaturalPeriodResult:
    frame = frame or PortalFrameParameters()
    dynamic = dynamic or NaturalPeriodParameters()

    if dynamic.floor_weight <= 0.0:
        raise ValueError("floor_weight must be positive")
    if dynamic.gravity <= 0.0:
        raise ValueError("gravity must be positive")

    static_result = run_static_case(
        "step02_static_reference",
        frame,
        beam_inertia_factor=dynamic.beam_inertia_factor,
        axial_area_factor=dynamic.axial_area_factor,
    )
    static_stiffness = static_result.opensees_stiffness
    sdof_angular_frequency = math.sqrt(static_stiffness / dynamic.floor_mass)
    sdof_period = 2.0 * math.pi / sdof_angular_frequency

    beam_inertia = (
        frame.beam.second_moment_of_area * dynamic.beam_inertia_factor
    )
    rho = relative_beam_column_stiffness(
        beam_inertia,
        frame.column.second_moment_of_area,
        frame.bay_width,
        frame.story_height,
    )
    textbook_bending_stiffness = theoretical_portal_stiffness(
        frame.elastic_modulus,
        frame.column.second_moment_of_area,
        frame.story_height,
        rho,
    )
    textbook_period = 2.0 * math.pi * math.sqrt(
        dynamic.floor_mass / textbook_bending_stiffness
    )

    _build_dynamic_model(frame, dynamic)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")

    eigenvalues = ops.eigen("-fullGenLapack", 1)
    eigenvalue = float(eigenvalues[0])
    if eigenvalue <= 0.0:
        raise RuntimeError(f"Non-positive eigenvalue: {eigenvalue}")

    opensees_angular_frequency = math.sqrt(eigenvalue)
    opensees_period = 2.0 * math.pi / opensees_angular_frequency
    modal = ops.modalProperties("-return")
    total_free_mass_x = float(modal["totalFreeMass"][0])

    left_mode_component = float(ops.nodeEigenvector(3, 1, 1))
    right_mode_component = float(ops.nodeEigenvector(4, 1, 1))

    return NaturalPeriodResult(
        floor_weight=dynamic.floor_weight,
        gravity=dynamic.gravity,
        floor_mass=dynamic.floor_mass,
        static_stiffness=static_stiffness,
        textbook_bending_stiffness=textbook_bending_stiffness,
        sdof_angular_frequency=sdof_angular_frequency,
        sdof_period=sdof_period,
        opensees_eigenvalue=eigenvalue,
        opensees_angular_frequency=opensees_angular_frequency,
        opensees_period=opensees_period,
        period_error_percent=100.0 * (opensees_period - sdof_period) / sdof_period,
        textbook_period=textbook_period,
        textbook_period_difference_percent=(
            100.0 * (opensees_period - textbook_period) / textbook_period
        ),
        total_free_mass_x=total_free_mass_x,
        mass_error_percent=(
            100.0
            * (total_free_mass_x - dynamic.floor_mass)
            / dynamic.floor_mass
        ),
        left_mode_component=left_mode_component,
        right_mode_component=right_mode_component,
        mode_shape_difference=abs(left_mode_component - right_mode_component),
    )


def validate_result(result: NaturalPeriodResult) -> None:
    if abs(result.period_error_percent) > 1.0e-8:
        raise AssertionError(
            "OpenSees period does not match the SDOF period: "
            f"{result.period_error_percent:.6g}%"
        )
    if abs(result.mass_error_percent) > 1.0e-10:
        raise AssertionError(
            f"Total free X mass is inconsistent: {result.mass_error_percent:.6g}%"
        )
    if result.mode_shape_difference > 1.0e-12:
        raise AssertionError("The tied roof-node mode components are not equal")


def _write_csv(path: Path, result: NaturalPeriodResult) -> None:
    row = asdict(result)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_json(
    path: Path,
    frame: PortalFrameParameters,
    dynamic: NaturalPeriodParameters,
    result: NaturalPeriodResult,
) -> None:
    payload = {
        "units": {"force": "kN", "length": "m", "time": "s"},
        "frame": {
            "elastic_modulus": frame.elastic_modulus,
            "bay_width": frame.bay_width,
            "story_height": frame.story_height,
            "beam_area": frame.beam.area,
            "beam_second_moment_of_area": frame.beam.second_moment_of_area,
            "column_area": frame.column.area,
            "column_second_moment_of_area": frame.column.second_moment_of_area,
        },
        "dynamic_parameters": asdict(dynamic),
        "result": asdict(result),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_comparison_plot(path: Path, result: NaturalPeriodResult) -> None:
    labels = ["SDOF\n(static k)", "OpenSees\neigen", "Textbook\nbending k"]
    values = [result.sdof_period, result.opensees_period, result.textbook_period]
    colors = ["#333333", "#2673b8", "#b94a48"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylabel("Natural period (s)")
    ax.set_title("One-story one-bay natural-period verification")
    ax.grid(axis="y", linewidth=0.35, alpha=0.55)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0.0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_outputs(
    result: NaturalPeriodResult,
    frame: PortalFrameParameters,
    dynamic: NaturalPeriodParameters,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "natural_period_result.csv"
    json_path = output_dir / "natural_period_result.json"
    plot_path = output_dir / "natural_period_comparison.png"
    _write_csv(csv_path, result)
    _write_json(json_path, frame, dynamic, result)
    _write_comparison_plot(plot_path, result)
    return [csv_path, json_path, plot_path]


def _print_summary(result: NaturalPeriodResult) -> None:
    print("Step 02 - One-story one-bay natural-period verification")
    print("Units: kN, m, s")
    print(f"Floor weight:               {result.floor_weight:.6f} kN")
    print(f"Floor mass W/g:             {result.floor_mass:.9f} kN-s^2/m")
    print(f"Static OpenSees stiffness:  {result.static_stiffness:.6f} kN/m")
    print(f"Textbook bending stiffness: {result.textbook_bending_stiffness:.6f} kN/m")
    print(f"SDOF period from static k:  {result.sdof_period:.9f} s")
    print(f"OpenSees eigenperiod:       {result.opensees_period:.9f} s")
    print(f"Period error:               {result.period_error_percent:.9e} %")
    print(f"Textbook bending period:    {result.textbook_period:.9f} s")
    print(
        "Axial-deformation period difference: "
        f"{result.textbook_period_difference_percent:.6f} %"
    )
    print(f"Total free X mass:          {result.total_free_mass_x:.9f} kN-s^2/m")
    print(f"Mode-shape L/R difference:  {result.mode_shape_difference:.3e}")


def main() -> None:
    frame = PortalFrameParameters()
    dynamic = NaturalPeriodParameters()
    result = run_natural_period_verification(frame, dynamic)
    validate_result(result)
    paths = write_outputs(result, frame, dynamic)
    _print_summary(result)
    print()
    print("Verification status: PASS")
    for path in paths:
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
