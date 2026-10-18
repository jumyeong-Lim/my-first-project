"""Step 05: 2D floor-constraint and top-floor mass-distribution sensitivity.

The verified Step 04 three-story, three-bay elastic frame is used for four
cases: tied/untied floor horizontal DOFs and centered/eccentric top-floor mass.
The eccentric case redistributes the same top-floor mass among the four frame
nodes.  In a 2D frame this is a mass-distribution sensitivity, not the 3D
torsional mass eccentricity that will be studied in the main model.

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
import numpy as np
import openseespy.opensees as ops

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.step01_static_stiffness import PROJECT_ROOT, PortalFrameParameters
from verification.step04_three_story_multibay_frame import node_tag


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "step05_floor_constraint_mass_sensitivity"
)


@dataclass(frozen=True)
class Step05Parameters:
    stories: int = 3
    bays: int = 3
    floor_weights: tuple[float, float, float] = (200.0, 200.0, 200.0)
    gravity: float = 9.80665
    top_eccentricity_ratio: float = 0.20

    @property
    def floor_masses(self) -> np.ndarray:
        return np.asarray(self.floor_weights, dtype=float) / self.gravity


@dataclass(frozen=True)
class StaticCompatibilityResult:
    tied: bool
    floor_node_displacements: list[list[float]]
    floor_average_displacements: list[float]
    floor_spreads: list[float]
    roof_effective_stiffness: float


@dataclass(frozen=True)
class Step05CaseResult:
    case_id: str
    tied: bool
    eccentric_top_mass: bool
    mass_fractions_by_floor: list[list[float]]
    top_mass_centroid: float
    top_mass_eccentricity: float
    top_mass_eccentricity_ratio: float
    total_input_mass: float
    total_free_mass_x: float
    mass_error_percent: float
    eigenvalues: list[float]
    periods: list[float]
    floor_average_mode_shapes: list[list[float]]
    nodal_mode_shapes: list[list[list[float]]]
    maximum_floor_modal_spreads: list[float]
    static: StaticCompatibilityResult


def case_id(tied: bool, eccentric_top_mass: bool) -> str:
    return (
        ("tied" if tied else "untied")
        + "_"
        + ("top_eccentric" if eccentric_top_mass else "centered")
    )


def _centered_fractions(node_count: int) -> np.ndarray:
    return np.full(node_count, 1.0 / node_count, dtype=float)


def _top_mass_fractions(
    frame: PortalFrameParameters,
    parameters: Step05Parameters,
    eccentric: bool,
) -> np.ndarray:
    node_count = parameters.bays + 1
    if not eccentric:
        return _centered_fractions(node_count)

    x = np.arange(node_count, dtype=float) * frame.bay_width
    center = 0.5 * parameters.bays * frame.bay_width
    target_eccentricity = parameters.top_eccentricity_ratio * (2.0 * center)
    centered_x = x - center
    slope = target_eccentricity / float(np.mean(centered_x**2))
    fractions = (1.0 + slope * centered_x) / node_count
    if np.any(fractions <= 0.0):
        raise ValueError("Requested eccentricity produces a non-positive nodal mass")
    return fractions / np.sum(fractions)


def mass_fractions_by_floor(
    frame: PortalFrameParameters,
    parameters: Step05Parameters,
    eccentric_top_mass: bool,
) -> np.ndarray:
    fractions = np.tile(
        _centered_fractions(parameters.bays + 1),
        (parameters.stories, 1),
    )
    fractions[-1, :] = _top_mass_fractions(
        frame,
        parameters,
        eccentric_top_mass,
    )
    return fractions


def build_step05_frame(
    frame: PortalFrameParameters,
    parameters: Step05Parameters,
    tied: bool,
    eccentric_top_mass: bool,
    include_mass: bool,
) -> tuple[list[list[int]], list[int], np.ndarray]:
    if parameters.stories != 3 or parameters.bays != 3:
        raise ValueError("Step 05 requires the Step 04 three-story, three-bay frame")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    all_levels: list[list[int]] = []
    for level in range(parameters.stories + 1):
        level_nodes = []
        for column in range(parameters.bays + 1):
            tag = node_tag(level, column, parameters.bays)
            ops.node(
                tag,
                column * frame.bay_width,
                level * frame.story_height,
            )
            level_nodes.append(tag)
        all_levels.append(level_nodes)

    for tag in all_levels[0]:
        ops.fix(tag, 1, 1, 1)

    ops.geomTransf("Linear", 1)
    element_tag = 1
    for story in range(parameters.stories):
        for column in range(parameters.bays + 1):
            ops.element(
                "elasticBeamColumn",
                element_tag,
                all_levels[story][column],
                all_levels[story + 1][column],
                frame.column.area,
                frame.elastic_modulus,
                frame.column.second_moment_of_area,
                1,
            )
            element_tag += 1
        for bay in range(parameters.bays):
            ops.element(
                "elasticBeamColumn",
                element_tag,
                all_levels[story + 1][bay],
                all_levels[story + 1][bay + 1],
                frame.beam.area,
                frame.elastic_modulus,
                frame.beam.second_moment_of_area,
                1,
            )
            element_tag += 1

    floor_nodes = all_levels[1:]
    masters = [nodes[0] for nodes in floor_nodes]
    if tied:
        for nodes in floor_nodes:
            for constrained in nodes[1:]:
                ops.equalDOF(nodes[0], constrained, 1)

    fractions = mass_fractions_by_floor(frame, parameters, eccentric_top_mass)
    if include_mass:
        for floor_index, nodes in enumerate(floor_nodes):
            for column_index, tag in enumerate(nodes):
                nodal_mass = (
                    parameters.floor_masses[floor_index]
                    * fractions[floor_index, column_index]
                )
                ops.mass(tag, float(nodal_mass), 0.0, 0.0)

    return floor_nodes, masters, fractions


def run_static_compatibility_case(
    frame: PortalFrameParameters,
    parameters: Step05Parameters,
    tied: bool,
) -> StaticCompatibilityResult:
    floor_nodes, _, _ = build_step05_frame(
        frame,
        parameters,
        tied=tied,
        eccentric_top_mass=False,
        include_mass=False,
    )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    roof_load = 1.0 / len(floor_nodes[-1])
    for tag in floor_nodes[-1]:
        ops.load(tag, roof_load, 0.0, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 20)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("Step 05 static compatibility analysis failed")

    displacements = np.asarray(
        [
            [float(ops.nodeDisp(tag, 1)) for tag in nodes]
            for nodes in floor_nodes
        ],
        dtype=float,
    )
    averages = np.mean(displacements, axis=1)
    spreads = np.ptp(displacements, axis=1)
    return StaticCompatibilityResult(
        tied=tied,
        floor_node_displacements=displacements.tolist(),
        floor_average_displacements=averages.tolist(),
        floor_spreads=spreads.tolist(),
        roof_effective_stiffness=1.0 / float(averages[-1]),
    )


def _normalize_modes(
    nodal_modes: np.ndarray,
    fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = nodal_modes.copy()
    floor_averages = np.zeros((nodal_modes.shape[0], 3), dtype=float)
    for mode_index in range(nodal_modes.shape[0]):
        raw_average = np.sum(nodal_modes[mode_index] * fractions, axis=1)
        roof_value = raw_average[-1]
        if abs(roof_value) < 1.0e-14:
            raise RuntimeError("Cannot normalize a mode with zero roof average")
        normalized[mode_index] /= roof_value
        floor_averages[mode_index] = np.sum(
            normalized[mode_index] * fractions,
            axis=1,
        )
    return normalized, floor_averages


def run_step05_case(
    tied: bool,
    eccentric_top_mass: bool,
    frame: PortalFrameParameters | None = None,
    parameters: Step05Parameters | None = None,
    static_result: StaticCompatibilityResult | None = None,
) -> Step05CaseResult:
    frame = frame or PortalFrameParameters()
    parameters = parameters or Step05Parameters()
    if static_result is None:
        static_result = run_static_compatibility_case(frame, parameters, tied)

    floor_nodes, _, fractions = build_step05_frame(
        frame,
        parameters,
        tied=tied,
        eccentric_top_mass=eccentric_top_mass,
        include_mass=True,
    )
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    eigenvalues = np.asarray(ops.eigen("-fullGenLapack", 3), dtype=float)
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError("Step 05 produced a non-positive eigenvalue")

    nodal_modes = np.zeros((3, parameters.stories, parameters.bays + 1))
    for mode_index in range(3):
        for floor_index, nodes in enumerate(floor_nodes):
            nodal_modes[mode_index, floor_index, :] = [
                float(ops.nodeEigenvector(tag, mode_index + 1, 1)) for tag in nodes
            ]
    nodal_modes, floor_average_modes = _normalize_modes(nodal_modes, fractions)
    modal_spreads = np.max(np.ptp(nodal_modes, axis=2), axis=1)

    modal_properties = ops.modalProperties("-return")
    total_free_mass = float(modal_properties["totalFreeMass"][0])
    total_input_mass = float(np.sum(parameters.floor_masses))
    periods = 2.0 * math.pi / np.sqrt(eigenvalues)

    x = np.arange(parameters.bays + 1, dtype=float) * frame.bay_width
    top_centroid = float(np.dot(fractions[-1], x))
    geometric_center = 0.5 * parameters.bays * frame.bay_width
    eccentricity = top_centroid - geometric_center
    floor_width = parameters.bays * frame.bay_width

    return Step05CaseResult(
        case_id=case_id(tied, eccentric_top_mass),
        tied=tied,
        eccentric_top_mass=eccentric_top_mass,
        mass_fractions_by_floor=fractions.tolist(),
        top_mass_centroid=top_centroid,
        top_mass_eccentricity=eccentricity,
        top_mass_eccentricity_ratio=eccentricity / floor_width,
        total_input_mass=total_input_mass,
        total_free_mass_x=total_free_mass,
        mass_error_percent=100.0 * (total_free_mass - total_input_mass) / total_input_mass,
        eigenvalues=eigenvalues.tolist(),
        periods=periods.tolist(),
        floor_average_mode_shapes=floor_average_modes.tolist(),
        nodal_mode_shapes=nodal_modes.tolist(),
        maximum_floor_modal_spreads=modal_spreads.tolist(),
        static=static_result,
    )


def run_step05_verification(
    frame: PortalFrameParameters | None = None,
    parameters: Step05Parameters | None = None,
) -> dict[str, Step05CaseResult]:
    frame = frame or PortalFrameParameters()
    parameters = parameters or Step05Parameters()
    static_results = {
        tied: run_static_compatibility_case(frame, parameters, tied)
        for tied in (True, False)
    }
    results = {}
    for tied in (True, False):
        for eccentric in (False, True):
            result = run_step05_case(
                tied,
                eccentric,
                frame,
                parameters,
                static_results[tied],
            )
            results[result.case_id] = result
    return results


def _floor_mac(reference: np.ndarray, candidate: np.ndarray, masses: np.ndarray) -> float:
    mass = np.diag(masses)
    numerator = float(reference.T @ mass @ candidate) ** 2
    denominator = float(reference.T @ mass @ reference) * float(
        candidate.T @ mass @ candidate
    )
    return numerator / denominator


def validate_results(
    results: dict[str, Step05CaseResult],
    parameters: Step05Parameters,
) -> None:
    expected = {
        "tied_centered",
        "tied_top_eccentric",
        "untied_centered",
        "untied_top_eccentric",
    }
    if set(results) != expected:
        raise AssertionError("Step 05 did not produce all four sensitivity cases")

    for result in results.values():
        if abs(result.mass_error_percent) > 1.0e-10:
            raise AssertionError(f"Mass mismatch in {result.case_id}")
        if not all(value > 0.0 for value in result.eigenvalues):
            raise AssertionError(f"Non-positive eigenvalue in {result.case_id}")
        fractions = np.asarray(result.mass_fractions_by_floor)
        if not np.allclose(np.sum(fractions, axis=1), 1.0, atol=1.0e-13):
            raise AssertionError(f"Floor mass fractions do not sum to one in {result.case_id}")

    tied_centered = results["tied_centered"]
    tied_eccentric = results["tied_top_eccentric"]
    if not np.allclose(tied_centered.periods, tied_eccentric.periods, rtol=1.0e-11):
        raise AssertionError("Tied model changed when only nodal mass location changed")
    if max(tied_centered.maximum_floor_modal_spreads) > 1.0e-11:
        raise AssertionError("Tied floor modes do not satisfy horizontal compatibility")
    if max(tied_centered.static.floor_spreads) > 1.0e-13:
        raise AssertionError("Tied floor static displacements are not identical")

    eccentric = results["untied_top_eccentric"]
    if not math.isclose(
        eccentric.top_mass_eccentricity_ratio,
        parameters.top_eccentricity_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-13,
    ):
        raise AssertionError("Requested top-floor eccentricity was not achieved")
    if results["untied_centered"].static.floor_spreads[-1] <= 0.0:
        raise AssertionError("Untied model did not develop any roof displacement spread")


def _case_label(result: Step05CaseResult) -> str:
    constraint = "Tied" if result.tied else "Untied"
    mass = "eccentric top mass" if result.eccentric_top_mass else "centered mass"
    return f"{constraint}, {mass}"


def _write_csv(
    path: Path,
    results: dict[str, Step05CaseResult],
    parameters: Step05Parameters,
) -> None:
    reference = results["tied_centered"]
    rows = []
    masses = parameters.floor_masses
    for result in results.values():
        for mode_index in range(3):
            reference_shape = np.asarray(reference.floor_average_mode_shapes[mode_index])
            shape = np.asarray(result.floor_average_mode_shapes[mode_index])
            rows.append(
                {
                    "case_id": result.case_id,
                    "tied": result.tied,
                    "eccentric_top_mass": result.eccentric_top_mass,
                    "mode": mode_index + 1,
                    "period": result.periods[mode_index],
                    "period_change_vs_tied_centered_percent": 100.0
                    * (result.periods[mode_index] - reference.periods[mode_index])
                    / reference.periods[mode_index],
                    "floor_average_mac_vs_tied_centered": _floor_mac(
                        reference_shape,
                        shape,
                        masses,
                    ),
                    "maximum_floor_modal_spread": result.maximum_floor_modal_spreads[
                        mode_index
                    ],
                    "top_mass_eccentricity_ratio": result.top_mass_eccentricity_ratio,
                    "roof_effective_stiffness": result.static.roof_effective_stiffness,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: Path,
    frame: PortalFrameParameters,
    parameters: Step05Parameters,
    results: dict[str, Step05CaseResult],
) -> None:
    payload = {
        "units": {"force": "kN", "length": "m", "time": "s"},
        "assumed_inputs": {
            "frame": {
                "elastic_modulus": frame.elastic_modulus,
                "bay_width": frame.bay_width,
                "story_height": frame.story_height,
                "beam": asdict(frame.beam),
                "column": asdict(frame.column),
            },
            "step05_parameters": asdict(parameters),
            "interpretation_limit": (
                "Top-floor mass eccentricity is represented by asymmetric nodal "
                "mass distribution in the 2D frame. It cannot generate 3D floor "
                "torsion and is not the final diaphragm-eccentricity model."
            ),
        },
        "derived_values": {key: asdict(value) for key, value in results.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_model_figure(
    path: Path,
    frame: PortalFrameParameters,
    parameters: Step05Parameters,
    results: dict[str, Step05CaseResult],
) -> None:
    order = [
        "tied_centered",
        "untied_centered",
        "tied_top_eccentric",
        "untied_top_eccentric",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    width = parameters.bays * frame.bay_width
    height = parameters.stories * frame.story_height
    for ax, key in zip(axes.flat, order):
        result = results[key]
        for column in range(parameters.bays + 1):
            x = column * frame.bay_width
            ax.plot([x, x], [0.0, height], color="#2673b8", linewidth=2.0)
        for story in range(1, parameters.stories + 1):
            y = story * frame.story_height
            ax.plot([0.0, width], [y, y], color="#e07a22", linewidth=2.0)
            fractions = np.asarray(result.mass_fractions_by_floor[story - 1])
            for column, fraction in enumerate(fractions):
                x = column * frame.bay_width
                ax.scatter(
                    x,
                    y,
                    s=18.0 + 1200.0 * fraction**1.5,
                    color="#b94a48" if story == parameters.stories else "#8c8c8c",
                    edgecolor="#7d2222" if story == parameters.stories else "#555555",
                    linewidth=1.0,
                    zorder=4,
                )
                if story == parameters.stories:
                    ax.text(
                        x,
                        y + 0.82 + 0.75 * fraction,
                        f"{100.0 * fraction:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            if result.tied:
                ax.plot(
                    [0.0, width],
                    [y + 0.33, y + 0.33],
                    color="#4c8c4a",
                    linewidth=1.5,
                )
        ax.set_title(_case_label(result), fontsize=10)
        ax.set_xlim(-1.5, width + 1.5)
        ax.set_ylim(-0.6, height + 1.75)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.suptitle(
        "Step 05 cases: floor-X constraint and top-floor nodal mass distribution",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Labels give each roof node's share of the same total floor mass; marker area is enlarged for clarity. "
        "Green line indicates tied floor-X DOFs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_period_plot(path: Path, results: dict[str, Step05CaseResult]) -> None:
    order = [
        "tied_centered",
        "untied_centered",
        "tied_top_eccentric",
        "untied_top_eccentric",
    ]
    modes = np.arange(1, 4, dtype=float)
    width = 0.19
    colors = ["#2673b8", "#e07a22", "#4c8c4a", "#b94a48"]
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for index, (key, color) in enumerate(zip(order, colors)):
        result = results[key]
        offset = (index - 1.5) * width
        bars = ax.bar(
            modes + offset,
            result.periods,
            width,
            color=color,
            label=_case_label(result),
        )
        for bar, value in zip(bars, result.periods):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.5,
            )
    ax.set_xticks(modes, ["Mode 1", "Mode 2", "Mode 3"])
    ax.set_xlabel("Mode")
    ax.set_ylabel("Natural period (s)")
    ax.set_title("Natural-period sensitivity: floor-X constraint and top mass distribution")
    ax.grid(axis="y", linewidth=0.35, alpha=0.55)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_static_compatibility_plot(
    path: Path,
    frame: PortalFrameParameters,
    results: dict[str, Step05CaseResult],
) -> None:
    x = np.arange(4) * frame.bay_width
    tied = np.asarray(results["tied_centered"].static.floor_node_displacements)
    untied = np.asarray(results["untied_centered"].static.floor_node_displacements)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    for floor_index, ax in enumerate(axes):
        tied_relative = 1.0e6 * (
            tied[floor_index] - np.mean(tied[floor_index])
        )
        untied_relative = 1.0e6 * (
            untied[floor_index] - np.mean(untied[floor_index])
        )
        ax.plot(x, tied_relative, "o-", label="Tied")
        ax.plot(x, untied_relative, "s--", label="Untied")
        ax.set_title(f"Floor {floor_index + 1}")
        ax.set_xlabel("Node x-coordinate (m)")
        ax.grid(True, linewidth=0.35, alpha=0.55)
    axes[0].set_ylabel("Displacement relative to floor mean (micrometers)")
    axes[-1].legend()
    fig.suptitle(
        "Within-floor deformation under a distributed 1 kN roof load",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_mode_plot(path: Path, results: dict[str, Step05CaseResult]) -> None:
    order = [
        "tied_centered",
        "untied_centered",
        "tied_top_eccentric",
        "untied_top_eccentric",
    ]
    colors = ["#2673b8", "#e07a22", "#4c8c4a", "#b94a48"]
    markers = ["o", "s", "D", "^"]
    floors = np.arange(1, 4)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=True)
    for mode_index, ax in enumerate(axes):
        for key, color, marker in zip(order, colors, markers):
            result = results[key]
            shape = np.asarray(result.floor_average_mode_shapes[mode_index])
            ax.plot(
                np.r_[0.0, shape],
                np.r_[0, floors],
                marker=marker,
                color=color,
                linewidth=1.7,
                label=_case_label(result),
            )
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.set_title(f"Mode {mode_index + 1}")
        ax.set_xlabel("Roof-average-normalized displacement")
        ax.grid(True, linewidth=0.35, alpha=0.55)
    axes[0].set_ylabel("Floor")
    axes[-1].legend(fontsize=7.5)
    fig.suptitle("Floor-average mode shapes", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_outputs(
    frame: PortalFrameParameters,
    parameters: Step05Parameters,
    results: dict[str, Step05CaseResult],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "step05_mode_results.csv",
        output_dir / "step05_results.json",
        output_dir / "step05_model_cases.png",
        output_dir / "step05_period_comparison.png",
        output_dir / "static_floor_compatibility.png",
        output_dir / "step05_mode_shapes.png",
    ]
    _write_csv(paths[0], results, parameters)
    _write_json(paths[1], frame, parameters, results)
    _write_model_figure(paths[2], frame, parameters, results)
    _write_period_plot(paths[3], results)
    _write_static_compatibility_plot(paths[4], frame, results)
    _write_mode_plot(paths[5], results)
    return paths


def _print_summary(results: dict[str, Step05CaseResult]) -> None:
    reference = results["tied_centered"]
    print("Step 05 - 2D floor-X constraint and top mass-distribution sensitivity")
    print("The eccentric cases are 2D nodal mass redistribution, not 3D torsion.")
    print("Units: kN, m, s")
    for result in results.values():
        changes = [
            100.0 * (value - reference.periods[index]) / reference.periods[index]
            for index, value in enumerate(result.periods)
        ]
        print()
        print(_case_label(result))
        print(
            f"  top CM={result.top_mass_centroid:.3f} m, "
            f"e/B={result.top_mass_eccentricity_ratio:.3f}, "
            f"roof stiffness={result.static.roof_effective_stiffness:.3f} kN/m"
        )
        print("  mode     period(s)   change vs tied-centered(%)   max floor spread")
        for mode_index in range(3):
            print(
                f"  {mode_index + 1:4d}  {result.periods[mode_index]:12.7f}"
                f"  {changes[mode_index]:27.6f}"
                f"  {result.maximum_floor_modal_spreads[mode_index]:16.8e}"
            )


def main() -> None:
    frame = PortalFrameParameters()
    parameters = Step05Parameters()
    results = run_step05_verification(frame, parameters)
    validate_results(results, parameters)
    paths = write_outputs(frame, parameters, results)
    _print_summary(results)
    print()
    print("Verification status: PASS")
    for path in paths:
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
