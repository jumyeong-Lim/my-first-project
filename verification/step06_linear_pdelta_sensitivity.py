"""Step 06: Linear/PDelta sensitivity of the three-story, three-bay frame.

The Step 04 reference frame is subjected to gravity loading before its tangent
eigenproperties and small-load roof stiffness are evaluated.  Column geometric
transformations are switched between Linear and PDelta; beams retain a Linear
transformation.  Gravity factors are included to demonstrate that PDelta
effects arise from compressive preload rather than from the command alone.

The OpenSees PDelta coordinate transformation captures P-large-Delta effects,
not P-small-delta effects.  Units: kN, m, s.
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
    PROJECT_ROOT / "results" / "verification" / "step06_linear_pdelta_sensitivity"
)


@dataclass(frozen=True)
class Step06Parameters:
    stories: int = 3
    bays: int = 3
    floor_weights: tuple[float, float, float] = (200.0, 200.0, 200.0)
    gravity: float = 9.80665
    gravity_factors: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    gravity_steps: int = 10
    roof_reference_load: float = 1.0

    @property
    def floor_masses(self) -> np.ndarray:
        return np.asarray(self.floor_weights, dtype=float) / self.gravity


@dataclass(frozen=True)
class Step06CaseResult:
    transformation: str
    gravity_factor: float
    applied_gravity_load: float
    base_vertical_reaction: float
    gravity_reaction_error_percent: float
    average_roof_vertical_displacement: float
    roof_lateral_displacement_increment: float
    roof_tangent_stiffness: float
    total_input_mass: float
    total_free_mass_x: float
    mass_error_percent: float
    eigenvalues: list[float]
    periods: list[float]
    mode_shapes: list[list[float]]


def build_step06_frame(
    frame: PortalFrameParameters,
    parameters: Step06Parameters,
    transformation: str,
) -> tuple[list[list[int]], list[int], list[int]]:
    if parameters.stories != 3 or parameters.bays != 3:
        raise ValueError("Step 06 requires a three-story, three-bay frame")
    if transformation not in {"Linear", "PDelta"}:
        raise ValueError("transformation must be 'Linear' or 'PDelta'")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    all_levels: list[list[int]] = []
    for level in range(parameters.stories + 1):
        nodes = []
        for column in range(parameters.bays + 1):
            tag = node_tag(level, column, parameters.bays)
            ops.node(
                tag,
                column * frame.bay_width,
                level * frame.story_height,
            )
            nodes.append(tag)
        all_levels.append(nodes)

    base_nodes = all_levels[0]
    for tag in base_nodes:
        ops.fix(tag, 1, 1, 1)

    ops.geomTransf(transformation, 1)
    ops.geomTransf("Linear", 2)
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
                2,
            )
            element_tag += 1

    floor_nodes = all_levels[1:]
    masters = [nodes[0] for nodes in floor_nodes]
    for floor_index, nodes in enumerate(floor_nodes):
        for constrained in nodes[1:]:
            ops.equalDOF(nodes[0], constrained, 1)
        nodal_mass = float(parameters.floor_masses[floor_index]) / len(nodes)
        for tag in nodes:
            ops.mass(tag, nodal_mass, 0.0, 0.0)

    return floor_nodes, masters, base_nodes


def _run_gravity_analysis(
    floor_nodes: list[list[int]],
    base_nodes: list[int],
    parameters: Step06Parameters,
    gravity_factor: float,
) -> tuple[float, float, float]:
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for floor_index, nodes in enumerate(floor_nodes):
        nodal_weight = (
            gravity_factor * parameters.floor_weights[floor_index] / len(nodes)
        )
        for tag in nodes:
            ops.load(tag, 0.0, -nodal_weight, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 30)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / parameters.gravity_steps)
    ops.analysis("Static")
    if ops.analyze(parameters.gravity_steps) != 0:
        raise RuntimeError(
            f"Gravity analysis failed for factor {gravity_factor:.3f}"
        )

    ops.reactions()
    base_reaction = float(sum(ops.nodeReaction(tag, 2) for tag in base_nodes))
    applied_load = gravity_factor * float(sum(parameters.floor_weights))
    if applied_load == 0.0:
        reaction_error = abs(base_reaction)
    else:
        reaction_error = 100.0 * (base_reaction - applied_load) / applied_load
    roof_vertical = float(
        np.mean([ops.nodeDisp(tag, 2) for tag in floor_nodes[-1]])
    )
    ops.loadConst("-time", 0.0)
    ops.wipeAnalysis()
    return applied_load, base_reaction, reaction_error, roof_vertical


def _run_eigenanalysis(
    floor_nodes: list[list[int]],
    masters: list[int],
    parameters: Step06Parameters,
) -> tuple[np.ndarray, np.ndarray, float]:
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    eigenvalues = np.asarray(ops.eigen("-fullGenLapack", 3), dtype=float)
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError("Step 06 produced a non-positive eigenvalue")

    modes = np.empty((3, 3), dtype=float)
    for mode_index in range(3):
        for floor_index, master in enumerate(masters):
            modes[mode_index, floor_index] = float(
                ops.nodeEigenvector(master, mode_index + 1, 1)
            )
        roof_value = modes[mode_index, -1]
        if abs(roof_value) < 1.0e-14:
            raise RuntimeError("Cannot normalize a mode with zero roof component")
        modes[mode_index] /= roof_value

    modal = ops.modalProperties("-return")
    total_free_mass_x = float(modal["totalFreeMass"][0])
    ops.wipeAnalysis()
    return eigenvalues, modes, total_free_mass_x


def _run_roof_stiffness_analysis(
    floor_nodes: list[list[int]],
    parameters: Step06Parameters,
) -> tuple[float, float]:
    initial_roof_displacement = float(
        np.mean([ops.nodeDisp(tag, 1) for tag in floor_nodes[-1]])
    )
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    nodal_load = parameters.roof_reference_load / len(floor_nodes[-1])
    for tag in floor_nodes[-1]:
        ops.load(tag, nodal_load, 0.0, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 30)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("Step 06 lateral stiffness analysis failed")

    final_roof_displacement = float(
        np.mean([ops.nodeDisp(tag, 1) for tag in floor_nodes[-1]])
    )
    increment = final_roof_displacement - initial_roof_displacement
    return increment, parameters.roof_reference_load / increment


def run_step06_case(
    transformation: str,
    gravity_factor: float,
    frame: PortalFrameParameters | None = None,
    parameters: Step06Parameters | None = None,
) -> Step06CaseResult:
    frame = frame or PortalFrameParameters()
    parameters = parameters or Step06Parameters()
    floor_nodes, masters, base_nodes = build_step06_frame(
        frame,
        parameters,
        transformation,
    )
    (
        applied_load,
        base_reaction,
        reaction_error,
        roof_vertical,
    ) = _run_gravity_analysis(
        floor_nodes,
        base_nodes,
        parameters,
        gravity_factor,
    )
    eigenvalues, modes, total_free_mass_x = _run_eigenanalysis(
        floor_nodes,
        masters,
        parameters,
    )
    roof_increment, roof_stiffness = _run_roof_stiffness_analysis(
        floor_nodes,
        parameters,
    )

    total_input_mass = float(np.sum(parameters.floor_masses))
    periods = 2.0 * math.pi / np.sqrt(eigenvalues)
    return Step06CaseResult(
        transformation=transformation,
        gravity_factor=gravity_factor,
        applied_gravity_load=applied_load,
        base_vertical_reaction=base_reaction,
        gravity_reaction_error_percent=reaction_error,
        average_roof_vertical_displacement=roof_vertical,
        roof_lateral_displacement_increment=roof_increment,
        roof_tangent_stiffness=roof_stiffness,
        total_input_mass=total_input_mass,
        total_free_mass_x=total_free_mass_x,
        mass_error_percent=100.0
        * (total_free_mass_x - total_input_mass)
        / total_input_mass,
        eigenvalues=eigenvalues.tolist(),
        periods=periods.tolist(),
        mode_shapes=modes.tolist(),
    )


def run_step06_verification(
    frame: PortalFrameParameters | None = None,
    parameters: Step06Parameters | None = None,
) -> dict[str, Step06CaseResult]:
    frame = frame or PortalFrameParameters()
    parameters = parameters or Step06Parameters()
    results = {}
    for transformation in ("Linear", "PDelta"):
        for gravity_factor in parameters.gravity_factors:
            result = run_step06_case(
                transformation,
                gravity_factor,
                frame,
                parameters,
            )
            key = f"{transformation.lower()}_g{gravity_factor:.2f}"
            results[key] = result
    return results


def _mass_weighted_mac(
    first: np.ndarray,
    second: np.ndarray,
    floor_masses: np.ndarray,
) -> float:
    mass = np.diag(floor_masses)
    numerator = float(first.T @ mass @ second) ** 2
    denominator = float(first.T @ mass @ first) * float(
        second.T @ mass @ second
    )
    return numerator / denominator


def validate_results(
    results: dict[str, Step06CaseResult],
    parameters: Step06Parameters,
) -> None:
    if len(results) != 2 * len(parameters.gravity_factors):
        raise AssertionError("Step 06 did not produce every sensitivity case")
    for result in results.values():
        if abs(result.gravity_reaction_error_percent) > 1.0e-8:
            raise AssertionError(
                f"Gravity reaction mismatch in {result.transformation}, "
                f"factor {result.gravity_factor}"
            )
        if abs(result.mass_error_percent) > 1.0e-10:
            raise AssertionError("Step 06 mass mismatch")
        if not all(value > 0.0 for value in result.eigenvalues):
            raise AssertionError("Step 06 non-positive eigenvalue")

    linear_zero = results["linear_g0.00"]
    pdelta_zero = results["pdelta_g0.00"]
    if not np.allclose(linear_zero.periods, pdelta_zero.periods, rtol=1.0e-11):
        raise AssertionError("Linear/PDelta differ without compressive preload")
    if not math.isclose(
        linear_zero.roof_tangent_stiffness,
        pdelta_zero.roof_tangent_stiffness,
        rel_tol=1.0e-10,
    ):
        raise AssertionError("Linear/PDelta stiffness differs without gravity")

    linear_periods = np.asarray(
        [results[f"linear_g{factor:.2f}"].periods for factor in parameters.gravity_factors]
    )
    if not np.allclose(linear_periods, linear_periods[0], rtol=1.0e-11):
        raise AssertionError("Linear transformation changed with gravity factor")

    pdelta_first_periods = np.asarray(
        [
            results[f"pdelta_g{factor:.2f}"].periods[0]
            for factor in parameters.gravity_factors
        ]
    )
    pdelta_stiffnesses = np.asarray(
        [
            results[f"pdelta_g{factor:.2f}"].roof_tangent_stiffness
            for factor in parameters.gravity_factors
        ]
    )
    if not np.all(np.diff(pdelta_first_periods) > 0.0):
        raise AssertionError("PDelta first period does not increase with gravity")
    if not np.all(np.diff(pdelta_stiffnesses) < 0.0):
        raise AssertionError("PDelta roof stiffness does not decrease with gravity")


def _write_csv(
    path: Path,
    results: dict[str, Step06CaseResult],
    parameters: Step06Parameters,
) -> None:
    rows = []
    for factor in parameters.gravity_factors:
        linear = results[f"linear_g{factor:.2f}"]
        pdelta = results[f"pdelta_g{factor:.2f}"]
        for transformation, result in (("Linear", linear), ("PDelta", pdelta)):
            for mode_index in range(3):
                rows.append(
                    {
                        "transformation": transformation,
                        "gravity_factor": factor,
                        "mode": mode_index + 1,
                        "period": result.periods[mode_index],
                        "period_change_vs_linear_percent": 100.0
                        * (result.periods[mode_index] - linear.periods[mode_index])
                        / linear.periods[mode_index],
                        "mac_vs_linear": _mass_weighted_mac(
                            np.asarray(linear.mode_shapes[mode_index]),
                            np.asarray(result.mode_shapes[mode_index]),
                            parameters.floor_masses,
                        ),
                        "roof_tangent_stiffness": result.roof_tangent_stiffness,
                        "stiffness_change_vs_linear_percent": 100.0
                        * (
                            result.roof_tangent_stiffness
                            - linear.roof_tangent_stiffness
                        )
                        / linear.roof_tangent_stiffness,
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: Path,
    frame: PortalFrameParameters,
    parameters: Step06Parameters,
    results: dict[str, Step06CaseResult],
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
            "step06_parameters": asdict(parameters),
            "gravity_distribution": (
                "Each floor weight is divided equally among its four frame nodes."
            ),
            "transformation_scope": (
                "Linear or PDelta is assigned to columns; beams remain Linear."
            ),
            "pdelta_scope": (
                "OpenSees PDelta transformation includes P-large-Delta but not "
                "P-small-delta effects."
            ),
        },
        "derived_values": {key: asdict(value) for key, value in results.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_model_figure(
    path: Path,
    frame: PortalFrameParameters,
    parameters: Step06Parameters,
) -> None:
    width = parameters.bays * frame.bay_width
    height = parameters.stories * frame.story_height
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for column in range(parameters.bays + 1):
        x = column * frame.bay_width
        ax.plot([x, x], [0.0, height], color="#2673b8", linewidth=2.4)
    for story in range(1, parameters.stories + 1):
        y = story * frame.story_height
        ax.plot([0.0, width], [y, y], color="#e07a22", linewidth=2.4)
        for column in range(parameters.bays + 1):
            x = column * frame.bay_width
            ax.annotate(
                "",
                xy=(x, y - 0.9),
                xytext=(x, y - 0.1),
                arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 1.2},
            )
    ax.annotate(
        "1 kN roof reference load",
        xy=(width, height),
        xytext=(width - 5.5, height + 1.2),
        arrowprops={"arrowstyle": "-|>", "color": "#b94a48", "lw": 1.8},
        color="#b94a48",
        ha="right",
    )
    ax.text(
        width + 0.7,
        0.5 * height,
        "Column transformation:\nLinear or PDelta",
        va="center",
        fontsize=10,
    )
    ax.set_xlim(-1.0, width + 5.8)
    ax.set_ylim(-0.6, height + 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(
        "Step 06 model: gravity preload followed by eigenvalue and roof-stiffness analyses",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_reference_period_plot(
    path: Path,
    results: dict[str, Step06CaseResult],
) -> None:
    linear = results["linear_g1.00"]
    pdelta = results["pdelta_g1.00"]
    modes = np.arange(1, 4, dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    bars_linear = ax.bar(modes - width / 2, linear.periods, width, label="Linear")
    bars_pdelta = ax.bar(modes + width / 2, pdelta.periods, width, label="PDelta")
    for bars, values in ((bars_linear, linear.periods), (bars_pdelta, pdelta.periods)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.6f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(0.0, max(max(linear.periods), max(pdelta.periods)) * 1.10)
    ax.set_xticks(modes, ["Mode 1", "Mode 2", "Mode 3"])
    ax.set_xlabel("Mode")
    ax.set_ylabel("Natural period (s)")
    ax.set_title("Natural periods after the reference gravity load (factor = 1.0)")
    ax.grid(axis="y", linewidth=0.35, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_gravity_sensitivity_plot(
    path: Path,
    results: dict[str, Step06CaseResult],
    parameters: Step06Parameters,
) -> None:
    factors = np.asarray(parameters.gravity_factors)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for transformation, marker in (("Linear", "o"), ("PDelta", "s")):
        first_periods = [
            results[f"{transformation.lower()}_g{factor:.2f}"].periods[0]
            for factor in factors
        ]
        stiffnesses = [
            results[
                f"{transformation.lower()}_g{factor:.2f}"
            ].roof_tangent_stiffness
            for factor in factors
        ]
        axes[0].plot(
            factors,
            first_periods,
            marker=marker,
            linewidth=1.8,
            label=transformation,
        )
        axes[1].plot(
            factors,
            stiffnesses,
            marker=marker,
            linewidth=1.8,
            label=transformation,
        )
    axes[0].set_title("First-mode period")
    axes[0].set_ylabel("Natural period (s)")
    axes[1].set_title("Roof tangent stiffness")
    axes[1].set_ylabel("Stiffness (kN/m)")
    for ax in axes:
        ax.set_xlabel("Gravity-load factor")
        ax.set_xticks(factors)
        ax.grid(True, linewidth=0.35, alpha=0.55)
        ax.legend()
    fig.suptitle("Gravity-preload sensitivity of Linear and PDelta models", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_mode_plot(
    path: Path,
    results: dict[str, Step06CaseResult],
) -> None:
    floors = np.arange(1, 4)
    linear = results["linear_g1.00"]
    pdelta = results["pdelta_g1.00"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), sharey=True)
    for mode_index, ax in enumerate(axes):
        for label, result, marker in (
            ("Linear", linear, "o"),
            ("PDelta", pdelta, "s"),
        ):
            shape = np.asarray(result.mode_shapes[mode_index])
            ax.plot(
                np.r_[0.0, shape],
                np.r_[0, floors],
                marker=marker,
                linewidth=1.8,
                label=label,
            )
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.set_title(f"Mode {mode_index + 1}")
        ax.set_xlabel("Roof-normalized displacement")
        ax.grid(True, linewidth=0.35, alpha=0.55)
    axes[0].set_ylabel("Floor")
    axes[-1].legend()
    fig.suptitle("Mode shapes after the reference gravity load", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_outputs(
    frame: PortalFrameParameters,
    parameters: Step06Parameters,
    results: dict[str, Step06CaseResult],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "step06_mode_results.csv",
        output_dir / "step06_results.json",
        output_dir / "step06_model_and_process.png",
        output_dir / "step06_period_comparison.png",
        output_dir / "step06_gravity_factor_sensitivity.png",
        output_dir / "step06_mode_shapes.png",
    ]
    _write_csv(paths[0], results, parameters)
    _write_json(paths[1], frame, parameters, results)
    _write_model_figure(paths[2], frame, parameters)
    _write_reference_period_plot(paths[3], results)
    _write_gravity_sensitivity_plot(paths[4], results, parameters)
    _write_mode_plot(paths[5], results)
    return paths


def _print_summary(
    results: dict[str, Step06CaseResult],
    parameters: Step06Parameters,
) -> None:
    print("Step 06 - Linear/PDelta gravity-preload sensitivity")
    print("Column transformation varies; beams remain Linear.")
    print("OpenSees PDelta includes P-large-Delta, not P-small-delta effects.")
    print("Units: kN, m, s")
    print()
    print(
        f"{'g-factor':>8s} {'transform':>10s} {'roof k':>13s} "
        f"{'T1':>11s} {'T2':>11s} {'T3':>11s}"
    )
    for factor in parameters.gravity_factors:
        linear = results[f"linear_g{factor:.2f}"]
        for transformation in ("Linear", "PDelta"):
            result = results[f"{transformation.lower()}_g{factor:.2f}"]
            print(
                f"{factor:8.2f} {transformation:>10s} "
                f"{result.roof_tangent_stiffness:13.3f} "
                f"{result.periods[0]:11.7f} "
                f"{result.periods[1]:11.7f} "
                f"{result.periods[2]:11.7f}"
            )
        pdelta = results[f"pdelta_g{factor:.2f}"]
        print(
            " " * 20
            + f"PDelta changes: k={100.0 * (pdelta.roof_tangent_stiffness - linear.roof_tangent_stiffness) / linear.roof_tangent_stiffness:+.4f}%, "
            + f"T1={100.0 * (pdelta.periods[0] - linear.periods[0]) / linear.periods[0]:+.4f}%"
        )


def main() -> None:
    frame = PortalFrameParameters()
    parameters = Step06Parameters()
    results = run_step06_verification(frame, parameters)
    validate_results(results, parameters)
    paths = write_outputs(frame, parameters, results)
    _print_summary(results, parameters)
    print()
    print("Verification status: PASS")
    for path in paths:
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
