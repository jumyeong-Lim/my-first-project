"""Step 03: compare a three-story portal frame with a shear-building model.

Each floor has one horizontal inertial degree of freedom.  The shear-building
story stiffness is calibrated from the Step 01 one-story portal-frame static
stiffness using the same member properties.  The full OpenSees model retains
member axial deformation and all joint rotations, so the comparison measures
the error introduced by the shear-building idealization.

Units: kN, m, s.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import numpy as np
# pyrefly: ignore [missing-import]
import openseespy.opensees as ops

from verification.step01_static_stiffness import (
    PROJECT_ROOT,
    PortalFrameParameters,
    run_static_case,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "verification" / "step03_three_story_shear_comparison"
)


@dataclass(frozen=True)
class ThreeStoryParameters:
    floor_weights: tuple[float, float, float] = (200.0, 200.0, 200.0)
    gravity: float = 9.80665
    beam_inertia_factor: float = 1.0
    axial_area_factor: float = 1.0

    @property
    def floor_masses(self) -> np.ndarray:
        return np.asarray(self.floor_weights, dtype=float) / self.gravity


@dataclass(frozen=True)
class ModeComparison:
    mode: int
    shear_eigenvalue: float
    frame_eigenvalue: float
    shear_period: float
    frame_period: float
    period_difference_percent: float
    mass_weighted_mac: float


@dataclass(frozen=True)
class ThreeStoryResult:
    story_stiffness: float
    floor_masses: tuple[float, float, float]
    shear_stiffness_matrix: list[list[float]]
    mass_matrix: list[list[float]]
    shear_mode_shapes: list[list[float]]
    frame_mode_shapes: list[list[float]]
    modes: list[ModeComparison]
    total_input_mass: float
    total_free_mass_x: float
    mass_error_percent: float
    maximum_tied_dof_difference: float
    maximum_shear_eigen_residual: float


def assemble_shear_matrices(
    story_stiffnesses: np.ndarray,
    floor_masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble K and M for floor DOFs ordered from first floor to roof."""

    story_stiffnesses = np.asarray(story_stiffnesses, dtype=float)
    floor_masses = np.asarray(floor_masses, dtype=float)
    if story_stiffnesses.shape != (3,) or floor_masses.shape != (3,):
        raise ValueError("Exactly three story stiffnesses and masses are required")
    if np.any(story_stiffnesses <= 0.0) or np.any(floor_masses <= 0.0):
        raise ValueError("Story stiffnesses and floor masses must be positive")

    k1, k2, k3 = story_stiffnesses
    stiffness = np.array(
        [
            [k1 + k2, -k2, 0.0],
            [-k2, k2 + k3, -k3],
            [0.0, -k3, k3],
        ],
        dtype=float,
    )
    mass = np.diag(floor_masses)
    return stiffness, mass


def _normalize_modes_at_roof(modes: np.ndarray) -> np.ndarray:
    normalized = np.asarray(modes, dtype=float).copy()
    for index in range(normalized.shape[1]):
        roof_value = normalized[-1, index]
        if abs(roof_value) < 1.0e-14:
            raise RuntimeError("A mode cannot be normalized by its roof component")
        normalized[:, index] /= roof_value
    return normalized


def solve_shear_building(
    stiffness: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the symmetric generalized eigenproblem K phi = lambda M phi."""

    inverse_sqrt_mass = np.diag(1.0 / np.sqrt(np.diag(mass)))
    transformed = inverse_sqrt_mass @ stiffness @ inverse_sqrt_mass
    eigenvalues, transformed_modes = np.linalg.eigh(transformed)
    modes = inverse_sqrt_mass @ transformed_modes
    modes = _normalize_modes_at_roof(modes)

    residuals = []
    for index, eigenvalue in enumerate(eigenvalues):
        vector = modes[:, index]
        residual = stiffness @ vector - eigenvalue * mass @ vector
        scale = np.linalg.norm(stiffness @ vector)
        residuals.append(np.linalg.norm(residual) / scale)
    return eigenvalues, modes, float(max(residuals))


def build_three_story_frame(
    frame: PortalFrameParameters,
    parameters: ThreeStoryParameters,
) -> tuple[list[int], list[int]]:
    """Build a three-story, one-bay elastic moment frame in OpenSees."""

    if parameters.beam_inertia_factor <= 0.0:
        raise ValueError("beam_inertia_factor must be positive")
    if parameters.axial_area_factor <= 0.0:
        raise ValueError("axial_area_factor must be positive")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    left_nodes = [1, 3, 5, 7]
    right_nodes = [2, 4, 6, 8]
    for level, (left, right) in enumerate(zip(left_nodes, right_nodes)):
        elevation = level * frame.story_height
        ops.node(left, 0.0, elevation)
        ops.node(right, frame.bay_width, elevation)

    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)
    ops.geomTransf("Linear", 1)

    column_area = frame.column.area * parameters.axial_area_factor
    beam_area = frame.beam.area * parameters.axial_area_factor
    column_inertia = frame.column.second_moment_of_area
    beam_inertia = (
        frame.beam.second_moment_of_area * parameters.beam_inertia_factor
    )

    element_tag = 1
    for story in range(3):
        ops.element(
            "elasticBeamColumn",
            element_tag,
            left_nodes[story],
            left_nodes[story + 1],
            column_area,
            frame.elastic_modulus,
            column_inertia,
            1,
        )
        element_tag += 1
        ops.element(
            "elasticBeamColumn",
            element_tag,
            right_nodes[story],
            right_nodes[story + 1],
            column_area,
            frame.elastic_modulus,
            column_inertia,
            1,
        )
        element_tag += 1
        ops.element(
            "elasticBeamColumn",
            element_tag,
            left_nodes[story + 1],
            right_nodes[story + 1],
            beam_area,
            frame.elastic_modulus,
            beam_inertia,
            1,
        )
        element_tag += 1

    floor_left = left_nodes[1:]
    floor_right = right_nodes[1:]
    for left, right, floor_mass in zip(
        floor_left, floor_right, parameters.floor_masses
    ):
        half_mass = float(floor_mass) / 2.0
        ops.mass(left, half_mass, 0.0, 0.0)
        ops.mass(right, half_mass, 0.0, 0.0)
        ops.equalDOF(left, right, 1)

    return floor_left, floor_right


def _mass_weighted_mac(
    first: np.ndarray,
    second: np.ndarray,
    mass: np.ndarray,
) -> float:
    numerator = float(first.T @ mass @ second) ** 2
    denominator = float(first.T @ mass @ first) * float(second.T @ mass @ second)
    return numerator / denominator


def run_three_story_comparison(
    frame: PortalFrameParameters | None = None,
    parameters: ThreeStoryParameters | None = None,
) -> ThreeStoryResult:
    frame = frame or PortalFrameParameters()
    parameters = parameters or ThreeStoryParameters()
    if parameters.gravity <= 0.0 or any(w <= 0.0 for w in parameters.floor_weights):
        raise ValueError("Gravity and floor weights must be positive")

    one_story = run_static_case(
        "step03_story_stiffness",
        frame,
        parameters.beam_inertia_factor,
        parameters.axial_area_factor,
    )
    story_stiffness = one_story.opensees_stiffness
    floor_masses = parameters.floor_masses
    shear_stiffness, mass = assemble_shear_matrices(
        np.full(3, story_stiffness), floor_masses
    )
    shear_eigenvalues, shear_modes, maximum_residual = solve_shear_building(
        shear_stiffness, mass
    )

    left_nodes, right_nodes = build_three_story_frame(frame, parameters)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    frame_eigenvalues = np.asarray(
        ops.eigen("-fullGenLapack", 3), dtype=float
    )
    if np.any(frame_eigenvalues <= 0.0):
        raise RuntimeError("The frame returned a non-positive eigenvalue")

    frame_modes = np.empty((3, 3), dtype=float)
    maximum_tied_difference = 0.0
    for mode_index in range(3):
        for floor_index, (left, right) in enumerate(zip(left_nodes, right_nodes)):
            left_value = float(ops.nodeEigenvector(left, mode_index + 1, 1))
            right_value = float(ops.nodeEigenvector(right, mode_index + 1, 1))
            frame_modes[floor_index, mode_index] = left_value
            maximum_tied_difference = max(
                maximum_tied_difference, abs(left_value - right_value)
            )
    frame_modes = _normalize_modes_at_roof(frame_modes)

    modal = ops.modalProperties("-return")
    total_free_mass_x = float(modal["totalFreeMass"][0])
    total_input_mass = float(np.sum(floor_masses))

    comparisons = []
    for index in range(3):
        shear_period = 2.0 * math.pi / math.sqrt(float(shear_eigenvalues[index]))
        frame_period = 2.0 * math.pi / math.sqrt(float(frame_eigenvalues[index]))
        comparisons.append(
            ModeComparison(
                mode=index + 1,
                shear_eigenvalue=float(shear_eigenvalues[index]),
                frame_eigenvalue=float(frame_eigenvalues[index]),
                shear_period=shear_period,
                frame_period=frame_period,
                period_difference_percent=(
                    100.0 * (frame_period - shear_period) / shear_period
                ),
                mass_weighted_mac=_mass_weighted_mac(
                    shear_modes[:, index], frame_modes[:, index], mass
                ),
            )
        )

    return ThreeStoryResult(
        story_stiffness=story_stiffness,
        floor_masses=tuple(float(value) for value in floor_masses),
        shear_stiffness_matrix=shear_stiffness.tolist(),
        mass_matrix=mass.tolist(),
        shear_mode_shapes=shear_modes.T.tolist(),
        frame_mode_shapes=frame_modes.T.tolist(),
        modes=comparisons,
        total_input_mass=total_input_mass,
        total_free_mass_x=total_free_mass_x,
        mass_error_percent=(
            100.0 * (total_free_mass_x - total_input_mass) / total_input_mass
        ),
        maximum_tied_dof_difference=maximum_tied_difference,
        maximum_shear_eigen_residual=maximum_residual,
    )


def validate_result(result: ThreeStoryResult) -> None:
    if len(result.modes) != 3:
        raise AssertionError("Three modes were not returned")
    if abs(result.mass_error_percent) > 1.0e-10:
        raise AssertionError("OpenSees free mass differs from the input mass")
    if result.maximum_tied_dof_difference > 1.0e-12:
        raise AssertionError("The two horizontal DOFs on a floor are not equal")
    if result.maximum_shear_eigen_residual > 1.0e-12:
        raise AssertionError("The analytical shear-building eigen residual is too large")
    if any(mode.mass_weighted_mac < 0.90 for mode in result.modes):
        raise AssertionError("A corresponding frame/shear mode has MAC below 0.90")


def run_sensitivity_cases(
    frame: PortalFrameParameters,
) -> dict[str, ThreeStoryResult]:
    cases = {
        "flexible_beam": ThreeStoryParameters(beam_inertia_factor=0.25),
        "reference": ThreeStoryParameters(),
        "stiff_beam": ThreeStoryParameters(beam_inertia_factor=4.0),
        "axially_rigid_reference": ThreeStoryParameters(axial_area_factor=1.0e6),
    }
    return {
        name: run_three_story_comparison(frame, parameters)
        for name, parameters in cases.items()
    }


def _write_csv(path: Path, results: dict[str, ThreeStoryResult]) -> None:
    rows = []
    for case_name, result in results.items():
        for mode in result.modes:
            rows.append(
                {
                    "case": case_name,
                    "story_stiffness": result.story_stiffness,
                    **asdict(mode),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: Path,
    frame: PortalFrameParameters,
    results: dict[str, ThreeStoryResult],
) -> None:
    payload = {
        "units": {"force": "kN", "length": "m", "time": "s"},
        "assumed_frame": {
            "elastic_modulus": frame.elastic_modulus,
            "bay_width": frame.bay_width,
            "story_height": frame.story_height,
            "beam": asdict(frame.beam),
            "column": asdict(frame.column),
        },
        "cases": {
            name: asdict(result) for name, result in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_period_plot(path: Path, result: ThreeStoryResult) -> None:
    modes = np.arange(1, 4)
    shear_periods = [mode.shear_period for mode in result.modes]
    frame_periods = [mode.frame_period for mode in result.modes]
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.bar(modes - width / 2.0, shear_periods, width, label="Shear building")
    ax.bar(modes + width / 2.0, frame_periods, width, label="OpenSees frame")
    ax.set_xticks(modes)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Natural period (s)")
    ax.set_title("Three-story one-bay period comparison")
    ax.grid(axis="y", linewidth=0.35, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_mode_plot(path: Path, result: ThreeStoryResult) -> None:
    floors = np.arange(1, 4)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.0), sharey=True)
    for index, ax in enumerate(axes):
        shear = np.asarray(result.shear_mode_shapes[index])
        frame = np.asarray(result.frame_mode_shapes[index])
        ax.plot(np.r_[0.0, shear], np.r_[0, floors], "o-", label="Shear")
        ax.plot(np.r_[0.0, frame], np.r_[0, floors], "s--", label="Frame")
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.set_title(f"Mode {index + 1}")
        ax.set_xlabel("Roof-normalized displacement")
        ax.grid(True, linewidth=0.35, alpha=0.55)
    axes[0].set_ylabel("Floor")
    axes[-1].legend()
    fig.suptitle("Three-story mode-shape comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_outputs(
    frame: PortalFrameParameters,
    results: dict[str, ThreeStoryResult],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "three_story_mode_comparison.csv"
    json_path = output_dir / "three_story_comparison.json"
    period_plot = output_dir / "three_story_period_comparison.png"
    mode_plot = output_dir / "three_story_mode_shapes.png"
    _write_csv(csv_path, results)
    _write_json(json_path, frame, results)
    _write_period_plot(period_plot, results["reference"])
    _write_mode_plot(mode_plot, results["reference"])
    return [csv_path, json_path, period_plot, mode_plot]


def _print_summary(results: dict[str, ThreeStoryResult]) -> None:
    print("Step 03 - Three-story shear-building comparison")
    print("Units: kN, m, s")
    for case_name, result in results.items():
        print()
        print(f"Case: {case_name}; story stiffness={result.story_stiffness:.6f} kN/m")
        print(f"{'mode':>4s} {'T_shear(s)':>12s} {'T_frame(s)':>12s} {'diff(%)':>11s} {'MAC':>10s}")
        for mode in result.modes:
            print(
                f"{mode.mode:4d} {mode.shear_period:12.6f} "
                f"{mode.frame_period:12.6f} "
                f"{mode.period_difference_percent:11.4f} "
                f"{mode.mass_weighted_mac:10.6f}"
            )


def main() -> None:
    frame = PortalFrameParameters()
    results = run_sensitivity_cases(frame)
    validate_result(results["reference"])
    paths = write_outputs(frame, results)
    _print_summary(results)
    print()
    print("Verification status: PASS")
    for path in paths:
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
