"""Step 04: verification of three-story, multi-bay 2D steel frames.

The 1-, 2-, and 3-bay frames use the same assumed member properties, bay
width, story height, and total floor weight.  Horizontal DOFs on each floor
are tied, while vertical translations and joint rotations remain independent.

For each frame, three static unit-load cases form the floor flexibility matrix
C.  The condensed floor stiffness Kc = inv(C) is then used in an independent
three-DOF eigenvalue solution and compared with the OpenSees eigenanalysis.

Units: kN, m, s.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from matplotlib.patches import Circle
import numpy as np
# pyrefly: ignore [missing-import]
import openseespy.opensees as ops

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.step01_static_stiffness import PROJECT_ROOT, PortalFrameParameters


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "verification" / "step04_three_story_multibay_frame"
)


@dataclass(frozen=True)
class MultiBayParameters:
    stories: int = 3
    bay_counts: tuple[int, int, int] = (1, 2, 3)
    floor_weights: tuple[float, float, float] = (200.0, 200.0, 200.0)
    gravity: float = 9.80665
    beam_inertia_factor: float = 1.0
    axial_area_factor: float = 1.0

    @property
    def floor_masses(self) -> np.ndarray:
        return np.asarray(self.floor_weights, dtype=float) / self.gravity


@dataclass(frozen=True)
class MultiBayModeResult:
    mode: int
    condensed_eigenvalue: float
    opensees_eigenvalue: float
    condensed_period: float
    opensees_period: float
    period_error_percent: float
    mass_weighted_mac: float


@dataclass(frozen=True)
class MultiBayResult:
    bay_count: int
    node_count: int
    element_count: int
    column_count: int
    beam_count: int
    roof_lateral_stiffness: float
    flexibility_matrix: list[list[float]]
    condensed_stiffness_matrix: list[list[float]]
    mass_matrix: list[list[float]]
    condensed_mode_shapes: list[list[float]]
    opensees_mode_shapes: list[list[float]]
    modes: list[MultiBayModeResult]
    total_input_mass: float
    total_free_mass_x: float
    mass_error_percent: float
    stiffness_symmetry_error: float
    maximum_tied_dof_difference: float
    maximum_eigen_residual: float


def node_tag(level: int, column: int, bay_count: int) -> int:
    return level * (bay_count + 1) + column + 1


def build_multibay_frame(
    bay_count: int,
    frame: PortalFrameParameters,
    parameters: MultiBayParameters,
    include_mass: bool,
) -> tuple[list[list[int]], list[int]]:
    """Build a three-story elastic frame and return floor nodes and masters."""

    if bay_count < 1:
        raise ValueError("bay_count must be at least one")
    if parameters.stories != 3:
        raise ValueError("This verification step requires exactly three stories")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    all_levels: list[list[int]] = []
    for level in range(parameters.stories + 1):
        level_nodes = []
        for column in range(bay_count + 1):
            tag = node_tag(level, column, bay_count)
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
    column_area = frame.column.area * parameters.axial_area_factor
    beam_area = frame.beam.area * parameters.axial_area_factor
    column_inertia = frame.column.second_moment_of_area
    beam_inertia = (
        frame.beam.second_moment_of_area * parameters.beam_inertia_factor
    )

    element_tag = 1
    for story in range(parameters.stories):
        for column in range(bay_count + 1):
            ops.element(
                "elasticBeamColumn",
                element_tag,
                all_levels[story][column],
                all_levels[story + 1][column],
                column_area,
                frame.elastic_modulus,
                column_inertia,
                1,
            )
            element_tag += 1
        for bay in range(bay_count):
            ops.element(
                "elasticBeamColumn",
                element_tag,
                all_levels[story + 1][bay],
                all_levels[story + 1][bay + 1],
                beam_area,
                frame.elastic_modulus,
                beam_inertia,
                1,
            )
            element_tag += 1

    floor_nodes = all_levels[1:]
    master_nodes = [nodes[0] for nodes in floor_nodes]
    for floor_index, nodes in enumerate(floor_nodes):
        master = nodes[0]
        for constrained in nodes[1:]:
            ops.equalDOF(master, constrained, 1)
        if include_mass:
            nodal_mass = float(parameters.floor_masses[floor_index]) / len(nodes)
            for tag in nodes:
                ops.mass(tag, nodal_mass, 0.0, 0.0)

    return floor_nodes, master_nodes


def calculate_floor_flexibility(
    bay_count: int,
    frame: PortalFrameParameters,
    parameters: MultiBayParameters,
) -> np.ndarray:
    """Obtain C_ij from three independent unit floor-load analyses."""

    flexibility = np.zeros((3, 3), dtype=float)
    for loaded_floor in range(3):
        _, masters = build_multibay_frame(
            bay_count, frame, parameters, include_mass=False
        )
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(masters[loaded_floor], 1.0, 0.0, 0.0)

        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-12, 20)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        status = ops.analyze(1)
        if status != 0:
            raise RuntimeError(
                f"Static analysis failed for {bay_count} bays, floor {loaded_floor + 1}"
            )
        flexibility[:, loaded_floor] = [
            float(ops.nodeDisp(master, 1)) for master in masters
        ]
    return flexibility


def _normalize_modes_at_roof(modes: np.ndarray) -> np.ndarray:
    normalized = np.asarray(modes, dtype=float).copy()
    for index in range(normalized.shape[1]):
        roof_component = normalized[-1, index]
        if abs(roof_component) < 1.0e-14:
            raise RuntimeError("Cannot normalize a mode with zero roof component")
        normalized[:, index] /= roof_component
    return normalized


def solve_condensed_eigenproblem(
    stiffness: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    inverse_sqrt_mass = np.diag(1.0 / np.sqrt(np.diag(mass)))
    transformed = inverse_sqrt_mass @ stiffness @ inverse_sqrt_mass
    eigenvalues, transformed_modes = np.linalg.eigh(transformed)
    modes = _normalize_modes_at_roof(inverse_sqrt_mass @ transformed_modes)

    residuals = []
    for index, eigenvalue in enumerate(eigenvalues):
        vector = modes[:, index]
        residual = stiffness @ vector - eigenvalue * mass @ vector
        residuals.append(
            np.linalg.norm(residual) / np.linalg.norm(stiffness @ vector)
        )
    return eigenvalues, modes, float(max(residuals))


def _mass_weighted_mac(
    first: np.ndarray,
    second: np.ndarray,
    mass: np.ndarray,
) -> float:
    numerator = float(first.T @ mass @ second) ** 2
    denominator = float(first.T @ mass @ first) * float(second.T @ mass @ second)
    return numerator / denominator


def run_multibay_case(
    bay_count: int,
    frame: PortalFrameParameters | None = None,
    parameters: MultiBayParameters | None = None,
) -> MultiBayResult:
    frame = frame or PortalFrameParameters()
    parameters = parameters or MultiBayParameters()

    flexibility = calculate_floor_flexibility(bay_count, frame, parameters)
    stiffness = np.linalg.inv(flexibility)
    stiffness_scale = np.linalg.norm(stiffness)
    stiffness_symmetry_error = float(
        np.linalg.norm(stiffness - stiffness.T) / stiffness_scale
    )
    roof_lateral_stiffness = 1.0 / float(flexibility[-1, -1])
    mass = np.diag(parameters.floor_masses)
    condensed_eigenvalues, condensed_modes, maximum_residual = (
        solve_condensed_eigenproblem(stiffness, mass)
    )

    floor_nodes, masters = build_multibay_frame(
        bay_count, frame, parameters, include_mass=True
    )
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    opensees_eigenvalues = np.asarray(
        ops.eigen("-fullGenLapack", 3), dtype=float
    )
    if np.any(opensees_eigenvalues <= 0.0):
        raise RuntimeError(f"Non-positive eigenvalue for {bay_count}-bay frame")

    opensees_modes = np.empty((3, 3), dtype=float)
    maximum_tied_difference = 0.0
    for mode_index in range(3):
        for floor_index, nodes in enumerate(floor_nodes):
            master_value = float(ops.nodeEigenvector(masters[floor_index], mode_index + 1, 1))
            opensees_modes[floor_index, mode_index] = master_value
            for tag in nodes[1:]:
                constrained_value = float(ops.nodeEigenvector(tag, mode_index + 1, 1))
                maximum_tied_difference = max(
                    maximum_tied_difference,
                    abs(master_value - constrained_value),
                )
    opensees_modes = _normalize_modes_at_roof(opensees_modes)

    modal = ops.modalProperties("-return")
    total_free_mass_x = float(modal["totalFreeMass"][0])
    total_input_mass = float(np.sum(parameters.floor_masses))

    mode_results = []
    for index in range(3):
        condensed_period = 2.0 * math.pi / math.sqrt(
            float(condensed_eigenvalues[index])
        )
        opensees_period = 2.0 * math.pi / math.sqrt(
            float(opensees_eigenvalues[index])
        )
        mode_results.append(
            MultiBayModeResult(
                mode=index + 1,
                condensed_eigenvalue=float(condensed_eigenvalues[index]),
                opensees_eigenvalue=float(opensees_eigenvalues[index]),
                condensed_period=condensed_period,
                opensees_period=opensees_period,
                period_error_percent=(
                    100.0
                    * (opensees_period - condensed_period)
                    / condensed_period
                ),
                mass_weighted_mac=_mass_weighted_mac(
                    condensed_modes[:, index], opensees_modes[:, index], mass
                ),
            )
        )

    column_count = parameters.stories * (bay_count + 1)
    beam_count = parameters.stories * bay_count
    return MultiBayResult(
        bay_count=bay_count,
        node_count=(parameters.stories + 1) * (bay_count + 1),
        element_count=column_count + beam_count,
        column_count=column_count,
        beam_count=beam_count,
        roof_lateral_stiffness=roof_lateral_stiffness,
        flexibility_matrix=flexibility.tolist(),
        condensed_stiffness_matrix=stiffness.tolist(),
        mass_matrix=mass.tolist(),
        condensed_mode_shapes=condensed_modes.T.tolist(),
        opensees_mode_shapes=opensees_modes.T.tolist(),
        modes=mode_results,
        total_input_mass=total_input_mass,
        total_free_mass_x=total_free_mass_x,
        mass_error_percent=(
            100.0 * (total_free_mass_x - total_input_mass) / total_input_mass
        ),
        stiffness_symmetry_error=stiffness_symmetry_error,
        maximum_tied_dof_difference=maximum_tied_difference,
        maximum_eigen_residual=maximum_residual,
    )


def run_multibay_verification(
    frame: PortalFrameParameters | None = None,
    parameters: MultiBayParameters | None = None,
) -> dict[int, MultiBayResult]:
    frame = frame or PortalFrameParameters()
    parameters = parameters or MultiBayParameters()
    return {
        bay_count: run_multibay_case(bay_count, frame, parameters)
        for bay_count in parameters.bay_counts
    }


def validate_results(results: dict[int, MultiBayResult]) -> None:
    if sorted(results) != [1, 2, 3]:
        raise AssertionError("Expected 1-, 2-, and 3-bay results")
    for bay_count, result in results.items():
        if abs(result.mass_error_percent) > 1.0e-10:
            raise AssertionError(f"Mass error for {bay_count}-bay frame")
        if result.stiffness_symmetry_error > 1.0e-10:
            raise AssertionError(f"Condensed stiffness is not symmetric for {bay_count} bays")
        if result.maximum_tied_dof_difference > 1.0e-12:
            raise AssertionError(f"Floor X constraints failed for {bay_count} bays")
        if result.maximum_eigen_residual > 1.0e-11:
            raise AssertionError(f"Eigen residual is excessive for {bay_count} bays")
        for mode in result.modes:
            if abs(mode.period_error_percent) > 1.0e-8:
                raise AssertionError(
                    f"Condensed/OpenSees period mismatch for {bay_count} bays, mode {mode.mode}"
                )
            if mode.mass_weighted_mac < 1.0 - 1.0e-10:
                raise AssertionError(
                    f"Condensed/OpenSees mode mismatch for {bay_count} bays, mode {mode.mode}"
                )

    stiffnesses = [results[count].roof_lateral_stiffness for count in (1, 2, 3)]
    if not stiffnesses[0] < stiffnesses[1] < stiffnesses[2]:
        raise AssertionError("Roof lateral stiffness does not increase with bay count")


def _write_csv(path: Path, results: dict[int, MultiBayResult]) -> None:
    rows = []
    for bay_count, result in results.items():
        for mode in result.modes:
            rows.append(
                {
                    "bay_count": bay_count,
                    "node_count": result.node_count,
                    "element_count": result.element_count,
                    "roof_lateral_stiffness": result.roof_lateral_stiffness,
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
    parameters: MultiBayParameters,
    results: dict[int, MultiBayResult],
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
            "multibay_parameters": asdict(parameters),
            "mass_comparison_basis": (
                "Total floor weight is fixed at 200 kN for every bay count; "
                "mass per node decreases as the number of columns increases."
            ),
        },
        "derived_values": {
            f"{bay_count}_bay": asdict(result)
            for bay_count, result in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_model_figure(
    path: Path,
    frame: PortalFrameParameters,
    parameters: MultiBayParameters,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2))
    for ax, bay_count in zip(axes, parameters.bay_counts):
        width = bay_count * frame.bay_width
        height = parameters.stories * frame.story_height
        for column in range(bay_count + 1):
            x = column * frame.bay_width
            ax.plot([x, x], [0.0, height], color="#2673b8", linewidth=2.6)
            ax.plot([x - 0.45, x + 0.45], [0.0, 0.0], color="#222222", linewidth=2.5)
            for hatch_x in np.linspace(x - 0.35, x + 0.35, 4):
                ax.plot([hatch_x, hatch_x - 0.18], [0.0, -0.25], color="#555555", linewidth=0.7)
        for story in range(1, parameters.stories + 1):
            y = story * frame.story_height
            ax.plot([0.0, width], [y, y], color="#e07a22", linewidth=2.6)
            for column in range(bay_count + 1):
                x = column * frame.bay_width
                ax.add_patch(
                    Circle(
                        (x, y),
                        0.16,
                        facecolor="#b23b3b",
                        edgecolor="#7d2222",
                        zorder=4,
                    )
                )
            ax.text(
                width + 0.35,
                y,
                rf"$u_{story}$",
                va="center",
                fontsize=9,
            )
        nodal_mass = parameters.floor_masses[0] / (bay_count + 1)
        ax.set_title(
            f"{bay_count}-bay frame\n"
            f"{nodal_mass:.3f} kN s$^2$/m per floor node"
        )
        ax.set_xlim(-0.8, width + 1.4)
        ax.set_ylim(-0.6, height + 0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.suptitle(
        "Three-story multi-bay 2D frames (assumed inputs: L=6 m, h=4 m, W=200 kN/floor)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary_plot(path: Path, results: dict[int, MultiBayResult]) -> None:
    bay_counts = np.array([1, 2, 3])
    stiffnesses = [results[count].roof_lateral_stiffness for count in bay_counts]
    first_periods = [results[count].modes[0].opensees_period for count in bay_counts]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))

    bars = axes[0].bar(bay_counts, stiffnesses, color="#2673b8", width=0.58)
    axes[0].set_xticks(bay_counts)
    axes[0].set_xlabel("Number of bays")
    axes[0].set_ylabel("Roof lateral stiffness (kN/m)")
    axes[0].set_title("Unit roof-load stiffness")
    axes[0].grid(axis="y", linewidth=0.35, alpha=0.55)
    for bar, value in zip(bars, stiffnesses):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].plot(bay_counts, first_periods, "o-", color="#b94a48", linewidth=2.0)
    axes[1].set_xticks(bay_counts)
    axes[1].set_xlabel("Number of bays")
    axes[1].set_ylabel("First-mode period (s)")
    axes[1].set_title("Fixed total floor mass")
    axes[1].grid(True, linewidth=0.35, alpha=0.55)
    for count, value in zip(bay_counts, first_periods):
        axes[1].text(count, value, f" {value:.3f}", va="bottom", fontsize=9)
    fig.suptitle("Effect of bay count on lateral stiffness and first-mode period", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_period_plot(path: Path, results: dict[int, MultiBayResult]) -> None:
    modes = np.arange(1, 4, dtype=float)
    width = 0.24
    colors = ["#2673b8", "#e07a22", "#4c8c4a"]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for index, bay_count in enumerate((1, 2, 3)):
        values = [mode.opensees_period for mode in results[bay_count].modes]
        offset = (index - 1) * width
        bars = ax.bar(
            modes + offset,
            values,
            width,
            color=colors[index],
            label=f"{bay_count} bay" if bay_count == 1 else f"{bay_count} bays",
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
    ax.set_xticks(modes, ["Mode 1", "Mode 2", "Mode 3"])
    ax.set_xlabel("Mode")
    ax.set_ylabel("Natural period (s)")
    ax.set_title("Natural periods of three-story multi-bay frames")
    ax.grid(axis="y", linewidth=0.35, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_mode_plot(path: Path, results: dict[int, MultiBayResult]) -> None:
    floors = np.arange(1, 4)
    colors = ["#2673b8", "#e07a22", "#4c8c4a"]
    markers = ["o", "s", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0), sharey=True)
    for mode_index, ax in enumerate(axes):
        for color, marker, bay_count in zip(colors, markers, (1, 2, 3)):
            shape = np.asarray(results[bay_count].opensees_mode_shapes[mode_index])
            ax.plot(
                np.r_[0.0, shape],
                np.r_[0, floors],
                marker=marker,
                color=color,
                linewidth=1.8,
                label=f"{bay_count} bay" if bay_count == 1 else f"{bay_count} bays",
            )
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.set_title(f"Mode {mode_index + 1}")
        ax.set_xlabel("Roof-normalized displacement")
        ax.grid(True, linewidth=0.35, alpha=0.55)
    axes[0].set_ylabel("Floor")
    axes[-1].legend(fontsize=8.5)
    fig.suptitle("Mode shapes of three-story multi-bay frames", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_stiffness_matrix_plot(
    path: Path,
    results: dict[int, MultiBayResult],
) -> None:
    matrices = [np.asarray(results[count].condensed_stiffness_matrix) for count in (1, 2, 3)]
    maximum = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    image = None
    for ax, bay_count, matrix in zip(axes, (1, 2, 3), matrices):
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        ax.set_xticks(range(3), ["u1", "u2", "u3"])
        ax.set_yticks(range(3), ["u1", "u2", "u3"])
        ax.set_title(f"{bay_count}-bay $K_c$ (kN/m)")
        for row in range(3):
            for column in range(3):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(matrix[row, column]) > 0.5 * maximum else "black",
                )
    assert image is not None
    fig.suptitle("Statically condensed floor stiffness matrices", fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.87, bottom=0.12, top=0.82, wspace=0.32)
    colorbar_axis = fig.add_axes([0.90, 0.20, 0.015, 0.56])
    fig.colorbar(image, cax=colorbar_axis, label="Stiffness (kN/m)")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_outputs(
    frame: PortalFrameParameters,
    parameters: MultiBayParameters,
    results: dict[int, MultiBayResult],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "multibay_mode_results.csv",
        output_dir / "multibay_results.json",
        output_dir / "multibay_models.png",
        output_dir / "multibay_stiffness_summary.png",
        output_dir / "multibay_period_comparison.png",
        output_dir / "multibay_mode_shapes.png",
        output_dir / "condensed_stiffness_matrices.png",
    ]
    _write_csv(paths[0], results)
    _write_json(paths[1], frame, parameters, results)
    _write_model_figure(paths[2], frame, parameters)
    _write_summary_plot(paths[3], results)
    _write_period_plot(paths[4], results)
    _write_mode_plot(paths[5], results)
    _write_stiffness_matrix_plot(paths[6], results)
    return paths


def _print_summary(results: dict[int, MultiBayResult]) -> None:
    print("Step 04 - Three-story multi-bay 2D frame verification")
    print("Units: kN, m, s")
    for bay_count, result in results.items():
        print()
        print(
            f"{bay_count} bay(s): nodes={result.node_count}, "
            f"elements={result.element_count}, "
            f"roof stiffness={result.roof_lateral_stiffness:.6f} kN/m"
        )
        print(f"{'mode':>4s} {'T_condensed':>13s} {'T_OpenSees':>13s} {'error(%)':>12s} {'MAC':>10s}")
        for mode in result.modes:
            print(
                f"{mode.mode:4d} {mode.condensed_period:13.7f} "
                f"{mode.opensees_period:13.7f} "
                f"{mode.period_error_percent:12.3e} "
                f"{mode.mass_weighted_mac:10.7f}"
            )


def main() -> None:
    frame = PortalFrameParameters()
    parameters = MultiBayParameters()
    results = run_multibay_verification(frame, parameters)
    validate_results(results)
    paths = write_outputs(frame, parameters, results)
    _print_summary(results)
    print()
    print("Verification status: PASS")
    for path in paths:
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
