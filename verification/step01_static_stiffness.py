"""Step 01: static lateral-stiffness verification of a one-story portal frame.

The benchmark follows the textbook assumptions used to obtain the two limits

    k = 6 E Ic / h^3       (zero beam flexural stiffness)
    k = 24 E Ic / h^3      (infinite beam flexural stiffness)

and the finite-beam-stiffness solution

    k = (24 E Ic / h^3) * (12 rho + 1) / (12 rho + 4)
    rho = (E Ib / L) / (2 E Ic / h) = Ib h / (2 Ic L).

The analytical expression neglects axial deformation.  The primary benchmark
therefore uses very large axial areas while retaining the reference bending
inertias.  A separate assumed-section case quantifies the small difference
introduced by member axial deformation.

Units: kN, m, s.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "verification" / "step01_static_stiffness"


@dataclass(frozen=True)
class ElasticSection2D:
    """Section properties required by a 2D elastic beam-column element."""

    area: float
    second_moment_of_area: float


@dataclass(frozen=True)
class PortalFrameParameters:
    """Geometry and elastic properties of the verification portal frame."""

    elastic_modulus: float = 2.0e8  # 200 GPa in kN/m^2
    bay_width: float = 6.0
    story_height: float = 4.0
    lateral_load: float = 1.0
    # Round, assumed properties are used deliberately.  They are not tied to
    # a catalogue section or to the sections used in the previous project.
    beam: ElasticSection2D = ElasticSection2D(0.010, 2.0e-4)
    column: ElasticSection2D = ElasticSection2D(0.020, 4.0e-4)


@dataclass(frozen=True)
class StiffnessResult:
    case_name: str
    beam_inertia_factor: float
    axial_area_factor: float
    rho: float
    theoretical_stiffness: float
    opensees_stiffness: float
    theoretical_coefficient: float
    opensees_coefficient: float
    relative_error_percent: float
    left_displacement: float
    right_displacement: float
    average_displacement: float
    displacement_asymmetry: float
    horizontal_reaction_residual: float


def relative_beam_column_stiffness(
    beam_inertia: float,
    column_inertia: float,
    bay_width: float,
    story_height: float,
) -> float:
    """Return rho = (EI_b/L)/(2EI_c/h); E cancels for one material."""

    return beam_inertia * story_height / (2.0 * column_inertia * bay_width)


def theoretical_portal_stiffness(
    elastic_modulus: float,
    column_inertia: float,
    story_height: float,
    rho: float,
) -> float:
    """Closed-form sway stiffness with member axial deformation neglected."""

    base = 24.0 * elastic_modulus * column_inertia / story_height**3
    return base * (12.0 * rho + 1.0) / (12.0 * rho + 4.0)


def flexible_beam_limit(params: PortalFrameParameters) -> float:
    return (
        6.0
        * params.elastic_modulus
        * params.column.second_moment_of_area
        / params.story_height**3
    )


def rigid_beam_limit(params: PortalFrameParameters) -> float:
    return (
        24.0
        * params.elastic_modulus
        * params.column.second_moment_of_area
        / params.story_height**3
    )


def build_portal_frame_model(
    params: PortalFrameParameters,
    beam_inertia_factor: float,
    axial_area_factor: float,
    lateral_load: float | None = None,
) -> None:
    if beam_inertia_factor <= 0.0:
        raise ValueError("beam_inertia_factor must be positive")
    if axial_area_factor <= 0.0:
        raise ValueError("axial_area_factor must be positive")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    width = params.bay_width
    height = params.story_height
    ops.node(1, 0.0, 0.0)
    ops.node(2, width, 0.0)
    ops.node(3, 0.0, height)
    ops.node(4, width, height)

    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)
    ops.geomTransf("Linear", 1)

    column_area = params.column.area * axial_area_factor
    beam_area = params.beam.area * axial_area_factor
    column_inertia = params.column.second_moment_of_area
    beam_inertia = params.beam.second_moment_of_area * beam_inertia_factor

    ops.element(
        "elasticBeamColumn",
        1,
        1,
        3,
        column_area,
        params.elastic_modulus,
        column_inertia,
        1,
    )
    ops.element(
        "elasticBeamColumn",
        2,
        2,
        4,
        column_area,
        params.elastic_modulus,
        column_inertia,
        1,
    )
    ops.element(
        "elasticBeamColumn",
        3,
        3,
        4,
        beam_area,
        params.elastic_modulus,
        beam_inertia,
        1,
    )

    if lateral_load is not None:
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        half_load = lateral_load / 2.0
        ops.load(3, half_load, 0.0, 0.0)
        ops.load(4, half_load, 0.0, 0.0)


def run_static_case(
    case_name: str,
    params: PortalFrameParameters,
    beam_inertia_factor: float,
    axial_area_factor: float = 1.0e6,
) -> StiffnessResult:
    """Build, solve, and check one lateral-load stiffness case."""

    build_portal_frame_model(
        params,
        beam_inertia_factor,
        axial_area_factor,
        lateral_load=params.lateral_load,
    )

    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 20)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    status = ops.analyze(1)
    if status != 0:
        raise RuntimeError(f"OpenSees analysis failed for {case_name}: code {status}")

    left_displacement = float(ops.nodeDisp(3, 1))
    right_displacement = float(ops.nodeDisp(4, 1))
    average_displacement = 0.5 * (left_displacement + right_displacement)
    if average_displacement <= 0.0:
        raise RuntimeError(f"Non-positive sway displacement for {case_name}")

    opensees_stiffness = params.lateral_load / average_displacement
    beam_inertia = params.beam.second_moment_of_area * beam_inertia_factor
    column_inertia = params.column.second_moment_of_area
    rho = relative_beam_column_stiffness(
        beam_inertia,
        column_inertia,
        params.bay_width,
        params.story_height,
    )
    theoretical_stiffness = theoretical_portal_stiffness(
        params.elastic_modulus,
        column_inertia,
        params.story_height,
        rho,
    )
    stiffness_scale = params.elastic_modulus * column_inertia / params.story_height**3

    ops.reactions()
    horizontal_reaction_residual = (
        float(ops.nodeReaction(1, 1))
        + float(ops.nodeReaction(2, 1))
        + params.lateral_load
    )

    return StiffnessResult(
        case_name=case_name,
        beam_inertia_factor=beam_inertia_factor,
        axial_area_factor=axial_area_factor,
        rho=rho,
        theoretical_stiffness=theoretical_stiffness,
        opensees_stiffness=opensees_stiffness,
        theoretical_coefficient=theoretical_stiffness / stiffness_scale,
        opensees_coefficient=opensees_stiffness / stiffness_scale,
        relative_error_percent=(
            100.0
            * (opensees_stiffness - theoretical_stiffness)
            / theoretical_stiffness
        ),
        left_displacement=left_displacement,
        right_displacement=right_displacement,
        average_displacement=average_displacement,
        displacement_asymmetry=abs(left_displacement - right_displacement),
        horizontal_reaction_residual=horizontal_reaction_residual,
    )


def run_benchmark_cases(
    params: PortalFrameParameters | None = None,
) -> list[StiffnessResult]:
    params = params or PortalFrameParameters()
    return [
        run_static_case("near_zero_beam_stiffness", params, 1.0e-8),
        run_static_case("reference_beam_inertia", params, 1.0),
        run_static_case("near_rigid_beam", params, 1.0e8),
        run_static_case(
            "assumed_sections_with_axial_deformation",
            params,
            1.0,
            axial_area_factor=1.0,
        ),
    ]


def validate_results(results: Iterable[StiffnessResult]) -> None:
    results = list(results)
    idealized = [result for result in results if result.axial_area_factor > 1.0]
    if len(idealized) != 3:
        raise AssertionError("Expected three axial-rigid benchmark cases")

    for result in idealized:
        if abs(result.relative_error_percent) > 1.0e-3:
            raise AssertionError(
                f"{result.case_name} exceeds analytical error tolerance: "
                f"{result.relative_error_percent:.6g}%"
            )
        if result.displacement_asymmetry > 1.0e-12:
            raise AssertionError(f"{result.case_name} is not symmetric")
        # The 1e6 axial-area multiplier makes the benchmark matrix strongly
        # scaled, so allow a small round-off residual relative to the load.
        if abs(result.horizontal_reaction_residual) > 1.0e-7:
            raise AssertionError(f"{result.case_name} fails reaction equilibrium")

    coefficients = [result.opensees_coefficient for result in idealized]
    if not coefficients[0] < coefficients[1] < coefficients[2]:
        raise AssertionError("Portal stiffness does not increase with beam inertia")


def _write_csv(path: Path, results: Iterable[StiffnessResult]) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: Path,
    params: PortalFrameParameters,
    results: Iterable[StiffnessResult],
) -> None:
    payload = {
        "units": {"force": "kN", "length": "m", "time": "s"},
        "parameters": {
            "elastic_modulus": params.elastic_modulus,
            "bay_width": params.bay_width,
            "story_height": params.story_height,
            "lateral_load": params.lateral_load,
            "beam": {
                **asdict(params.beam),
                "area": params.beam.area,
                "second_moment_of_area": params.beam.second_moment_of_area,
            },
            "column": {
                **asdict(params.column),
                "area": params.column.area,
                "second_moment_of_area": params.column.second_moment_of_area,
            },
        },
        "limits": {
            "flexible_beam_stiffness": flexible_beam_limit(params),
            "rigid_beam_stiffness": rigid_beam_limit(params),
            "normalized_flexible_limit": 6.0,
            "normalized_rigid_limit": 24.0,
        },
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_stiffness_plot(path: Path, params: PortalFrameParameters) -> None:
    factors = np.logspace(-6.0, 6.0, 49)
    theoretical_coefficients = []
    opensees_coefficients = []
    column_inertia = params.column.second_moment_of_area
    scale = params.elastic_modulus * column_inertia / params.story_height**3

    for factor in factors:
        beam_inertia = params.beam.second_moment_of_area * factor
        rho = relative_beam_column_stiffness(
            beam_inertia,
            column_inertia,
            params.bay_width,
            params.story_height,
        )
        theoretical_coefficients.append(
            theoretical_portal_stiffness(
                params.elastic_modulus,
                column_inertia,
                params.story_height,
                rho,
            )
            / scale
        )
        result = run_static_case(
            "plot_sweep",
            params,
            float(factor),
            axial_area_factor=1.0e6,
        )
        opensees_coefficients.append(result.opensees_coefficient)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.semilogx(factors, theoretical_coefficients, color="black", linewidth=2.0, label="Theory")
    ax.semilogx(
        factors,
        opensees_coefficients,
        color="#2673b8",
        linestyle="none",
        marker="o",
        markersize=3.2,
        label="OpenSees",
    )
    ax.axhline(6.0, color="#b94a48", linestyle="--", linewidth=1.1, label="Flexible-beam limit")
    ax.axhline(24.0, color="#3c8d40", linestyle="--", linewidth=1.1, label="Rigid-beam limit")
    ax.set_xlabel("Beam inertia factor, $I_b/I_{b,reference}$")
    ax.set_ylabel(r"Normalized stiffness, $k/(EI_c/h^3)$")
    ax.set_title("One-story one-bay portal-frame stiffness verification")
    ax.set_ylim(4.5, 25.5)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.55)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_outputs(
    results: list[StiffnessResult],
    params: PortalFrameParameters,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "static_stiffness_results.csv"
    json_path = output_dir / "static_stiffness_results.json"
    plot_path = output_dir / "static_stiffness_verification.png"
    _write_csv(csv_path, results)
    _write_json(json_path, params, results)
    _write_stiffness_plot(plot_path, params)
    return [csv_path, json_path, plot_path]


def _print_summary(
    results: Iterable[StiffnessResult], params: PortalFrameParameters
) -> None:
    print("Step 01 - One-story one-bay static stiffness verification")
    print("Units: kN, m, s")
    print(
        "Textbook limits [kN/m]: "
        f"flexible beam={flexible_beam_limit(params):.6f}, "
        f"rigid beam={rigid_beam_limit(params):.6f}"
    )
    print()
    print(
        f"{'case':42s} {'rho':>12s} {'k_theory':>14s} "
        f"{'k_OpenSees':>14s} {'error(%)':>12s} {'coefficient':>13s}"
    )
    for result in results:
        print(
            f"{result.case_name:42s} "
            f"{result.rho:12.5e} "
            f"{result.theoretical_stiffness:14.6f} "
            f"{result.opensees_stiffness:14.6f} "
            f"{result.relative_error_percent:12.6f} "
            f"{result.opensees_coefficient:13.6f}"
        )


def main() -> None:
    params = PortalFrameParameters()
    results = run_benchmark_cases(params)
    validate_results(results)
    paths = write_outputs(results, params)
    _print_summary(results, params)
    print()
    print("Verification status: PASS")
    for path in paths:
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
