"""STEP08: 대칭 3D 골조에서 바닥판 면내강성의 영향을 분리해 비교한다.

질량편심은 0으로 유지한다. 모든 모델의 골조, 질량, 중력하중은 동일하다.
강체는 운동학적 구속이며, 나머지 두 조건은 유한한 면내강성의 셸 요소다.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verification.step07_3d_reference_frame import (
    PROJECT_ROOT, Parameters3D, build_model, configure_static, gravity_analysis,
)

OUTPUT_DIR = PROJECT_ROOT / "results/verification/step08_diaphragm_inplane_sensitivity"
PARAMETERS = Parameters3D()
THICKNESS_M = 0.12                  # 가정한 바닥판 유효 두께
SHELL_POISSON = 0.20                # 가정한 바닥판 포아송비
# 이 값들은 특정 실제 바닥판을 나타내지 않는 연구용 면내강성 수준이다.
SHELL_E_KN_M2 = {"semirigid": 2.0e7, "flexible": 2.0e4,
                 "rigid_limit": 2.0e10}
STATIC_ROOF_FORCE_KN = 12.0
EIGEN_MODES_FINITE = 24


def build_case(kind: str, p: Parameters3D = PARAMETERS,
               member_scale: float = 1.0, transformation: str = "PDelta"):
    """STEP07의 동일 골조를 생성한 다음 유한강성 조건에만 셸을 추가한다."""
    if kind == "rigid":
        model = build_model(p, transformation=transformation,
                            member_scale=member_scale)
        model["shell_elements"] = []
        return model
    return build_finite_case(kind, p, member_scale=member_scale,
                             transformation=transformation)


def build_finite_case(kind: str, p: Parameters3D = PARAMETERS, subdivisions=1,
                      shell_e=None, member_scale: float = 1.0,
                      transformation: str = "PDelta"):
    """구속식 없이 골조와 셸을 조립한다. 바닥 질량은 골조 절점에만 둔다."""
    if kind not in (*SHELL_E_KN_M2, "floor_beams_only"):
        raise ValueError(kind)
    if transformation not in ("Linear", "PDelta"):
        raise ValueError("지원하지 않는 기둥 기하변환")
    member_scale = float(member_scale)
    if not np.isfinite(member_scale) or member_scale <= 0:
        raise ValueError("부재 단면 특성 배율은 유한한 양수여야 한다")
    if not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("경간 분할 수는 양의 정수여야 합니다")
    if kind == "floor_beams_only" and subdivisions != 1:
        raise ValueError("셸 없는 비교 모델은 기본 격자를 사용합니다")
    from verification.step01_static_stiffness import PortalFrameParameters
    f = PortalFrameParameters()
    xy = np.array([(i*p.bay_length-p.lengths[0]/2, j*p.bay_length-p.lengths[1]/2)
                   for j in range(p.bays_y+1) for i in range(p.bays_x+1)])
    count = len(xy)
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    levels, coords, edges = [], {}, []
    for level in range(p.stories+1):
        row = []
        for index, (x, y) in enumerate(xy):
            tag = 1+level*count+index
            coord = (float(x), float(y), level*p.story_height)
            ops.node(tag, *coord)
            coords[tag] = coord
            row.append(tag)
            if level == 0:
                ops.fix(tag, 1, 1, 1, 1, 1, 1)
            else:
                # STEP07과 동일한 절점별 수평질량. 셸 자체의 질량은 0이다.
                m = p.floor_mass/count
                ops.mass(tag, m, m, 0.0, 0.0, 0.0, 0.0)
        levels.append(row)
    ops.geomTransf(transformation, 1, 0.0, 1.0, 0.0)
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)
    g = f.elastic_modulus/(2*(1+p.poisson))
    def member(a, b, column):
        if column:
            area, iy, iz, torsion, transf = (f.column.area, p.column_iy,
                f.column.second_moment_of_area, p.column_j, 1)
        else:
            area, iy, iz, torsion, transf = (f.beam.area,
                f.beam.second_moment_of_area, p.beam_iz, p.beam_j, 2)
        # 연구용 단면 민감도: 보·기둥 A, Iy, Iz, J를 동일 배율로 조정한다.
        # 층 질량과 중력하중은 바꾸지 않으므로 규격 단면 변경을 뜻하지 않는다.
        area, iy, iz, torsion = (member_scale*value for value in
                                 (area, iy, iz, torsion))
        tag = len(edges)+1
        ops.element("elasticBeamColumn", tag, a, b, area,
                    f.elastic_modulus, g, torsion, iy, iz, transf)
        edges.append((a, b))
    shell_elements = []
    if kind != "floor_beams_only":
        # Ep_modifier=0으로 셸의 면외 휨·횡전단강성을 제거한다. STEP07 골조의
        # 수직 강성을 셸이 추가하지 않고, E*h 변화가 면내강성에만 작용한다.
        modulus = SHELL_E_KN_M2[kind] if shell_e is None else float(shell_e)
        if modulus <= 0:
            raise ValueError("셸 탄성계수는 양수여야 합니다")
        ops.section("ElasticMembranePlateSection", 1, modulus,
                    SHELL_POISSON, THICKNESS_M, 0.0, 0.0)
    shell_levels = []
    for level in range(1, p.stories+1):
        nodes = levels[level]
        for index in range(count):
            member(levels[level-1][index], nodes[index], True)
        r = subdivisions
        nx, ny = p.bays_x*r+1, p.bays_y*r+1
        grid = {}
        for j in range(ny):
            for i in range(nx):
                if i % r == 0 and j % r == 0:
                    tag = nodes[(j//r)*(p.bays_x+1)+i//r]
                else:
                    tag = 10000+level*100000+j*nx+i
                    position = (i*p.bay_length/r-p.lengths[0]/2,
                                j*p.bay_length/r-p.lengths[1]/2, level*p.story_height)
                    ops.node(tag, *position)
                    coords[tag] = position
                    if i % r != 0 and j % r != 0:
                        # 면외강성 0인 막 내부 절점에는 사용하지 않는 자유도를 남기지 않는다.
                        # 셸 내부의 면외 자유도만 제거한다. 보 위 절점은 모두 자유다.
                        ops.fix(tag, 0, 0, 1, 1, 1, 0)
                grid[i, j] = tag
        shell_levels.append(list(grid.values()))
        # 보도 동일한 격자점에서 분할하여 보-바닥판 접합을 연속적으로 유지한다.
        for j in range(0, ny, r):
            for i in range(nx-1):
                member(grid[i, j], grid[i+1, j], False)
        for j in range(ny-1):
            for i in range(0, nx, r):
                member(grid[i, j], grid[i, j+1], False)
        if kind != "floor_beams_only":
            for j in range(ny-1):
                for i in range(nx-1):
                    corners = [grid[i, j], grid[i+1, j], grid[i+1, j+1], grid[i, j+1]]
                    tag = 1000000+len(shell_elements)+1
                    ops.element("ShellMITC4", tag, *corners, 1)
                    shell_elements.append(corners)
    return {"levels": levels, "masters": [], "xy": xy, "coords": coords,
            "edges": edges, "shell_elements": shell_elements, "shell_levels": shell_levels}


def create_and_gravitate(kind, p=PARAMETERS):
    model = build_case(kind, p)
    gravity = gravity_analysis(model, p, 1.0)
    if gravity["relative_error"] > 1e-8:
        raise AssertionError("중력 반력 평형 실패")
    return model, gravity


def horizontal_mass_vector(p=PARAMETERS):
    return np.full(2*(p.bays_x+1)*(p.bays_y+1)*p.stories,
                   p.floor_mass/((p.bays_x+1)*(p.bays_y+1)))


def joint_horizontal_array(model, accessor):
    return np.array([[accessor(tag, dof) for dof in (1, 2)]
                     for floor in model["levels"][1:] for tag in floor])


def modes_of(kind, p=PARAMETERS):
    model, gravity = create_and_gravitate(kind, p)
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    count = 3*p.stories if kind == "rigid" else EIGEN_MODES_FINITE
    eigenvalues = np.asarray(ops.eigen("-fullGenLapack", count), dtype=float)
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        raise AssertionError(f"{kind}: 유효하지 않은 고유치")
    md = horizontal_mass_vector(p)
    vectors = []
    for n in range(1, count+1):
        raw = joint_horizontal_array(model,
            lambda tag, dof: ops.nodeEigenvector(tag, n, dof)).ravel()
        norm = np.sqrt(np.sum(md*raw**2))
        if norm <= 1e-12:
            raise AssertionError(f"{kind}: 질량이 없는 모드")
        vectors.append(raw/norm)
    vectors = np.asarray(vectors)
    modal_properties = ops.modalProperties("-return")
    return {"periods": (2*np.pi/np.sqrt(eigenvalues)).tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "joint_modes": vectors.tolist(), "gravity": gravity,
            "total_horizontal_mass": [float(sum(md[0::2])), float(sum(md[1::2]))],
            "opensees_total_free_mass": modal_properties["totalFreeMass"],
            "opensees_center_of_mass": modal_properties["centerOfMass"]}


def match_rigid_modes(rigid, finite, p=PARAMETERS):
    """모드 순번이 바뀌어도 동일한 질량 절점의 가중 MAC으로 대응시킨다."""
    a, b = np.asarray(rigid["joint_modes"]), np.asarray(finite["joint_modes"])
    md = horizontal_mass_vector(p)
    mac = (a*md) @ b.T
    mac = mac**2
    row, col = linear_sum_assignment(-mac)
    return [{"rigid_mode": int(i+1), "finite_mode": int(j+1),
             "mac": float(mac[i, j]),
             "period_s": float(finite["periods"][j]),
             "roof_inplane_mode_ratio": roof_mode_distortion_ratio(b[j], p),
             "period_change_pct": 100*(finite["periods"][j]/rigid["periods"][i]-1)}
            for i, j in sorted(zip(row, col))]


def roof_mode_distortion_ratio(vector, p=PARAMETERS):
    """지붕층 모드의 강체평면 운동에서 벗어나는 성분의 노름 비율이다."""
    n = (p.bays_x+1)*(p.bays_y+1)
    roof = np.asarray(vector)[-2*n:]
    xy = np.array([(i*p.bay_length-p.lengths[0]/2,
                    j*p.bay_length-p.lengths[1]/2)
                   for j in range(p.bays_y+1) for i in range(p.bays_x+1)])
    geometry = np.zeros((2*n, 3))
    geometry[0::2, 0], geometry[1::2, 1] = 1.0, 1.0
    geometry[0::2, 2], geometry[1::2, 2] = -xy[:, 1], xy[:, 0]
    fitted = geometry @ np.linalg.lstsq(geometry, roof, rcond=None)[0]
    return float(np.linalg.norm(roof-fitted)/np.linalg.norm(roof))


def roof_static(kind, p=PARAMETERS, direction=1):
    """최상층에 균등 수평력을 가해 층 내 상대변위의 크기를 측정한다."""
    model, _ = create_and_gravitate(kind, p)
    nodes = model["levels"][-1]
    initial = joint_horizontal_array(model, ops.nodeDisp)
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for tag in nodes:
        load = [0.0]*6
        load[direction-1] = STATIC_ROOF_FORCE_KN/len(nodes)
        ops.load(tag, *load)
    configure_static(1.0)
    if ops.analyze(1) != 0:
        raise RuntimeError(f"{kind}: 수평 정적해석 실패")
    disp = joint_horizontal_array(model, ops.nodeDisp)-initial
    roof = disp[-len(nodes):, direction-1]
    # 강체 평면의 병진·회전을 최소제곱으로 제거한 잔차를 면내 왜곡으로 정의한다.
    xy = model["xy"]
    geometry = np.zeros((2*len(nodes), 3))
    geometry[0::2, 0], geometry[1::2, 1] = 1.0, 1.0
    geometry[0::2, 2], geometry[1::2, 2] = -xy[:, 1], xy[:, 0]
    fit, *_ = np.linalg.lstsq(geometry, disp[-len(nodes):].ravel(), rcond=None)
    residual = disp[-len(nodes):]-geometry.dot(fit).reshape(-1, 2)
    return {"roof_mean_displacement_mm": float(np.mean(roof)*1000),
            "roof_range_mm": float(np.ptp(roof)*1000),
            "roof_rms_inplane_distortion_mm": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))*1000),
            "roof_joint_xy": disp[-len(nodes):].tolist(),
            "roof_rigid_fit": fit.tolist()}


def run_verification(p=PARAMETERS):
    results = {}
    for kind in ("rigid", "rigid_limit", "semirigid", "flexible", "floor_beams_only"):
        results[kind] = {"modal": modes_of(kind, p),
                         "static_x": roof_static(kind, p, 1),
                         "static_y": roof_static(kind, p, 2)}
    rigid = results["rigid"]["modal"]
    for kind in (*SHELL_E_KN_M2, "floor_beams_only"):
        results[kind]["match_to_rigid"] = match_rigid_modes(rigid, results[kind]["modal"], p)
    # STEP07의 전체 중력하중 및 총 질량을 모든 비교 모델에서 동일하게 유지한다.
    expected = p.floor_mass*p.stories
    for kind, r in results.items():
        np.testing.assert_allclose(r["modal"]["opensees_total_free_mass"][:2], expected, rtol=1e-9)
        np.testing.assert_allclose(r["modal"]["opensees_center_of_mass"][:2], 0.0, atol=1e-9)
    # 큰 셸 강성의 해가 구속식의 강체 해에 가까워지는지 수치적으로 확인한다.
    limit = results["rigid_limit"]["match_to_rigid"]
    if max(abs(item["period_change_pct"]) for item in limit) > 0.001:
        raise AssertionError("유한강성 셸의 강체 극한 수렴 실패")
    return results


def write_outputs(results, p=PARAMETERS, directory=OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    info = {"scope": "대칭·질량편심 0·PDelta 중력 선행; 바닥판 면내강성만 변화",
            "parameters": asdict(p), "assumed_shell_thickness_m": THICKNESS_M,
            "assumed_shell_poisson": SHELL_POISSON,
            "assumed_shell_E_kN_m2": SHELL_E_KN_M2,
            "shell_mass_density": 0.0, "shell_out_of_plane_modifier": 0.0,
            "static_roof_force_kN": STATIC_ROOF_FORCE_KN,
            "results": results}
    (directory/"step08_results.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory/"step08_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "E_shell_kN_m2", "roof_range_X_mm", "roof_distortion_X_mm",
                         "roof_range_Y_mm", "roof_distortion_Y_mm", "mode_1_period_s",
                         "mode_2_period_s", "mode_3_period_s"])
        for kind, r in results.items():
            writer.writerow([kind, SHELL_E_KN_M2.get(kind, "constraint"),
                r["static_x"]["roof_range_mm"], r["static_x"]["roof_rms_inplane_distortion_mm"],
                r["static_y"]["roof_range_mm"], r["static_y"]["roof_rms_inplane_distortion_mm"],
                *r["modal"]["periods"][:3]])
    # 세 연구모델의 고유치를 강체 기준 모드에 맞추어 저장한다.
    # 유한강성 모델에는 새 국부모드가 나타나므로 전체 모드 번호도 함께 기록한다.
    primary = ("rigid", "semirigid", "flexible")
    with (directory/"step08_eigen_comparison.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rigid_reference_mode", "condition", "actual_mode_order",
                         "eigenvalue_rad2_s2", "period_s", "mass_weighted_MAC",
                         "roof_inplane_mode_distortion_ratio"])
        for n in range(9):
            for kind in primary:
                if kind == "rigid":
                    index, mac, distortion = n, 1.0, 0.0
                else:
                    item = results[kind]["match_to_rigid"][n]
                    index, mac = item["finite_mode"]-1, item["mac"]
                    distortion = item["roof_inplane_mode_ratio"]
                writer.writerow([n+1, kind, index+1,
                    results[kind]["modal"]["eigenvalues"][index],
                    results[kind]["modal"]["periods"][index], mac, distortion])
    labels = ["rigid", "rigid_limit", "semirigid", "flexible"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, name in enumerate(("static_x", "static_y")):
        values = [results[k][name]["roof_rms_inplane_distortion_mm"] for k in labels]
        axes[i].bar(labels, values, color=["#414a55", "#3687bd", "#daa544", "#c45b52"])
        axes[i].set(ylabel="Roof in-plane distortion RMS (mm)", title=f"Uniform roof {name[-1].upper()} load")
        axes[i].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(directory/"step08_inplane_distortion.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind in labels:
        ax.plot(range(1, 10), results[kind]["modal"]["periods"][:9], marker="o", label=kind)
    ax.set(xlabel="Eigenmode order", ylabel="Period (s)", title="Diaphragm stiffness: first nine periods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory/"step08_periods.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for kind, color in (("rigid", "#414a55"), ("semirigid", "#daa544"),
                        ("flexible", "#c45b52")):
        if kind == "rigid":
            indices = list(range(9))
        else:
            indices = [item["finite_mode"]-1 for item in results[kind]["match_to_rigid"]]
        eigenvalues = np.asarray(results[kind]["modal"]["eigenvalues"])[indices]
        periods = np.asarray(results[kind]["modal"]["periods"])[indices]
        axes[0].plot(range(1, 10), eigenvalues, "o-", color=color, label=kind)
        axes[1].plot(range(1, 10), periods, "o-", color=color, label=kind)
    axes[0].set(xlabel="Rigid reference mode", ylabel="Eigenvalue λ = ω² (rad²/s²)",
                title="Matched eigenvalues", yscale="log")
    axes[1].set(xlabel="Rigid reference mode", ylabel="Natural period T (s)",
                title="Matched periods")
    for ax in axes:
        ax.set_xticks(range(1, 10))
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("STEP08: eigen analysis of the three diaphragm models\n"
                 "Modes paired by mass-weighted MAC; actual mode orders are in CSV")
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(directory/"step08_three_model_eigen_comparison.png", dpi=180)
    plt.close(fig)
    # 각 강체 모드에 대응하는 유한강성 모드를 MAC으로 찾아 비교한다.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for kind, color in (("rigid_limit", "#3687bd"), ("semirigid", "#daa544"),
                        ("flexible", "#c45b52"), ("floor_beams_only", "#555555")):
        matched = results[kind]["match_to_rigid"]
        x = np.array([item["rigid_mode"] for item in matched])
        axes[0].plot(x, [item["period_change_pct"] for item in matched],
                     marker="o", color=color, label=kind)
        axes[1].plot(x, [item["mac"] for item in matched],
                     marker="o", color=color, label=kind)
    axes[0].set(xlabel="Matched rigid reference mode", ylabel="Period change (%)",
                title="Matched periods (not raw mode order)")
    axes[1].set(xlabel="Matched rigid reference mode", ylabel="Mass-weighted MAC",
                title="Matched mode-shape similarity", ylim=(0, 1.04))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/"step08_matched_modes.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind in ("rigid_limit", "semirigid", "flexible", "floor_beams_only"):
        matched = results[kind]["match_to_rigid"]
        ax.plot([item["rigid_mode"] for item in matched],
                [100*item["roof_inplane_mode_ratio"] for item in matched],
                marker="o", label=kind)
    ax.set(xlabel="Matched rigid reference mode", ylabel="Roof in-plane modal distortion (%)",
           title="Modal deformation beyond rigid-body floor motion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/"step08_modal_distortion.png", dpi=180)
    plt.close(fig)

    # 같은 골조 절점과 보 격자를 세 조건이 어떻게 연결하는지 평면도로 나타낸다.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    nx, ny = p.bays_x+1, p.bays_y+1
    xy = np.array([(i*p.bay_length-p.lengths[0]/2,
                    j*p.bay_length-p.lengths[1]/2)
                   for j in range(ny) for i in range(nx)])
    for ax, kind, color in zip(axes, ("rigid", "semirigid", "flexible"),
                               ("#c2d9ea", "#e4b65c", "#f1d4ce")):
        for j in range(p.bays_y):
            for i in range(p.bays_x):
                ax.add_patch(plt.Rectangle(
                    (i*p.bay_length-p.lengths[0]/2, j*p.bay_length-p.lengths[1]/2),
                    p.bay_length, p.bay_length, facecolor=color, alpha=.5,
                    edgecolor="none"))
        for j in range(ny):
            ax.plot(xy[j*nx:(j+1)*nx, 0], xy[j*nx:(j+1)*nx, 1], color="#34495e")
        for i in range(nx):
            ax.plot(xy[i::nx, 0], xy[i::nx, 1], color="#34495e")
        ax.scatter(xy[:, 0], xy[:, 1], color="#263745", s=15)
        caption = ("rigidDiaphragm constraint" if kind == "rigid" else
                   f"ShellMITC4, E={SHELL_E_KN_M2[kind]:.0e} kN/m²")
        ax.set(title=f"{kind}\n{caption}", xlabel="X (m)", ylabel="Y (m)",
               xlim=(-10, 10), ylim=(-7, 7))
        ax.set_aspect("equal")
    fig.suptitle("Same frame grid and nodal mass; assumed shell E changes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(directory/"step08_model_cases.png", dpi=180)
    plt.close(fig)

    # 비틀림 모드 2·5를 지붕 평면에서 비교한다. 그림 변위의 배율은 임의이며
    # 각 모드를 지붕 최대 수평변위 1.5 m로 정규화한다.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2))
    xy = np.array([(i*p.bay_length-p.lengths[0]/2,
                    j*p.bay_length-p.lengths[1]/2)
                   for j in range(p.bays_y+1) for i in range(p.bays_x+1)])
    mass = horizontal_mass_vector(p)
    for row, reference_mode in enumerate((2, 5)):
        ref = np.asarray(results["rigid"]["modal"]["joint_modes"][reference_mode-1])
        for col, kind in enumerate(("rigid", "semirigid", "flexible")):
            index = reference_mode-1 if kind == "rigid" else next(
                item["finite_mode"]-1 for item in results[kind]["match_to_rigid"]
                if item["rigid_mode"] == reference_mode)
            vector = np.asarray(results[kind]["modal"]["joint_modes"][index])
            if np.dot(ref*mass, vector) < 0:
                vector = -vector
            motion = vector[-2*len(xy):].reshape(-1, 2)
            motion *= 1.5/max(np.linalg.norm(v) for v in motion)
            moved = xy+motion
            ax = axes[row, col]
            for j in range(p.bays_y+1):
                for i in range(p.bays_x):
                    a, b = j*(p.bays_x+1)+i, j*(p.bays_x+1)+i+1
                    ax.plot(*xy[[a, b]].T, color="0.7", linestyle="--", linewidth=0.8)
                    ax.plot(*moved[[a, b]].T, color="#286ba5", linewidth=1.0)
            for j in range(p.bays_y):
                for i in range(p.bays_x+1):
                    a, b = j*(p.bays_x+1)+i, (j+1)*(p.bays_x+1)+i
                    ax.plot(*xy[[a, b]].T, color="0.7", linestyle="--", linewidth=0.8)
                    ax.plot(*moved[[a, b]].T, color="#286ba5", linewidth=1.0)
            ax.quiver(xy[:, 0], xy[:, 1], motion[:, 0], motion[:, 1],
                      angles="xy", scale_units="xy", scale=1, color="#c45b52", width=0.004)
            ratio = roof_mode_distortion_ratio(vector, p)
            ax.set(xlabel="X (m)", ylabel="Y (m)", xlim=(-11, 11), ylim=(-8, 8),
                   title=f"Rigid mode {reference_mode} → {kind} mode {index+1}\n"
                         f"T={results[kind]['modal']['periods'][index]:.4f} s; distortion={ratio:.1%}")
            ax.set_aspect("equal")
    fig.suptitle("Roof torsional modes: dashed original, blue displaced, red modal direction\n"
                 "Each roof mode independently amplified to 1.5 m", fontsize=12)
    fig.subplots_adjust(left=.06, right=.98, bottom=.08, top=.86, hspace=.7, wspace=.26)
    fig.savefig(directory/"step08_roof_mode_shapes.png", dpi=180)
    plt.close(fig)


def main():
    results = run_verification()
    write_outputs(results)
    for kind, result in results.items():
        print(kind, "T1-3", np.round(result["modal"]["periods"][:3], 6),
              "roof distortion X/Y (mm)",
              round(result["static_x"]["roof_rms_inplane_distortion_mm"], 7),
              round(result["static_y"]["roof_rms_inplane_distortion_mm"], 7))
    print("STEP08:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
