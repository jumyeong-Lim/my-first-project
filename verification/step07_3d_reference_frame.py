"""STEP07: 대칭 3D 탄성 골조의 강체 바닥·질량·중력·병진/비틀림 검증.

좌표는 X/Y 수평, Z 수직이며 단위는 kN, m, s이다.
각 층의 독립 평면 자유도는 [Ux, Uy, Rz]이다. 실절점 Uz/Rx/Ry는
자유롭게 남기되 질량을 주지 않는다. 이 조건에서 정적 응축의 고유치가
전체 모델의 유한 고유치와 같아야 한다. 실험 검증이 아닌 수치 일관성 검증이다.
"""
from __future__ import annotations

import csv
import json
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

OUTPUT_DIR = PROJECT_ROOT / "results/verification/step07_3d_reference_frame"
DOFS = (1, 2, 6)
FAMILIES = ("X", "Y", "Rz")


@dataclass(frozen=True)
class Parameters3D:
    """단면·치수·중량은 연구자가 가정한 검증값이며 실건물 설계값이 아니다."""
    stories: int = 3
    bays_x: int = 3
    bays_y: int = 2
    bay_length: float = 6.0
    story_height: float = 4.0
    floor_weight: float = 200.0
    gravity: float = 9.80665
    poisson: float = 0.30
    # 기존 2D의 보 I는 3D 보의 수직 휨 Iy로, 기둥 I는 Iz로 계승한다.
    # 아래 약축 관성과 Saint-Venant 비틀림상수 J는 추가 가정이다.
    beam_iz: float = 1.0e-4
    beam_j: float = 2.0e-6
    column_iy: float = 2.0e-4
    column_j: float = 8.0e-6

    @property
    def floor_mass(self):
        return self.floor_weight / self.gravity

    @property
    def lengths(self):
        return self.bays_x * self.bay_length, self.bays_y * self.bay_length


def layout(p: Parameters3D):
    """평면 중심을 원점으로 두고 절점질량으로 층의 회전관성을 계산한다."""
    lx, ly = p.lengths
    xy = np.array([(i*p.bay_length-lx/2, j*p.bay_length-ly/2)
                   for j in range(p.bays_y+1) for i in range(p.bays_x+1)])
    # 층 절점에 동일 질량을 배치한다. 연속 균등 바닥질량과는 다른 가정이다.
    nodal_mass = p.floor_mass / len(xy)
    polar = nodal_mass * np.sum(xy**2)
    # 회전관성은 단면의 비틀림상수 J와 다른 값이다: 단위 kN*s^2*m.
    mass_diagonal = np.tile([p.floor_mass, p.floor_mass, polar], p.stories)
    return xy, mass_diagonal


def build_model(p: Parameters3D, transformation="Linear", mass_location="nodes",
                member_scale: float = 1.0):
    """3D 골조를 생성한다. 질량 입력법 두 가지의 동등성도 검증한다."""
    if transformation not in ("Linear", "PDelta") or mass_location not in ("nodes", "master"):
        raise ValueError("지원하지 않는 변환 또는 질량 배치 방식")
    member_scale = float(member_scale)
    if not np.isfinite(member_scale) or member_scale <= 0:
        raise ValueError("부재 단면 특성 배율은 유한한 양수여야 한다")
    if min(p.stories, p.bays_x, p.bays_y) < 1 or min(p.floor_weight, p.bay_length, p.story_height) <= 0:
        raise ValueError("층수·경간수·중량·길이는 양수여야 한다")
    f = PortalFrameParameters()
    xy, md = layout(p)
    count = len(xy)
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    levels = []
    coords = {}
    for level in range(p.stories+1):
        nodes = []
        for index, (x, y) in enumerate(xy):
            tag = 1 + level*count + index
            coord = (float(x), float(y), level*p.story_height)
            ops.node(tag, *coord)
            coords[tag] = coord
            nodes.append(tag)
            if level == 0:
                ops.fix(tag, 1, 1, 1, 1, 1, 1)
        levels.append(nodes)

    # 기둥: 로컬 x=전역 Z, 로컬 y=전역 X, 로컬 z=전역 Y.
    # 따라서 X 횡변위에는 기둥 Iz, Y 횡변위에는 Iy가 주로 작용한다.
    ops.geomTransf(transformation, 1, 0.0, 1.0, 0.0)
    # 수평 보: 로컬 z=전역 Z. 보의 수직 휨에 Iy를 사용한다.
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)
    shear_modulus = f.elastic_modulus / (2*(1+p.poisson))
    edges = []

    def add_element(a, b, column):
        if column:
            area, iy, iz, torsion, trans = (f.column.area, p.column_iy,
                f.column.second_moment_of_area, p.column_j, 1)
        else:
            area, iy, iz, torsion, trans = (f.beam.area,
                f.beam.second_moment_of_area, p.beam_iz, p.beam_j, 2)
        # 연구용 가정 단면의 A, Iy, Iz, J만 같은 비율로 변화시킨다.
        # 이는 규격 단면의 변경이 아니며 층 질량과 중력하중은 그대로 유지한다.
        area, iy, iz, torsion = (member_scale*value for value in
                                 (area, iy, iz, torsion))
        ops.element("elasticBeamColumn", len(edges)+1, a, b, area,
                    f.elastic_modulus, shear_modulus, torsion, iy, iz, trans)
        edges.append((a, b))

    masters = []
    for level in range(1, p.stories+1):
        nodes = levels[level]
        for index in range(count):
            add_element(levels[level-1][index], nodes[index], True)
        for j in range(p.bays_y+1):
            for i in range(p.bays_x):
                a = j*(p.bays_x+1)+i
                add_element(nodes[a], nodes[a+1], False)
        for j in range(p.bays_y):
            for i in range(p.bays_x+1):
                a = j*(p.bays_x+1)+i
                add_element(nodes[a], nodes[a+p.bays_x+1], False)
        master = (p.stories+1)*count + level
        ops.node(master, 0.0, 0.0, level*p.story_height)
        # 기준절점은 평면 운동만 담당한다. 이 fix는 실제 골조의 Uz/Rx/Ry를 묶지 않는다.
        ops.fix(master, 0, 0, 1, 1, 1, 0)
        ops.rigidDiaphragm(3, master, *nodes)
        masters.append(master)
        if mass_location == "nodes":
            # 수평 질량만 입력한다. 구속변환이 절점 위치에 따른 회전관성을 만든다.
            for tag in nodes:
                m = p.floor_mass/count
                ops.mass(tag, m, m, 0.0, 0.0, 0.0, 0.0)
        else:
            # 검증 전용 대안: 실절점 질량 없이 기준절점에 M,M,Jz를 한 번만 입력.
            ops.mass(master, p.floor_mass, p.floor_mass, 0.0, 0.0, 0.0, float(md[2]))
    return {"levels": levels, "masters": masters, "xy": xy,
            "mass": md, "coords": coords, "edges": edges}


def configure_static(step):
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 40)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", step)
    ops.analysis("Static")


def gravity_analysis(model, p, factor):
    """층중량을 수직하중으로 입력하고 중력평형 후 하중을 고정한다."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nodes in model["levels"][1:]:
        for tag in nodes:
            ops.load(tag, 0.0, 0.0, -factor*p.floor_weight/len(nodes), 0.0, 0.0, 0.0)
    configure_static(0.1)
    if ops.analyze(10) != 0:
        raise RuntimeError("중력해석 수렴 실패")
    ops.reactions()
    reaction = sum(ops.nodeReaction(tag, 3) for tag in model["levels"][0])
    expected = p.floor_weight*p.stories*factor
    ops.loadConst("-time", 0.0)
    return {"applied_kN": expected, "reaction_kN": reaction,
            "relative_error": abs(reaction-expected)/max(1.0, expected)}


def floor_vector(model):
    return np.array([ops.nodeDisp(tag, dof) for tag in model["masters"] for dof in DOFS])


def eigen_result(model, p):
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    values = np.array(ops.eigen("-fullGenLapack", 3*p.stories))
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise AssertionError("유한 양의 고유치 확인 실패")
    modes = np.array([[ops.nodeEigenvector(tag, n+1, dof)
        for tag in model["masters"] for dof in DOFS] for n in range(len(values))])
    md = model["mass"]
    modes /= np.sqrt(np.sum(modes**2*md, axis=1))[:, None]
    energies = np.array([[np.sum(md[k::3]*v[k::3]**2) for k in range(3)] for v in modes])
    # 단위가 다른 병진(m)과 회전(rad)의 원시 크기를 직접 비교하지 않는다.
    # 질량 정규화된 모드의 운동에너지 비중으로 X/Y/Rz 성격을 판별한다.
    participation = np.array([[np.sum(md[k::3]*v[k::3])**2 / np.sum(md[k::3])
        for k in range(3)] for v in modes])
    compatibility = 0.0
    for n, vector in enumerate(modes):
        raw = np.array([ops.nodeEigenvector(t, n+1, d) for t in model["masters"] for d in DOFS])
        scale = np.sqrt(np.sum(raw**2*md))
        for level, nodes in enumerate(model["levels"][1:]):
            ux, uy, rz = vector[3*level:3*level+3]
            for tag, (x, y) in zip(nodes, model["xy"]):
                expected = np.array([ux-y*rz, uy+x*rz, rz])
                actual = np.array([ops.nodeEigenvector(tag, n+1, d) for d in DOFS])/scale
                compatibility = max(compatibility, float(np.max(np.abs(actual-expected))))
    modal = ops.modalProperties("-return")
    return {"eigenvalues": values.tolist(), "periods": (2*np.pi/np.sqrt(values)).tolist(),
            "modes_mass_normalized": modes.tolist(), "family": [FAMILIES[i] for i in energies.argmax(axis=1)],
            "energy_fraction_X_Y_Rz": energies.tolist(), "participation_ratio_X_Y_Rz": participation.tolist(),
            "compatibility_error": compatibility, "opensees_total_free_mass": modal["totalFreeMass"],
            "opensees_center_of_mass": modal["centerOfMass"]}


def condensed_stiffness(p, transformation, factor, amplitude=0.01):
    """각 층 Fx/Fy/Mz의 ±미소 하중으로 중력평형점의 유연도를 수치 미분한다.

    각 하중은 새 모델의 같은 중력평형점에서 시작한다. 하중을 누적하지 않는다.
    C의 병진/회전 행·열은 서로 단위가 다르며, 저장된 자유도 순서를 따라야 한다.
    """
    size = 3*p.stories
    c = np.zeros((size, size))
    for column in range(size):
        for sign in (-1, 1):
            model = build_model(p, transformation)
            gravity_analysis(model, p, factor)
            initial = floor_vector(model)
            ops.timeSeries("Linear", 2)
            ops.pattern("Plain", 2, 2)
            force = [0.0]*6
            force[DOFS[column % 3]-1] = sign*amplitude
            ops.load(model["masters"][column//3], *force)
            configure_static(1.0)
            if ops.analyze(1) != 0:
                raise RuntimeError("유연도 산정 정적해석 실패")
            c[:, column] += sign*(floor_vector(model)-initial)/(2*amplitude)
    # 원래 행렬의 상호성 오차를 검증한 후에만 대칭화한다.
    symmetry = np.max(np.abs(c-c.T))/np.max(np.abs(c))
    if symmetry > 1e-7:
        raise AssertionError(f"유연도 상호성 확인 실패: {symmetry}")
    c = (c+c.T)/2
    return c, np.linalg.solve(c, np.eye(size)), symmetry


def run_case(p, transformation="Linear", factor=1.0):
    model = build_model(p, transformation)
    balance = gravity_analysis(model, p, factor)
    eigen = eigen_result(model, p)
    md = model["mass"]
    # 기준절점 질량 M,M,Jz와 분포 절점질량의 결과가 같아야 한다.
    alternative = build_model(p, transformation, "master")
    gravity_analysis(alternative, p, factor)
    equivalent = eigen_result(alternative, p)
    c, k, symmetry = condensed_stiffness(p, transformation, factor)
    normalized_k = k/np.sqrt(np.outer(md, md))
    condensed_values, condensed_vectors = np.linalg.eigh(normalized_k)
    condensed_modes = condensed_vectors.T/np.sqrt(md)
    full_modes = np.array(eigen["modes_mass_normalized"])
    # 양쪽 모드가 질량 정규화되어 있으므로 내적 제곱이 질량가중 MAC이다.
    mode_mac = np.sum(condensed_modes*md*full_modes, axis=1)**2
    full_values = np.array(eigen["eigenvalues"])
    error = float(np.max(np.abs(condensed_values/full_values-1)))
    mass_error = float(np.max(np.abs(np.array(equivalent["eigenvalues"])/full_values-1)))
    # 대칭 모델의 서로 다른 방향 사이 결합항은 수치적으로 0이어야 한다.
    scaled = k/np.sqrt(np.outer(np.diag(k), np.diag(k)))
    mask = np.arange(len(md))[:, None] % 3 != np.arange(len(md))[None, :] % 3
    coupling = float(np.max(np.abs(scaled[mask])))
    result = {"parameters": asdict(p), "transformation": transformation, "gravity_factor": factor,
        "gravity": balance, "floor_mass": p.floor_mass, "floor_polar_mass_inertia": float(md[2]),
        "floor_mass_centers_xy": [[0.0, 0.0]]*p.stories,
        "mass_diagonal": md.tolist(), "flexibility": c.tolist(), "condensed_stiffness": k.tolist(),
        "condensed_mode_mac": mode_mac.tolist(),
        "condensed_eigenvalues": condensed_values.tolist(), "condensed_eigen_relative_error": error,
        "equivalent_mass_eigen_relative_error": mass_error, "flexibility_symmetry_error": symmetry,
        "normalized_cross_direction_coupling": coupling, **eigen}
    validate_case(result)
    return result


def validate_case(r):
    """조건은 임의 결과를 맞추기 위한 보정이 아닌 수치 검증 허용오차이다."""
    for key in ("condensed_eigen_relative_error", "equivalent_mass_eigen_relative_error",
                "normalized_cross_direction_coupling", "compatibility_error"):
        if r[key] > 1e-6:
            raise AssertionError(f"{key}: {r[key]}")
    if r["gravity"]["relative_error"] > 1e-9:
        raise AssertionError("수직 반력 평형 오류")
    expected = r["floor_mass"]*r["parameters"]["stories"]
    np.testing.assert_allclose(r["opensees_total_free_mass"][:2], expected, rtol=1e-10)
    np.testing.assert_allclose(r["opensees_total_free_mass"][5],
                              r["floor_polar_mass_inertia"]*r["parameters"]["stories"], rtol=1e-10)
    np.testing.assert_allclose(r["opensees_center_of_mass"][:2], 0.0, atol=1e-10)
    np.testing.assert_allclose(r["condensed_mode_mac"], 1.0, atol=1e-8)
    np.testing.assert_allclose(np.sum(r["participation_ratio_X_Y_Rz"], axis=0), 1.0, atol=1e-8)
    if np.min(np.max(r["energy_fraction_X_Y_Rz"], axis=1)) < 0.999:
        raise AssertionError("대칭 기준모델의 방향별 모드 분리 실패")


def run_verification():
    """소형 모델→3층 기준 모델 순서로 중력 전후와 PDelta 효과를 확인한다."""
    results = {}
    for label, p in (("small", Parameters3D(stories=1, bays_x=1, bays_y=1)),
                     ("reference", Parameters3D())):
        for trans, factor in (("Linear", 0.0), ("PDelta", 0.0), ("Linear", 1.0), ("PDelta", 1.0)):
            key = f"{label}_{trans}_g{factor:g}"
            results[key] = run_case(p, trans, factor)
        base = np.array(results[f"{label}_Linear_g0"]["eigenvalues"])
        for key in (f"{label}_PDelta_g0", f"{label}_Linear_g1"):
            np.testing.assert_allclose(results[key]["eigenvalues"], base, rtol=1e-9)
        if np.any(np.array(results[f"{label}_PDelta_g1"]["periods"]) <= results[f"{label}_Linear_g1"]["periods"]):
            raise AssertionError("중력축력에 따른 주기 증가 확인 실패")
    return results


def draw_frame(ax, model, displacements=None):
    for a, b in model["edges"]:
        points = np.array([model["coords"][a], model["coords"][b]])
        if displacements is not None:
            ax.plot(*points.T, color="0.8", linewidth=0.6)
            points += np.array([displacements[a], displacements[b]])
        ax.plot(*points.T, color="#286ba5", linewidth=1.0)
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    coords = np.array(list(model["coords"].values()))
    ax.set_box_aspect(np.maximum(np.ptp(coords, axis=0), 1))
    ax.view_init(elev=22, azim=-55)


def write_outputs(results, directory=OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"units": "kN, m, s; generalized DOFs per floor: Ux, Uy, Rz",
        "opensees_version": ops.version(), "scope": "대칭 강체 바닥, 수평질량, 재료 탄성; 바닥판 면내 유연성 및 편심은 후속 단계",
        "existing_2d_assumed_inputs": asdict(PortalFrameParameters()), "results": results}
    (directory/"step07_results.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    with (directory/"step07_modes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "mode", "family", "period_s", "mass_ratio_X", "mass_ratio_Y", "mass_ratio_Rz"])
        for key, r in results.items():
            for n, period in enumerate(r["periods"]):
                writer.writerow([key, n+1, r["family"][n], period, *r["participation_ratio_X_Y_Rz"][n]])
    p = Parameters3D()
    model = build_model(p)
    # 형상 및 연결정보도 저장하여 그림과 실제 모델 입력을 대조할 수 있게 한다.
    model_data = {"physical_nodes": model["coords"], "element_end_nodes": model["edges"],
        "floor_nodes": model["levels"][1:], "retained_nodes": model["masters"],
        "retained_coordinates": [[0, 0, i*p.story_height] for i in range(1, p.stories+1)],
        "nodal_horizontal_mass_per_direction": p.floor_mass/len(model["xy"]),
        "nodal_vertical_weight_kN": p.floor_weight/len(model["xy"]),
        "retained_node_mass_in_primary_model": 0,
        "constraint": "rigidDiaphragm(3); 기준절점 Ux,Uy,Rz만 활성; 실절점 Uz,Rx,Ry 독립",
        "section_local_axes": "기둥 local x=Z,y=X,z=Y; 수평 보 local z=Z"}
    (directory/"step07_model.json").write_text(json.dumps(model_data, ensure_ascii=False, indent=2), encoding="utf-8")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_frame(ax, model)
    for level in range(1, p.stories+1):
        ax.scatter([0], [0], [level*p.story_height], color="#bf4b45", s=40)
    ax.set_title("STEP07: symmetric 3D reference frame\nRigid floor, 200 kN/floor (assumed); red: retained nodes")
    fig.tight_layout()
    fig.savefig(directory/"step07_model.png", dpi=180)
    plt.close(fig)

    linear, pd = results["reference_Linear_g1"], results["reference_PDelta_g1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(1, 10)
    for r, offset, label in ((linear, -0.18, "Linear"), (pd, 0.18, "PDelta")):
        axes[0].bar(x+offset, r["periods"], 0.36, label=label)
    axes[0].set(xticks=x, xlabel="Mode", ylabel="Period (s)", title="After gravity: all nine horizontal modes")
    axes[0].legend()
    bottom = np.zeros(9)
    for i, family in enumerate(FAMILIES):
        values = np.array(pd["energy_fraction_X_Y_Rz"])[:, i]*100
        axes[1].bar(x, values, bottom=bottom, label=family)
        bottom += values
    axes[1].set(xticks=x, xlabel="Mode", ylabel="Modal kinetic-energy fraction (%)", title="Symmetric PDelta model: mode classification")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, 120)
    fig.tight_layout()
    fig.savefig(directory/"step07_periods_and_families.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5))
    for panel, family in enumerate(FAMILIES):
        n = pd["family"].index(family)
        mode = np.array(pd["modes_mass_normalized"][n]).reshape(-1, 3)
        disp = {tag: np.zeros(3) for tag in model["coords"]}
        for level, nodes in enumerate(model["levels"][1:]):
            ux, uy, rz = mode[level]
            for tag, (xx, yy) in zip(nodes, model["xy"]):
                disp[tag] = np.array([ux-yy*rz, uy+xx*rz, 0.0])
        scale = 1.5/max(np.linalg.norm(v) for v in disp.values())
        disp = {tag: v*scale for tag, v in disp.items()}
        ax = fig.add_subplot(1, 3, panel+1, projection="3d")
        draw_frame(ax, model, disp)
        ax.set_title(f"{family}: mode {n+1}, T={pd['periods'][n]:.4f} s")
    fig.suptitle("First mode in each family (gravity + PDelta)\nHorizontal modal motion only; arbitrary amplified scale", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(directory/"step07_translation_torsion_modes.png", dpi=180)
    plt.close(fig)

    # 1~9차 전체를 고유치 순서대로 그린다. 각 모드의 최대 수평성분을
    # 1.5 m로 독립 정규화하므로 서로 다른 모드의 실제 변위 크기를 비교하면 안 된다.
    fig = plt.figure(figsize=(17, 15))
    for n in range(9):
        mode = np.asarray(pd["modes_mass_normalized"][n]).reshape(-1, 3)
        displacement = {tag: np.zeros(3) for tag in model["coords"]}
        for level, nodes in enumerate(model["levels"][1:]):
            ux, uy, rz = mode[level]
            for tag, (xx, yy) in zip(nodes, model["xy"]):
                displacement[tag] = np.array([ux-yy*rz, uy+xx*rz, 0.0])
        scale = 1.5/max(np.linalg.norm(v) for v in displacement.values())
        displacement = {tag: v*scale for tag, v in displacement.items()}
        ax = fig.add_subplot(3, 3, n+1, projection="3d")
        draw_frame(ax, model, displacement)
        ax.set_title(f"Mode {n+1}: {pd['family'][n]}, T={pd['periods'][n]:.4f} s", fontsize=10)
        ax.tick_params(labelsize=7)
    fig.suptitle("STEP07: all nine modes in eigenvalue order (gravity + PDelta)\n"
                 "Gray: undeformed; blue: horizontal mode, independently amplified", fontsize=13)
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.91, wspace=.08, hspace=.23)
    fig.savefig(directory/"step07_all_nine_mode_shapes.png", dpi=180)
    plt.close(fig)

    # 수직 방향 모드형상의 부호변화를 3D 원근도와 독립적으로 확인한다.
    fig, axes = plt.subplots(3, 3, figsize=(11, 12), sharey=True)
    heights = np.arange(p.stories+1)*p.story_height
    for n, ax in enumerate(axes.flat):
        mode = np.asarray(pd["modes_mass_normalized"][n]).reshape(-1, 3)
        family = pd["family"][n]
        dof = FAMILIES.index(family)
        component = np.r_[0.0, mode[:, dof]]
        component /= np.max(np.abs(component))
        ax.plot(component, heights, "o-", color={"X": "#286ba5", "Y": "#c45b52", "Rz": "#78649b"}[family])
        ax.axvline(0, color="0.7", linewidth=.8)
        ax.set(xlim=(-1.15, 1.15), ylim=(0, heights[-1]),
               title=f"Mode {n+1}: {family}, T={pd['periods'][n]:.4f} s",
               xlabel="Signed modal component / peak", ylabel="Height (m)")
        ax.grid(alpha=.25)
    fig.suptitle("STEP07: vertical profiles of modes 1–9\n"
                 "Each governing component normalized independently; eigenvector sign is arbitrary")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(directory/"step07_all_nine_mode_profiles.png", dpi=180)
    plt.close(fig)

    # 평면도는 병진과 비틀림의 차이를 3D 원근도보다 명확히 보여준다.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    roof_nodes = set(model["levels"][-1])
    for ax, family in zip(axes, FAMILIES):
        n = pd["family"].index(family)
        ux, uy, rz = np.array(pd["modes_mass_normalized"][n])[-3:]
        xy = model["xy"]
        motion = np.column_stack((ux-xy[:, 1]*rz, uy+xy[:, 0]*rz))
        motion *= 1.5/np.max(np.linalg.norm(motion, axis=1))
        moved = {tag: pos+delta for tag, pos, delta in zip(model["levels"][-1], xy, motion)}
        for a, b in model["edges"]:
            if a in roof_nodes and b in roof_nodes:
                original = np.array([model["coords"][a][:2], model["coords"][b][:2]])
                ax.plot(*original.T, color="0.65", linestyle="--", linewidth=0.8)
                ax.plot(*np.array([moved[a], moved[b]]).T, color="#286ba5", linewidth=1.2)
        ax.quiver(xy[:, 0], xy[:, 1], motion[:, 0], motion[:, 1], angles="xy", scale_units="xy", scale=1, color="#b84b42")
        ax.set(title=f"{family}: mode {n+1}", xlabel="X (m)", ylabel="Y (m)")
        ax.set_aspect("equal")
    fig.suptitle("Roof plan: translation and torsion\nDashed: initial; blue: amplified modal shape; arrows: horizontal motion")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(directory/"step07_roof_mode_plans.png", dpi=180)
    plt.close(fig)
    ops.wipe()


def main():
    results = run_verification()
    write_outputs(results)
    for key, r in results.items():
        print(key, "T(s)=", np.round(r["periods"], 6), "families=", r["family"])
        print(f"  condensed error={r['condensed_eigen_relative_error']:.3e}; mass equivalence={r['equivalent_mass_eigen_relative_error']:.3e}")
    print("STEP07 PASS:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
