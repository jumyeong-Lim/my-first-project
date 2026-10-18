# ============================================================
# 3D Steel Frame Model with Rigid Diaphragm
# Unit: kN - m - sec
# Case2 : Rigid diaphragm & Lumped mass & Irregular geometry
# ============================================================

import openseespy.opensees as ops
import opsvis as opsv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULT_DIR = Path(__file__).resolve().parent / "Result_Proto2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Model initialize
# ------------------------------------------------------------
ops.wipe()
ops.model("BasicBuilder", "-ndm", 3, "-ndf", 6)


# ------------------------------------------------------------
# 2. Basic unit and constants
# ------------------------------------------------------------
g = 9.80665  # m/sec^2

# ------------------------------------------------------------
# 3. Geometry information
# ------------------------------------------------------------
nBayX = 3
nBayY = 3
nStory = 3

bay_length_x = 6.0     # m
bay_length_y = 6.0     # m
story_height = 4.0     # m

# 전체 평면 크기
total_length_x = nBayX * bay_length_x   # 18 m
total_length_y = nBayY * bay_length_y   # 18 m

# diaphragm 중심 좌표
Xc = 9.0
Yc = 9.0


# ------------------------------------------------------------
# 4. Material properties: SM460
# ------------------------------------------------------------
E = 2.0e8          # 200,000 MPa = 200 GPa = 2.0e8 kN/m^2
Fy = 460000.0      # 460 MPa = 460,000 kN/m^2
nu = 0.3           # Poisson's ratio
G = E / (2.0 * (1.0 + nu))    # Shear modulus


# ------------------------------------------------------------
# 5. Section property calculation function
# ------------------------------------------------------------
### H형강 단면 성능을 단면 치수로부터 계산한다.

def calc_h_section_properties(H, B, tw, tf):

    hw = H - 2.0 * tf               # H형강 웨브 높이 [m]
    A = 2.0 * B * tf + hw * tw      # H형강 단면적 [m^2]
    I_strong = (B * H**3 - (B - tw) * hw**3) / 12.0      # 강축 단면2차모멘트 [m^4]
    I_weak = (2.0 * tf * B**3 + hw * tw**3) / 12.0       # 약축 단면2차모멘트 [m^4]
    J = (2.0 * B * tf**3 + hw * tw**3) / 3.0             # H형강 비틀림 상수 [m^4]

    return A, I_strong, I_weak, J


# ------------------------------------------------------------
# 6. Section dimensions
# ------------------------------------------------------------

### Beam: H-400x200x8x13
beam_H = 0.400
beam_B = 0.200
beam_tw = 0.008
beam_tf = 0.013

### Column: H-350x350x12x19
column_H = 0.350
column_B = 0.350
column_tw = 0.012
column_tf = 0.019

A_beam, Iy_beam, Iz_beam, J_beam = calc_h_section_properties(
    beam_H, beam_B, beam_tw, beam_tf
)

A_column, Iy_column, Iz_column, J_column = calc_h_section_properties(
    column_H, column_B, column_tw, column_tf
)


# ------------------------------------------------------------
# 7. Node generation
# ------------------------------------------------------------
# node_tags[(floor, ix, iy)] 형태로 저장
# floor = 0: base
# floor = 1: 2F level, z = 4 m
# floor = 2: 3F level, z = 8 m
# floor = 3: Roof level, z = 12 m

node_tags = {}

for floor in range(nStory + 1):
    z = floor * story_height

    for ix in range(nBayX + 1):
        x = ix * bay_length_x

        for iy in range(nBayY + 1):
            y = iy * bay_length_y

            node_tag = 1000 * floor + 100 * ix + iy + 1
            ops.node(node_tag, x, y, z)

            node_tags[(floor, ix, iy)] = node_tag


# ------------------------------------------------------------
# 8. Boundary condition
# ------------------------------------------------------------
# 기초부 모든 노드는 고정단으로 설정한다.

for ix in range(nBayX + 1):
    for iy in range(nBayY + 1):
        base_node = node_tags[(0, ix, iy)]
        ops.fix(base_node, 1, 1, 1, 1, 1, 1)


# ------------------------------------------------------------
# 9. Geometric transformation
# ------------------------------------------------------------
# 현재는 기본 모델링 단계이므로 Linear transformation을 사용한다.
# 추후 P-Delta 효과를 고려하려면 column_transf_type을 "PDelta"로 변경할 수 있다.

column_transf_tag = 1
beam_transf_tag = 2

column_transf_type = "Linear"
beam_transf_type = "Linear"

# 기둥은 부재축이 global Z방향이므로, vecxz를 global X방향으로 둔다.
ops.geomTransf(column_transf_type, column_transf_tag, 1.0, 0.0, 0.0)

# 보는 부재축이 global X 또는 Y방향이므로, vecxz를 global Z방향으로 둔다.
ops.geomTransf(beam_transf_type, beam_transf_tag, 0.0, 0.0, 1.0)


# ------------------------------------------------------------
# 10. Column elements
# ------------------------------------------------------------
column_elements = []
ele_tag = 10000

for story in range(1, nStory + 1):
    lower_floor = story - 1
    upper_floor = story

    for ix in range(nBayX + 1):
        for iy in range(nBayY + 1):
            i_node = node_tags[(lower_floor, ix, iy)]
            j_node = node_tags[(upper_floor, ix, iy)]

            ops.element("elasticBeamColumn", ele_tag, i_node, j_node, A_column, E, G, J_column, Iy_column, Iz_column, column_transf_tag)

            column_elements.append(ele_tag)
            ele_tag += 1


# ------------------------------------------------------------
# 11. Beam elements in X direction
# ------------------------------------------------------------
beam_x_elements = []
ele_tag = 20000

for floor in range(1, nStory + 1):
    for iy in range(nBayY + 1):
        for ix in range(nBayX):
            i_node = node_tags[(floor, ix, iy)]
            j_node = node_tags[(floor, ix + 1, iy)]

            ops.element("elasticBeamColumn", ele_tag, i_node, j_node, A_beam, E, G, J_beam, Iy_beam, Iz_beam, beam_transf_tag)

            beam_x_elements.append(ele_tag)
            ele_tag += 1


# ------------------------------------------------------------
# 12. Beam elements in Y direction
# ------------------------------------------------------------
beam_y_elements = []
ele_tag = 30000

for floor in range(1, nStory + 1):
    for ix in range(nBayX + 1):
        for iy in range(nBayY):
            i_node = node_tags[(floor, ix, iy)]
            j_node = node_tags[(floor, ix, iy + 1)]

            ops.element("elasticBeamColumn", ele_tag, i_node, j_node, A_beam, E, G, J_beam, Iy_beam, Iz_beam, beam_transf_tag)

            beam_y_elements.append(ele_tag)
            ele_tag += 1


# ------------------------------------------------------------
# 13. Rigid diaphragm and floor mass
# ------------------------------------------------------------
# Mass control parameter : Lumped mass
# Convert self weight into masses : X
# Load to mass : DL Scale 1, Mass direction X,Y
# Floor diaphragm Masses : 2, 3, Roof층 모두 Mp = 200 kN/g
# 따라서 여기서는 부재 자중을 자동 질량으로 변환하지 않고, 각 층 diaphragm retained node에 X,Y방향 질량만 입력한다.
# Mp = 200 kN → 단위: kN-sec^2/m

Mp = 200.0
Mr = 0.0

diaphragm_nodes = {}

# 질량참여율 계산용 리스트/딕셔너리
mass_node_tags = []
node_mass_dict = {}

for floor in range(1, nStory + 1):
    z = floor * story_height

    retained_node = 9000 + floor
    ops.node(retained_node, Xc, Yc, z)

    # retained node는 diaphragm의 Ux, Uy, Rz만 사용한다.
    # 연결되지 않은 Uz, Rx, Ry 자유도는 고정해서 불안정 자유도를 방지한다.
    ops.fix(retained_node, 0, 0, 1, 1, 1, 0)

    mass_vector = [Mp, Mp, 0.0, 0.0, 0.0, Mr]
    ops.mass(retained_node, *mass_vector)

    mass_node_tags.append(retained_node)
    node_mass_dict[retained_node] = mass_vector

    # # X,Y방향 질량만 입력한다.
    # # Mr = 0이므로 Rz 회전질량관성은 0으로 둔다.
    # ops.mass(retained_node, Mp, Mp, 0.0, 0.0, 0.0, Mr)

    constrained_nodes = []

    for ix in range(nBayX + 1):
        for iy in range(nBayY + 1):
            structural_node = node_tags[(floor, ix, iy)]
            constrained_nodes.append(structural_node)

    # x-y 평면 diaphragm이므로, 수직방향은 global Z = 3
    ops.rigidDiaphragm(3, retained_node, *constrained_nodes)

    diaphragm_nodes[floor] = retained_node



# ------------------------------------------------------------
# 13-1. Additional nodal mass for irregular mass distribution
# ------------------------------------------------------------
# Irregular mass distribution is generated by adding extra nodal mass at selected corner nodes.
# Nodal Mass input:
# Node 1304 : 20 kN/g
# Node 2304 : 30 kN/g
# Node 3304 : 40 kN/g

additional_nodal_mass = {1304: 20.0, 2304: 30.0, 3304: 40.0}

additional_nodal_mass_info = {}

for nodeTag, add_mass in additional_nodal_mass.items():

    if nodeTag not in ops.getNodeTags():
        raise ValueError(f"Additional mass node {nodeTag} does not exist in the model.")

    # X,Y방향 질량만 추가한다.
    # Z방향 질량과 회전질량은 0으로 둔다.
    add_mass_vector = [add_mass, add_mass, 0.0, 0.0, 0.0, 0.0]

    ops.mass(nodeTag, *add_mass_vector)

    # 질량참여율 계산에 추가 nodal mass도 포함한다.
    if nodeTag not in mass_node_tags:
        mass_node_tags.append(nodeTag)

    node_mass_dict[nodeTag] = add_mass_vector

    coord = ops.nodeCoord(nodeTag)

    additional_nodal_mass_info[nodeTag] = {
        "mass": add_mass,
        "coord": coord
    }



# ------------------------------------------------------------
# 14. Constraint handler
# ------------------------------------------------------------
# rigidDiaphragm을 사용하므로 Transformation constraint handler를 사용한다.

ops.constraints("Transformation")


# ------------------------------------------------------------
# 15. Model summary
# ------------------------------------------------------------
print("============================================================")
print("3D Steel Frame Model Generation Completed")
print("============================================================")
print("Beam section H-400x200x8x13")
print(f"A_beam                    : {A_beam:.6e} m^2")
print(f"Iy_beam                   : {Iy_beam:.6e} m^4")
print(f"Iz_beam                   : {Iz_beam:.6e} m^4")
print(f"J_beam                    : {J_beam:.6e} m^4")
print("------------------------------------------------------------")
print("Column section H-350x350x12x19")
print(f"A_column                  : {A_column:.6e} m^2")
print(f"Iy_column                 : {Iy_column:.6e} m^4")
print(f"Iz_column                 : {Iz_column:.6e} m^4")
print(f"J_column                  : {J_column:.6e} m^4")
print("------------------------------------------------------------")
print(f"Floor translational mass Mp: {Mp:.6e} kN-sec^2/m")
print(f"Floor rotational mass Mr   : {Mr:.6e}")
print("------------------------------------------------------------")
print("Additional nodal mass for irregular model")

for nodeTag, info in additional_nodal_mass_info.items():
    coord = info["coord"]
    print(
        f"Node {nodeTag}: "
        f"{info['mass']:.6e} kN-sec^2/m, "
        f"coord = ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})"
    )

print("============================================================")



# ------------------------------------------------------------
# 16. Visualization functions
# ------------------------------------------------------------
# 현재 모델은 3D 모델이므로 nodeCoord에서 x, y, z 좌표를 모두 받도록 수정한다.

import matplotlib.pyplot as plt
import opsvis as opsv


def plot_model_like_liel_3d(title=None, node=True, element=True, image=True,
                            label_size=10, axis_label_size=13, tick_size=11,
                            margin=1.5):
    """
    - 3D 골조 형상 확인
    - node tag 확인
    - element tag 확인
    - diaphragm retained node 위치 확인
    """

    plt.rcParams["font.family"] = "Times New Roman"

    opsv.plot_model(
        node_labels=node,
        element_labels=element,
        fig_wi_he=(11, 7.5)
    )

    ax = plt.gca()

    if title is not None and title != "":
        plt.title(title, fontsize=16, fontname="Times New Roman")
    else:
        plt.title("")

    ax.set_xlabel("X (m)", fontsize=axis_label_size, fontname="Times New Roman")
    ax.set_ylabel("Y (m)", fontsize=axis_label_size, fontname="Times New Roman")

    # 3D axis인 경우 Z label 추가
    if hasattr(ax, "set_zlabel"):
        ax.set_zlabel("Z (m)", fontsize=axis_label_size, fontname="Times New Roman")

    # node / element label 크기 조정
    for text in ax.texts:
        text.set_fontsize(label_size)
        text.set_fontname("Times New Roman")
        text.set_clip_on(False)

    # 좌표 범위 자동 계산
    node_tags_all = ops.getNodeTags()

    x_list = []
    y_list = []
    z_list = []

    for nodeTag in node_tags_all:
        coord = ops.nodeCoord(nodeTag)

        x_list.append(coord[0])
        y_list.append(coord[1])

        if len(coord) >= 3:
            z_list.append(coord[2])

    ax.set_xlim(min(x_list) - margin, max(x_list) + margin)
    ax.set_ylim(min(y_list) - margin, max(y_list) + margin)

    if len(z_list) > 0 and hasattr(ax, "set_zlim"):
        ax.set_zlim(min(z_list) - margin, max(z_list) + margin)

    ax.tick_params(axis="both", labelsize=tick_size)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    if hasattr(ax, "get_zticklabels"):
        for label in ax.get_zticklabels():
            label.set_fontname("Times New Roman")

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "model_view.png", dpi=300, bbox_inches="tight")

    if image == True:
        plt.show()
    else:
        plt.close()


def plot_mode_shapes_like_liel_3d(nmode, period_list=None, image=True):
    """
    opsvis의 plot_mode_shape()를 사용
    """

    plt.rcParams["font.family"] = "Times New Roman"

    for i in range(nmode):
        opsv.plot_mode_shape(i + 1)

        if period_list is not None:
            plt.title(
                "Mode " + str(i + 1) + ", T = " + str(round(period_list[i], 3)) + " sec",
                fontsize=16,
                fontname="Times New Roman"
            )
        else:
            plt.title(
                "Mode " + str(i + 1),
                fontsize=16,
                fontname="Times New Roman"
            )

        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"mode_shape_{i + 1:02d}.png", dpi=300, bbox_inches="tight")

    if image == True:
        plt.show()
    else:
        plt.close()



# ------------------------------------------------------------
# 17. Plot model
# ------------------------------------------------------------
plot_model_like_liel_3d(
    title="3D Steel Frame Model with Rigid Diaphragm",
    node=True,
    element=True,
    image=True,
    label_size=9,
    axis_label_size=13,
    tick_size=11,
    margin=5.0
)


# ------------------------------------------------------------
# 18. Eigenvalue analysis function
# ------------------------------------------------------------
def run_eigen_analysis(num_modes):
    ops.wipeAnalysis()

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")

    lambd = ops.eigen("-fullGenLapack", num_modes)

    omega_list = []
    period_list = []

    for i in range(num_modes):
        lam = lambd[i]

        if lam <= 0:
            print("Mode", i + 1, "Lambda:", lam)
            print("Warning: eigenvalue가 0 이하입니다.")
            omega_list.append(None)
            period_list.append(None)
            continue

        omega = lam ** 0.5
        period = 2.0 * 3.141592653589793 / omega

        omega_list.append(omega)
        period_list.append(period)

        print("Mode", i + 1, "Lambda:", lam)
        print("Mode", i + 1, "Omega :", omega)
        print("Mode", i + 1, "Period:", period)

    return lambd, omega_list, period_list



# ------------------------------------------------------------
# 19. Modal participation mass calculation
# ------------------------------------------------------------
def calculate_modal_participation(num_modes, mass_node_tags, node_mass_dict):
    """
    각 모드의 방향별 질량참여율을 계산한다.

    TRAN-X, TRAN-Y:
    - 각 방향 병진 질량참여율 계산

    ROTN-Z:
    - DOF 6에 직접 입력된 Mrz만 사용하는 것이 아니라,
      기준 질량중심에서 떨어진 병진질량 Mx, My가 만드는
      등가 회전질량참여율을 계산한다.
    """

    # ------------------------------------------------------------
    # 1. Total translational mass
    # ------------------------------------------------------------
    total_mass_x = 0.0
    total_mass_y = 0.0
    total_mass_z = 0.0

    # 질량중심 계산용
    cm_mass_sum = 0.0
    xcm_num = 0.0
    ycm_num = 0.0

    for nodeTag in mass_node_tags:
        mass_vector = node_mass_dict[nodeTag]

        mx = mass_vector[0]
        my = mass_vector[1]
        mz = mass_vector[2]

        coord = ops.nodeCoord(nodeTag)
        x = coord[0]
        y = coord[1]

        total_mass_x += mx
        total_mass_y += my
        total_mass_z += mz

        # X, Y 방향 질량이 같은 경우가 일반적이므로 평균값을 질량중심 계산에 사용
        m_cm = 0.5 * (mx + my)

        if m_cm != 0.0:
            cm_mass_sum += m_cm
            xcm_num += m_cm * x
            ycm_num += m_cm * y

    Xcm_mass = xcm_num / cm_mass_sum
    Ycm_mass = ycm_num / cm_mass_sum

    # ------------------------------------------------------------
    # 2. Total rotational mass about Z-axis
    # ------------------------------------------------------------
    total_mass_rz = 0.0

    for nodeTag in mass_node_tags:
        mass_vector = node_mass_dict[nodeTag]

        mx = mass_vector[0]
        my = mass_vector[1]
        mrz = mass_vector[5]

        coord = ops.nodeCoord(nodeTag)
        x = coord[0]
        y = coord[1]

        dx = x - Xcm_mass
        dy = y - Ycm_mass

        # Izz = Σ(mx*dy^2 + my*dx^2 + mrz)
        total_mass_rz += mx * dy**2 + my * dx**2 + mrz

    print("============================================================")
    print("Total mass check")
    print("============================================================")
    print(f"TRAN-X   total mass = {total_mass_x:.6f}")
    print(f"TRAN-Y   total mass = {total_mass_y:.6f}")
    print(f"TRAN-Z   total mass = {total_mass_z:.6f}")
    print(f"ROTN-Z   total mass = {total_mass_rz:.6f}")
    print("------------------------------------------------------------")
    print(f"Mass center Xcm = {Xcm_mass:.6f} m")
    print(f"Mass center Ycm = {Ycm_mass:.6f} m")

    print("============================================================")
    print("Modal Participation Mass Ratio")
    print("============================================================")

    cumulative_ratio = {
        "TRAN-X": 0.0,
        "TRAN-Y": 0.0,
        "TRAN-Z": 0.0,
        "ROTN-Z": 0.0
    }

    cumulative_mass = {
        "TRAN-X": 0.0,
        "TRAN-Y": 0.0,
        "TRAN-Z": 0.0,
        "ROTN-Z": 0.0
    }

    participation_result = {}

    # ------------------------------------------------------------
    # 3. Modal participation calculation
    # ------------------------------------------------------------
    for mode in range(1, num_modes + 1):

        # generalized mass = phi^T M phi
        generalized_mass = 0.0

        for nodeTag in mass_node_tags:
            mass_vector = node_mass_dict[nodeTag]

            for dof in range(1, 7):
                m = mass_vector[dof - 1]

                if m == 0.0:
                    continue

                phi = ops.nodeEigenvector(nodeTag, mode, dof)
                generalized_mass += m * phi * phi

        participation_result[mode] = {}

        # ------------------------------------------------------------
        # TRAN-X participation
        # ------------------------------------------------------------
        Lx = 0.0

        for nodeTag in mass_node_tags:
            mx = node_mass_dict[nodeTag][0]

            if mx == 0.0:
                continue

            phi_x = ops.nodeEigenvector(nodeTag, mode, 1)
            Lx += mx * phi_x

        if total_mass_x != 0.0 and generalized_mass != 0.0:
            gamma_x = Lx / generalized_mass
            eff_mass_x = Lx * Lx / generalized_mass
            ratio_x = eff_mass_x / total_mass_x * 100.0
        else:
            gamma_x = 0.0
            eff_mass_x = 0.0
            ratio_x = 0.0

        # ------------------------------------------------------------
        # TRAN-Y participation
        # ------------------------------------------------------------
        Ly = 0.0

        for nodeTag in mass_node_tags:
            my = node_mass_dict[nodeTag][1]

            if my == 0.0:
                continue

            phi_y = ops.nodeEigenvector(nodeTag, mode, 2)
            Ly += my * phi_y

        if total_mass_y != 0.0 and generalized_mass != 0.0:
            gamma_y = Ly / generalized_mass
            eff_mass_y = Ly * Ly / generalized_mass
            ratio_y = eff_mass_y / total_mass_y * 100.0
        else:
            gamma_y = 0.0
            eff_mass_y = 0.0
            ratio_y = 0.0

        # ------------------------------------------------------------
        # TRAN-Z participation
        # ------------------------------------------------------------
        Lz = 0.0

        for nodeTag in mass_node_tags:
            mz = node_mass_dict[nodeTag][2]

            if mz == 0.0:
                continue

            phi_z = ops.nodeEigenvector(nodeTag, mode, 3)
            Lz += mz * phi_z

        if total_mass_z != 0.0 and generalized_mass != 0.0:
            gamma_z = Lz / generalized_mass
            eff_mass_z = Lz * Lz / generalized_mass
            ratio_z = eff_mass_z / total_mass_z * 100.0
        else:
            gamma_z = 0.0
            eff_mass_z = 0.0
            ratio_z = 0.0

        # ------------------------------------------------------------
        # ROTN-Z participation
        # ------------------------------------------------------------
        Lrz = 0.0

        for nodeTag in mass_node_tags:
            mass_vector = node_mass_dict[nodeTag]

            mx = mass_vector[0]
            my = mass_vector[1]
            mrz = mass_vector[5]

            coord = ops.nodeCoord(nodeTag)
            x = coord[0]
            y = coord[1]

            dx = x - Xcm_mass
            dy = y - Ycm_mass

            phi_x = ops.nodeEigenvector(nodeTag, mode, 1)
            phi_y = ops.nodeEigenvector(nodeTag, mode, 2)
            phi_rz = ops.nodeEigenvector(nodeTag, mode, 6)

            # unit Rz rotation influence vector:
            # ux = -dy, uy = dx
            Lrz += mx * phi_x * (-dy)
            Lrz += my * phi_y * ( dx)
            Lrz += mrz * phi_rz

        if total_mass_rz != 0.0 and generalized_mass != 0.0:
            gamma_rz = Lrz / generalized_mass
            eff_mass_rz = Lrz * Lrz / generalized_mass
            ratio_rz = eff_mass_rz / total_mass_rz * 100.0
        else:
            gamma_rz = 0.0
            eff_mass_rz = 0.0
            ratio_rz = 0.0

        # ------------------------------------------------------------
        # Save and print
        # ------------------------------------------------------------
        mode_result = {
            "TRAN-X": [gamma_x, eff_mass_x, ratio_x],
            "TRAN-Y": [gamma_y, eff_mass_y, ratio_y],
            "TRAN-Z": [gamma_z, eff_mass_z, ratio_z],
            "ROTN-Z": [gamma_rz, eff_mass_rz, ratio_rz]
        }

        print("------------------------------------------------------------")
        print(f"Mode {mode}")
        print(f"Generalized mass = {generalized_mass:.6e}")
        print("------------------------------------------------------------")
        print(f"{'Direction':>10s} {'Gamma':>14s} {'Eff.Mass':>14s} {'Ratio(%)':>12s} {'Cum.(%)':>12s}")

        for dir_name, values in mode_result.items():

            gamma = values[0]
            eff_mass = values[1]
            ratio = values[2]

            cumulative_ratio[dir_name] += ratio
            cumulative_mass[dir_name] += eff_mass

            participation_result[mode][dir_name] = {
                "gamma": gamma,
                "effective_mass": eff_mass,
                "mass_ratio": ratio,
                "cumulative_ratio": cumulative_ratio[dir_name],
                "cumulative_mass": cumulative_mass[dir_name]
            }

            print(
                f"{dir_name:>10s} "
                f"{gamma:14.6e} "
                f"{eff_mass:14.6f} "
                f"{ratio:12.4f} "
                f"{cumulative_ratio[dir_name]:12.4f}"
            )

    print("============================================================")
    print("Cumulative effective modal mass")
    print("============================================================")

    for dir_name in cumulative_mass.keys():
        print(
            f"{dir_name:8s} "
            f"effective mass sum = {cumulative_mass[dir_name]:.6f}, "
            f"ratio sum = {cumulative_ratio[dir_name]:.4f}%"
        )

    return participation_result



# ------------------------------------------------------------
# 20. Eigenvalue analysis, modal participation, and mode shape plot
# ------------------------------------------------------------
num_modes = 6

lambd, omega_list, period_list = run_eigen_analysis(num_modes)

participation_result = calculate_modal_participation(
    num_modes=num_modes,
    mass_node_tags=mass_node_tags,
    node_mass_dict=node_mass_dict
)

plot_mode_shapes_like_liel_3d(
    nmode=num_modes,
    period_list=period_list,
    image=True
)
