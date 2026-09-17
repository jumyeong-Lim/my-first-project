import math
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from matplotlib.animation import FuncAnimation, PillowWriter
# pyrefly: ignore [missing-import]
import openseespy.opensees as ops
# pyrefly: ignore [missing-import]
import opsvis as opsv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
ANIMATIONS_DIR = PROJECT_ROOT / "results" / "animations"
MATRICES_DIR = PROJECT_ROOT / "results" / "matrices"


def build_three_story_one_bay_frame():
    """Build a 2D one-bay, three-story elastic frame without rigid diaphragm."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Unit system: N, m, kg
    bay_width = 6.0
    story_height = 3.0
    num_stories = 3

    # Material and section properties for a simple reinforced-concrete-like frame.
    elastic_modulus = 25.0e9
    column_area = 0.40 * 0.40
    column_inertia = 0.40 * 0.40**3 / 12.0
    beam_area = 0.30 * 0.50
    beam_inertia = 0.30 * 0.50**3 / 12.0

    node_tags = {}
    for level in range(num_stories + 1):
        y = level * story_height
        left_node = level * 2 + 1
        right_node = level * 2 + 2

        ops.node(left_node, 0.0, y)
        ops.node(right_node, bay_width, y)
        node_tags[(level, "left")] = left_node
        node_tags[(level, "right")] = right_node

    # Fixed column bases.
    ops.fix(node_tags[(0, "left")], 1, 1, 1)
    ops.fix(node_tags[(0, "right")], 1, 1, 1)

    # Translational floor mass for dynamic properties.
    floor_mass = 25_000.0
    for level in range(1, num_stories + 1):
        ops.mass(node_tags[(level, "left")], floor_mass / 2.0, 0.0, 0.0)
        ops.mass(node_tags[(level, "right")], floor_mass / 2.0, 0.0, 0.0)

    ops.geomTransf("Linear", 1)

    element_tag = 1
    for level in range(num_stories):
        lower_left = node_tags[(level, "left")]
        upper_left = node_tags[(level + 1, "left")]
        lower_right = node_tags[(level, "right")]
        upper_right = node_tags[(level + 1, "right")]

        ops.element(
            "elasticBeamColumn",
            element_tag,
            lower_left,
            upper_left,
            column_area,
            elastic_modulus,
            column_inertia,
            1,
        )
        element_tag += 1

        ops.element(
            "elasticBeamColumn",
            element_tag,
            lower_right,
            upper_right,
            column_area,
            elastic_modulus,
            column_inertia,
            1,
        )
        element_tag += 1

    for level in range(1, num_stories + 1):
        ops.element(
            "elasticBeamColumn",
            element_tag,
            node_tags[(level, "left")],
            node_tags[(level, "right")],
            beam_area,
            elastic_modulus,
            beam_inertia,
            1,
        )
        element_tag += 1

    return node_tags


def run_gravity_analysis(node_tags):
    """Apply simple vertical floor loads and run a static analysis."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    floor_vertical_load = -120_000.0
    for level in range(1, 4):
        ops.load(node_tags[(level, "left")], 0.0, floor_vertical_load / 2.0, 0.0)
        ops.load(node_tags[(level, "right")], 0.0, floor_vertical_load / 2.0, 0.0)

    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 20)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    return ops.analyze(1)


def get_modal_periods(num_modes=3):
    eigenvalues = ops.eigen(num_modes)
    periods = []
    for eigenvalue in eigenvalues:
        omega = math.sqrt(eigenvalue)
        periods.append(2.0 * math.pi / omega)
    return periods


def get_mode_shapes(node_tags, num_modes=3, num_stories=3):
    mode_shapes = {}
    for mode in range(1, num_modes + 1):
        roof_x = ops.nodeEigenvector(node_tags[(num_stories, "left")], mode, 1)
        scale = roof_x if abs(roof_x) > 1.0e-12 else 1.0
        story_shapes = []

        for level in range(1, num_stories + 1):
            left_x = ops.nodeEigenvector(node_tags[(level, "left")], mode, 1) / scale
            right_x = ops.nodeEigenvector(node_tags[(level, "right")], mode, 1) / scale
            story_shapes.append(
                {
                    "story": level,
                    "left_x": left_x,
                    "right_x": right_x,
                    "average_x": (left_x + right_x) / 2.0,
                }
            )

        mode_shapes[mode] = story_shapes

    return mode_shapes


def save_matrix_csv(output_path, labels, matrix):
    with output_path.open("w", encoding="utf-8") as file:
        file.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            file.write(label + "," + ",".join(f"{value:.10e}" for value in row) + "\n")
    return output_path


def save_matrix_heatmap(output_path, labels, matrix, title):
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    max_abs = max(abs(value) for row in matrix for value in row) or 1.0
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)

    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def summarize_matrix(name, matrix):
    size = len(matrix)
    total_entries = size * size
    nonzero_entries = sum(1 for row in matrix for value in row if abs(value) > 1.0e-9)
    sparsity = 100.0 * (1.0 - nonzero_entries / total_entries)
    diagonal = [matrix[index][index] for index in range(size)]
    is_symmetric = all(
        abs(matrix[row][col] - matrix[col][row]) <= 1.0e-6
        for row in range(size)
        for col in range(size)
    )

    print(f"{name}:")
    print(f"  Size: {size} x {size}")
    print(f"  Nonzero entries: {nonzero_entries} / {total_entries}")
    print(f"  Sparsity: {sparsity:.1f}%")
    print(f"  Symmetric: {is_symmetric}")
    print(f"  Diagonal min/max: {min(diagonal):.3e} / {max(diagonal):.3e}")


def export_mass_and_stiffness_matrices():
    mass_labels = [
        "Story1_LX",
        "Story1_RX",
        "Story2_LX",
        "Story2_RX",
        "Story3_LX",
        "Story3_RX",
    ]
    nodal_mass = 25_000.0 / 2.0
    mass_matrix = [
        [nodal_mass if row == col else 0.0 for col in range(6)]
        for row in range(6)
    ]

    stiffness_matrix = get_active_stiffness_matrix()
    stiffness_labels = [f"eq{i + 1}" for i in range(len(stiffness_matrix))]

    mass_csv = save_matrix_csv(
        MATRICES_DIR / "without_rigid_diaphragm_mass_matrix.csv",
        mass_labels,
        mass_matrix,
    )
    mass_heatmap = save_matrix_heatmap(
        FIGURES_DIR / "without_rigid_diaphragm_mass_matrix_heatmap.png",
        mass_labels,
        mass_matrix,
        "Without rigid diaphragm mass matrix",
    )
    stiffness_csv = save_matrix_csv(
        MATRICES_DIR / "without_rigid_diaphragm_stiffness_matrix.csv",
        stiffness_labels,
        stiffness_matrix,
    )
    stiffness_heatmap = save_matrix_heatmap(
        FIGURES_DIR / "without_rigid_diaphragm_stiffness_matrix_heatmap.png",
        stiffness_labels,
        stiffness_matrix,
        "Without rigid diaphragm active stiffness matrix",
    )

    print("\n=== Without rigid diaphragm matrix summary ===")
    print("Independent lateral DOFs: left and right x DOFs at each floor.")
    print("Each floor mass is split equally between the left and right nodes.")
    summarize_matrix("Mass matrix [kg]", mass_matrix)
    summarize_matrix("Active stiffness matrix", stiffness_matrix)
    print(f"  Mass CSV: {mass_csv}")
    print(f"  Mass heatmap: {mass_heatmap}")
    print(f"  Stiffness CSV: {stiffness_csv}")
    print(f"  Stiffness heatmap: {stiffness_heatmap}")


def get_active_stiffness_matrix():
    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.algorithm("Linear")
    ops.integrator("GimmeMCK", 0.0, 0.0, 1.0)
    ops.analysis("Transient")
    ops.analyze(1, 0.0)

    values = ops.printA("-ret")
    size = int(math.sqrt(len(values)))
    return [
        values[row * size : (row + 1) * size]
        for row in range(size)
    ]


def prepare_output_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
    MATRICES_DIR.mkdir(parents=True, exist_ok=True)


def save_model_plot(output_path=FIGURES_DIR / "frame_without_rigid_diaphragm.png"):
    plt.figure(figsize=(6.0, 7.0))
    opsv.plot_model(node_labels=1, element_labels=1)
    plt.title("2D 1-bay 3-story frame without rigid diaphragm")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_mode_shape_plots(num_modes=3):
    paths = []
    bay_width = 6.0
    story_height = 3.0
    deformation_scale = 1.2

    for mode in range(1, num_modes + 1):
        output_path = FIGURES_DIR / f"without_rigid_diaphragm_mode_{mode}.png"
        roof_x = ops.nodeEigenvector(7, mode, 1)
        scale = roof_x if abs(roof_x) > 1.0e-12 else 1.0
        left_by_story = {
            story: ops.nodeEigenvector(story * 2 + 1, mode, 1) / scale
            for story in range(1, 4)
        }
        right_by_story = {
            story: ops.nodeEigenvector(story * 2 + 2, mode, 1) / scale
            for story in range(1, 4)
        }
        left_shift = {0: 0.0}
        right_shift = {0: 0.0}
        for story in range(1, 4):
            left_shift[story] = deformation_scale * left_by_story[story]
            right_shift[story] = deformation_scale * right_by_story[story]

        left_nodes = [(left_shift[level], level * story_height) for level in range(4)]
        right_nodes = [(bay_width + right_shift[level], level * story_height) for level in range(4)]

        fig, ax = plt.subplots(figsize=(6.0, 7.0))
        draw_frame_shape(ax, [(0.0, level * story_height) for level in range(4)], [(bay_width, level * story_height) for level in range(4)], "0.65", "--", "undeformed")
        draw_frame_shape(ax, left_nodes, right_nodes, "blue", "-", "mode shape")
        ax.set_title(f"Without rigid diaphragm mode {mode} shape")
        ax.text(
            0.02,
            0.96,
            "\n".join(
                [
                    f"Story {story}: L={left_by_story[story]: .3f}, R={right_by_story[story]: .3f}"
                    for story in range(1, 4)
                ]
            ),
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        ax.set_xlim(-3.0, 9.0)
        ax.set_ylim(-0.5, 9.8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        paths.append(output_path)
    return paths


def draw_frame_shape(ax, left_nodes, right_nodes, color, linestyle, label):
    ax.plot(
        [point[0] for point in left_nodes],
        [point[1] for point in left_nodes],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
    )
    ax.plot(
        [point[0] for point in right_nodes],
        [point[1] for point in right_nodes],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
    )
    for level in range(1, 4):
        ax.plot(
            [left_nodes[level][0], right_nodes[level][0]],
            [left_nodes[level][1], right_nodes[level][1]],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )


def save_mode_shape_animations(mode_shapes):
    paths = []
    bay_width = 6.0
    story_height = 3.0
    frame_count = 24
    deformation_scale = 1.2

    for mode, story_shapes in mode_shapes.items():
        output_path = ANIMATIONS_DIR / f"without_rigid_diaphragm_mode_{mode}.gif"
        left_by_story = {shape["story"]: shape["left_x"] for shape in story_shapes}
        right_by_story = {shape["story"]: shape["right_x"] for shape in story_shapes}
        fig, ax = plt.subplots(figsize=(6.0, 7.0))

        def draw_frame(frame):
            ax.clear()
            factor = math.sin(2.0 * math.pi * frame / frame_count)
            left_shift = {0: 0.0}
            right_shift = {0: 0.0}
            for story in range(1, 4):
                left_shift[story] = deformation_scale * factor * left_by_story[story]
                right_shift[story] = deformation_scale * factor * right_by_story[story]

            left_nodes = [(left_shift[level], level * story_height) for level in range(4)]
            right_nodes = [(bay_width + right_shift[level], level * story_height) for level in range(4)]

            ax.plot([0.0, 0.0], [0.0, 9.0], color="0.75", linestyle="--", linewidth=1.0)
            ax.plot([bay_width, bay_width], [0.0, 9.0], color="0.75", linestyle="--", linewidth=1.0)
            for level in range(1, 4):
                y = level * story_height
                ax.plot([0.0, bay_width], [y, y], color="0.75", linestyle="--", linewidth=1.0)

            ax.plot([p[0] for p in left_nodes], [p[1] for p in left_nodes], color="blue", linewidth=2.0)
            ax.plot([p[0] for p in right_nodes], [p[1] for p in right_nodes], color="blue", linewidth=2.0)
            for level in range(1, 4):
                ax.plot(
                    [left_nodes[level][0], right_nodes[level][0]],
                    [left_nodes[level][1], right_nodes[level][1]],
                    color="blue",
                    linewidth=2.0,
                )

            ax.set_title(f"Without rigid diaphragm mode {mode}")
            ax.set_xlim(-2.0, 8.0)
            ax.set_ylim(-0.5, 9.8)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.3)

        animation = FuncAnimation(fig, draw_frame, frames=frame_count, interval=80)
        animation.save(output_path, writer=PillowWriter(fps=12))
        plt.close(fig)
        paths.append(output_path)
    return paths


if __name__ == "__main__":
    prepare_output_dirs()
    nodes = build_three_story_one_bay_frame()
    plot_path = save_model_plot()
    analysis_result = run_gravity_analysis(nodes)
    periods = get_modal_periods()
    mode_shapes = get_mode_shapes(nodes)
    export_mass_and_stiffness_matrices()
    mode_plot_paths = save_mode_shape_plots()
    mode_animation_paths = save_mode_shape_animations(mode_shapes)

    print("2D 1-bay 3-story frame model without rigid diaphragm created.")
    print(f"Model plot saved to: {plot_path}")
    print("Mode shape plots saved to:", [str(path) for path in mode_plot_paths])
    print("Mode shape animations saved to:", [str(path) for path in mode_animation_paths])
    print(f"Gravity analysis result code: {analysis_result}")
    print("Modal periods [sec]:", [round(period, 4) for period in periods])
    print("Mode shapes, roof left x normalized to 1.0:")
    for mode, story_shapes in mode_shapes.items():
        print(f"  Mode {mode}:")
        for shape in story_shapes:
            print(
                "    "
                f"Story {shape['story']}: "
                f"left_x={shape['left_x']:.4f}, "
                f"right_x={shape['right_x']:.4f}, "
                f"avg_x={shape['average_x']:.4f}"
            )
    print("Roof left node displacement [m]:", [round(value, 8) for value in ops.nodeDisp(nodes[(3, "left")])])
    print("Roof right node displacement [m]:", [round(value, 8) for value in ops.nodeDisp(nodes[(3, "right")])])
