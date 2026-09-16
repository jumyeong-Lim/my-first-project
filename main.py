import math
# pyrefly: ignore [missing-import]
import openseespy.opensees as ops


def build_three_story_one_bay_frame():
    """Build a 2D one-bay, three-story elastic frame model."""
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


if __name__ == "__main__":
    nodes = build_three_story_one_bay_frame()
    analysis_result = run_gravity_analysis(nodes)
    periods = get_modal_periods()

    print("2D 1-bay 3-story frame model created.")
    print(f"Gravity analysis result code: {analysis_result}")
    print("Modal periods [sec]:", [round(period, 4) for period in periods])
    print("Roof left node displacement [m]:", [round(value, 8) for value in ops.nodeDisp(nodes[(3, "left")])])
    print("Roof right node displacement [m]:", [round(value, 8) for value in ops.nodeDisp(nodes[(3, "right")])])
