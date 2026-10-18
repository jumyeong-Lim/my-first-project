"""STEP11: STEP10 결과의 메시·가정 단면특성·PDelta 강건성 검증.

STEP10의 36조합을 기준으로 삼는다.
추가 해석은 U/X99/Y99/C99 네 질량배치만 대상으로 한다. 단면 배율은
실제 형강 규격이 아니라 모든 보·기둥 A/Iy/Iz/J의 가상 공통 배율이다.
질량, 중력하중, 바닥판 강성 및 접합은 모든 민감도 해석에서 유지한다.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from verification.step10_diaphragm_mass_interaction import (
    OUT as STEP10_OUT, P, XY, DIAPHRAGMS, MESH, PROJECT_ROOT,
    mass_cases, modal, static, scalar_results,
)

OUT = PROJECT_ROOT / "results/verification/step11_robustness_conclusion"
ANCHORS = ("U", "X99", "Y99", "C99")
METRICS = (
    "T1_s", "X_roof_max_mm", "Y_roof_max_mm",
    "X_roof_joint_max_mm", "Y_roof_joint_max_mm",
    "X_roof_distortion_mm", "Y_roof_distortion_mm",
    "first_mode_roof_rotation_share", "first_mode_roof_distortion_ratio",
)


def metric_bundle(case):
    """STEP10과 같은 응답지표를 추출하고 메시 독립적인 골조 절점 최대값을 추가한다."""
    result = scalar_results(case)
    for direction in ("X", "Y"):
        displacement = np.asarray(case[f"static_{direction}"][-1]["joint_displacements_mm"])
        result[f"{direction}_roof_joint_max_mm"] = float(
            np.max(np.abs(displacement[:, "XY".index(direction)])))
    result["first_mode_roof_distortion_ratio"] = float(
        case["modal"]["character"][0]["floor_modal_distortion_ratio"][-1])
    return result


def first_modes_match(reference, comparison, mass_weights):
    """질량가중 MAC으로 같은 물리적 모드를 찾아 단순 차수 비교의 함정을 피한다."""
    count = min(9, len(reference["modal"]["vectors"]), len(comparison["modal"]["vectors"]))
    a = np.asarray(reference["modal"]["vectors"][:count])
    b = np.asarray(comparison["modal"]["vectors"][:count])
    md = np.tile(np.repeat(P.floor_mass*np.asarray(mass_weights), 2), P.stories)
    score = ((a*md) @ b.T)**2
    row, column = linear_sum_assignment(-score)
    match = column[np.argsort(row)]
    return {
        "matched_orders_first_9": (match+1).tolist(),
        "MAC_first_9": score[np.arange(count), match].tolist(),
        "minimum_MAC_first_3": float(np.min(score[np.arange(min(3, count)), match[:3]])),
    }


def evaluate(diaphragm, mass_case, mesh=MESH, member_scale=1.0,
             transformation="PDelta"):
    """중력 선행해석을 매번 새로 수행하여 변형률 이력과 하중 조합을 섞지 않는다."""
    weights = mass_cases()[mass_case]
    common = {"mesh": mesh, "member_scale": member_scale,
              "transformation": transformation}
    result = {
        "diaphragm": diaphragm, "mass_case": mass_case,
        "modal": modal(diaphragm, weights, **common),
        "static_X": static(diaphragm, weights, 0, **common),
        "static_Y": static(diaphragm, weights, 1, **common),
    }
    return result


def scenarios():
    """각 축을 독립적으로 교란하여 원인별 민감도를 분리한다."""
    yield "baseline", "reference", MESH, 1.0, "PDelta", ("R", "S", "F")
    for mesh in (4, 16):
        yield "mesh", f"{mesh}x{mesh}", mesh, 1.0, "PDelta", ("S", "F")
    for factor in (0.75, 1.25):
        yield "section", f"{factor:.2f}", MESH, factor, "PDelta", ("R", "S", "F")
    yield "geometry", "Linear", MESH, 1.0, "Linear", ("R", "S", "F")


def response_map(records, scenario, level):
    """모든 시나리오에서 같은 질량배치의 R/S/F 응답을 반환한다."""
    result = {}
    for mass_case in ANCHORS:
        for diaphragm in DIAPHRAGMS:
            key = (scenario, level, diaphragm, mass_case)
            if key not in records:
                # 강체 다이어프램에는 셸 메시가 없으므로 메시 비교에서만 재사용한다.
                if scenario != "mesh" or diaphragm != "R":
                    raise KeyError(f"누락된 STEP11 조건: {key}")
                key = ("baseline", "reference", diaphragm, mass_case)
            result[(diaphragm, mass_case)] = records[key]["metrics"]
    return result


def contrasts(records):
    """상호작용 I=Q(F,E)-Q(R,E)-Q(F,U)+Q(R,U)를 조건별로 재계산한다."""
    output = []
    for family, level, *_ in scenarios():
        current = response_map(records, family, level)
        for diaphragm in ("S", "F"):
            for mass_case in ("X99", "Y99", "C99"):
                for metric in METRICS:
                    r_u = current[("R", "U")][metric]
                    r_e = current[("R", mass_case)][metric]
                    d_u = current[(diaphragm, "U")][metric]
                    d_e = current[(diaphragm, mass_case)][metric]
                    output.append({
                        "family": family, "level": level, "diaphragm": diaphragm,
                        "mass_case": mass_case, "metric": metric,
                        "R_U": r_u, "R_E": r_e, "D_U": d_u, "D_E": d_e,
                        "mass_effect_R": r_e-r_u,
                        "diaphragm_effect_U": d_u-r_u,
                        "diaphragm_effect_E": d_e-r_e,
                        "interaction": d_e-r_e-d_u+r_u,
                        "same_mass_difference_pct": 100*(d_e-r_e)/r_e if abs(r_e)>1e-12 else None,
                    })
    return output


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(baseline_path):
    """모델·후처리 코드와 기준 결과가 바뀌면 오래된 계산 캐시를 사용하지 않는다."""
    digest = hashlib.sha256()
    for filename in (
        "step07_3d_reference_frame.py", "step08_diaphragm_inplane_sensitivity.py",
        "step10_diaphragm_mass_interaction.py", "step11_robustness_conclusion.py",
    ):
        digest.update((PROJECT_ROOT/"verification"/filename).read_bytes())
    digest.update(baseline_path.read_bytes())
    return digest.hexdigest()


def diagnostics(records, rows, interactions):
    """결론에 쓰는 수치를 자동 집계하여 표·보고서의 재현성을 높인다."""
    by_metric = {}
    for family, level in (("mesh", "16x16"), ("section", "0.75"),
                          ("section", "1.25"), ("geometry", "Linear")):
        for metric in ("T1_s", "X_roof_joint_max_mm", "Y_roof_joint_max_mm"):
            subset = [r for r in rows if r["family"] == family and
                      r["level"] == level and r["metric"] == metric]
            worst = max(subset, key=lambda r: abs(r["change_pct"]))
            by_metric[f"{family}_{level}_{metric}"] = {
                "max_abs_change_pct": abs(worst["change_pct"]),
                "signed_change_pct": worst["change_pct"],
                "case": f"{worst['diaphragm']}_{worst['mass_case']}",
            }
    c99 = {}
    for metric in ("T1_s", "X_roof_joint_max_mm", "Y_roof_joint_max_mm"):
        selected = [r for r in interactions if r["diaphragm"] == "F" and
                    r["mass_case"] == "C99" and r["metric"] == metric]
        c99[metric] = {
            "minimum_interaction": min(r["interaction"] for r in selected),
            "maximum_interaction": max(r["interaction"] for r in selected),
            "all_positive": all(r["interaction"] > 0 for r in selected),
        }
    low_mac = sorted((
        {"case": f"{r['diaphragm']}_{r['mass_case']}", "family": r["family"],
         "level": r["level"], "minimum_MAC_first_3": r["MAC_first_3_min"]}
        for r in rows if r["metric"] == "T1_s" and r["MAC_first_3_min"] < .8
    ), key=lambda r: r["minimum_MAC_first_3"])
    return {"worst_relative_change": by_metric, "F_C99_interaction": c99,
            "first_3_mode_low_MAC": low_mac}


def crossing_check(baseline):
    """가까운 3·4차 모드의 개별 MAC 저하가 모드 교차인지 부분공간으로 점검한다."""
    weights = mass_cases()["X99"]
    finite = modal("F", weights, member_scale=.75)
    reference = baseline["F_X99"]["modal"]
    md = np.tile(np.repeat(P.floor_mass*weights, 2), P.stories)
    a = np.asarray(reference["vectors"][2:4])
    b = np.asarray(finite["vectors"][2:4])
    overlap = (a*md) @ b.T
    singular = np.linalg.svd(overlap, compute_uv=False)
    return {
        "case": "F_X99", "section_factor": .75, "mode_group": [3, 4],
        "reference_periods_s": reference["periods_s"][2:4],
        "perturbed_periods_s": finite["periods_s"][2:4],
        "individual_MAC_matrix": (overlap**2).tolist(),
        "subspace_singular_values": singular.tolist(),
    }


def make_plots(records, interactions):
    """시나리오별 메시 수렴과 상호작용 유지 여부를 같은 축에서 보여준다."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, metric, title in zip(
        axes,
        ("T1_s", "X_roof_joint_max_mm", "Y_roof_joint_max_mm"),
        ("First period (s)", "Roof joint max |Ux| (mm)", "Roof joint max |Uy| (mm)"),
    ):
        for diaphragm in ("S", "F"):
            values = []
            for mesh in (4, 8, 16):
                key = ("baseline", "reference", diaphragm, "C99") if mesh == 8 else (
                    "mesh", f"{mesh}x{mesh}", diaphragm, "C99")
                values.append(records[key]["metrics"][metric])
            ax.plot((4, 8, 16), values, "o-", label=diaphragm)
        ax.set(xlabel="Shell subdivisions per bay", ylabel=title, xticks=(4, 8, 16))
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("C99 mesh sensitivity; frame-joint peak avoids changing sample locations")
    fig.tight_layout()
    fig.savefig(OUT/"step11_mesh_convergence.png", dpi=170)
    plt.close(fig)

    selected = [row for row in interactions if row["diaphragm"] == "F" and
                row["mass_case"] == "C99" and row["metric"] in
                ("T1_s", "X_roof_joint_max_mm", "Y_roof_joint_max_mm")]
    labels = [f"{family}\n{level}" for family, level, *_ in scenarios()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric, title in zip(
        axes, ("T1_s", "X_roof_joint_max_mm", "Y_roof_joint_max_mm"),
        ("T1 interaction (s)", "Roof joint X interaction (mm)", "Roof joint Y interaction (mm)"),
    ):
        values = [next(r["interaction"] for r in selected if
                       r["family"] == family and r["level"] == level and r["metric"] == metric)
                  for family, level, *_ in scenarios()]
        ax.bar(range(len(labels)), values, color="#3274a1")
        ax.axhline(0, color=".2", lw=.8)
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel=title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("C99: diaphragm–mass interaction across independent perturbations")
    fig.tight_layout()
    fig.savefig(OUT/"step11_interaction_robustness.png", dpi=170)
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_path = STEP10_OUT/"step10_results.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"STEP10 기준 결과가 없습니다: {baseline_path}")
    current_fingerprint = fingerprint(baseline_path)
    previous_path = OUT/"step11_results.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
    use_cache = previous is None or previous["metadata"].get("input_fingerprint") in (
        None, current_fingerprint)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["results"]
    records = {}
    weights_by_case = mass_cases()
    # 재실행 시 완료한 조건은 읽고 미완료 조건만 계산한다.
    for family, level, mesh, member_scale, transformation, diaphragms in scenarios():
        for mass_case in ANCHORS:
            for diaphragm in diaphragms:
                key = (family, level, diaphragm, mass_case)
                info = {"family": family, "level": level, "mesh": mesh,
                        "member_scale": member_scale, "transformation": transformation,
                        "diaphragm": diaphragm, "mass_case": mass_case}
                if family == "baseline":
                    case = baseline[f"{diaphragm}_{mass_case}"]
                    records[key] = {"config": info, "metrics": metric_bundle(case),
                                    "modal": case["modal"]}
                    continue
                path = OUT/f"case_{family}_{level}_{diaphragm}_{mass_case}.json"
                if use_cache and path.exists():
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    if cached["config"] == info:
                        records[key] = cached
                        continue
                print(f"STEP11 {family} {level} {diaphragm}_{mass_case}", flush=True)
                case = evaluate(diaphragm, mass_case, mesh, member_scale, transformation)
                match = first_modes_match(baseline[f"{diaphragm}_{mass_case}"],
                                          case, weights_by_case[mass_case])
                record = {"config": info, "metrics": metric_bundle(case),
                          "mode_match_to_baseline": match,
                          "cumulative_participation": case["modal"]["cumulative_participation"],
                          "mass_orthogonality_error": case["modal"]["orthogonality_error"]}
                path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                records[key] = record
    # 오래된 STEP10 파일과 현행 코드가 일치하는지 기준 두 조건을 재현한다.
    for diaphragm, mass_case in (("R", "U"), ("F", "C99")):
        case = evaluate(diaphragm, mass_case)
        expected = records[("baseline", "reference", diaphragm, mass_case)]["metrics"]
        actual = metric_bundle(case)
        for metric in METRICS:
            np.testing.assert_allclose(actual[metric], expected[metric], rtol=1e-7,
                                       atol=1e-9, err_msg=f"STEP10 기준 재현 실패: {metric}")
    rows = []
    for key, record in records.items():
        if key[0] == "baseline":
            continue
        ref = records[("baseline", "reference", key[2], key[3])]["metrics"]
        for metric in METRICS:
            value = record["metrics"][metric]
            base = ref[metric]
            rows.append({**record["config"], "metric": metric, "value": value,
                         "baseline_value": base, "difference": value-base,
                         "change_pct": 100*(value-base)/base if abs(base)>1e-12 else None,
                         "MAC_first_3_min": record["mode_match_to_baseline"]["minimum_MAC_first_3"]})
    interactions = contrasts(records)
    summary = diagnostics(records, rows, interactions)
    summary["mode_crossing_check"] = crossing_check(baseline)
    write_csv(OUT/"step11_sensitivity.csv", rows)
    write_csv(OUT/"step11_interactions.csv", interactions)
    (OUT/"step11_diagnostics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(records, interactions)
    metadata = {
        "reference": str(baseline_path), "mass_cases": ANCHORS,
        "mesh_per_bay": [4, 8, 16], "baseline_mesh": MESH,
        "section_property_factors": [0.75, 1.0, 1.25],
        "gravity_preload": True,
        "column_transforms": ["Linear", "PDelta"],
        "constant": ["geometry", "total_floor_mass", "gravity_load", "slab_E",
                     "slab_thickness", "slab_connection", "beam_linear_transform"],
        "section_factor_scope": "all beam and column A,Iy,Iz,J; not a catalog shape",
        "static_load": "each nodal horizontal mass × 1 m/s²",
        "input_fingerprint": current_fingerprint,
    }
    (OUT/"step11_results.json").write_text(json.dumps(
        {"metadata": metadata, "records": {"/".join(k): v for k, v in records.items()
             if k[0] != "baseline"}, "interactions": interactions},
        ensure_ascii=False), encoding="utf-8")
    print(f"STEP11 complete: {len(records)-12} sensitivity cases", flush=True)
    return rows, interactions


if __name__ == "__main__":
    run()
