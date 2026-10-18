"""STEP12: 기존 파일을 보존하며 셸의 구성식·단위시험·본모델 재현성을 검사한다.

시험 치수와 하중은 해석해가 있는 수치 검증용이다. 실제 바닥판 물성이 아니다.
실행: python -m verification.step12_shell_audit
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verification import step10_diaphragm_mass_interaction as original

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/verification/step12_shell_audit"
E_VALUES = {"S": 2.0e7, "F": 2.0e4}
NU, THICKNESS = .20, .12


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preservation_manifest():
    """기존 STEP01~11 파일의 바이트를 비교한다. 신규 STEP12/13은 제외한다."""
    files = []
    for folder in (ROOT / "verification", ROOT / "tests", ROOT / "results/verification"):
        for path in folder.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if any(x in relative.lower() for x in ("step12", "step13")):
                continue
            files.append(path)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def configure():
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-12, 30)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def rectangle(e, nx=12, ny=2, length=6., width=1., modifier=0., rho=0.):
    """XY 평면의 반시계방향 사각요소. 로컬 법선은 +Z다."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.section("ElasticMembranePlateSection", 1, e, NU, THICKNESS, rho, modifier)
    nodes, coords = {}, {}
    for j in range(ny+1):
        for i in range(nx+1):
            tag = 1+j*(nx+1)+i
            nodes[i, j] = tag
            coords[tag] = np.array([length*i/nx, width*j/ny, 0.])
            ops.node(tag, *coords[tag])
    for j in range(ny):
        for i in range(nx):
            ops.element("ShellMITC4", 1+j*nx+i, nodes[i, j], nodes[i+1, j],
                        nodes[i+1, j+1], nodes[i, j+1], 1)
    return nodes, coords


def load_pattern():
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)


def edge_weights(n):
    weights = np.ones(n+1)/n
    weights[[0, -1]] *= .5
    return weights


def solve():
    configure()
    code = ops.analyze(1)
    if code:
        raise RuntimeError(f"독립 셸 시험 정적해석 실패: {code}")
    ops.reactions()


def tension(e, modifier=0., sign=1.):
    nx, ny, length, width, force = 12, 2, 6., 1., sign*.001
    nodes, coords = rectangle(e, nx, ny, length, width, modifier)
    for (i, j), tag in nodes.items():
        # 좌변 Ux와 좌하단 Uy만 고정하여 포아송 수축을 허용한다.
        ops.fix(tag, int(i == 0), int(i == 0 and j == 0), 1, 1, 1, 0)
    load_pattern()
    for j, weight in enumerate(edge_weights(ny)):
        ops.load(nodes[nx, j], force*weight, 0, 0, 0, 0, 0)
    solve()
    measured = sum(w*ops.nodeDisp(nodes[nx, j], 1) for j, w in enumerate(edge_weights(ny)))
    theory = force*length/(e*width*THICKNESS)
    balance = sum(ops.nodeReaction(nodes[0, j], 1) for j in range(ny+1))+force
    return {"test": "A_tension" if sign > 0 else "A_compression", "nx": nx, "ny": ny,
            "E": e, "Ep_modifier": modifier, "load_kN": force,
            "theory_m": theory, "FE_m": measured, "error_pct": 100*(measured/theory-1),
            "equilibrium_error_kN": balance}


def shear(e, modifier=0.):
    nx, ny, length, width, gamma = 12, 2, 6., 1., 1e-5
    nodes, coords = rectangle(e, nx, ny, length, width, modifier)
    for tag in coords:
        ops.fix(tag, 0, 0, 1, 1, 1, 0)
    load_pattern()
    for (i, j), tag in nodes.items():
        if i in (0, nx) or j in (0, ny):
            # 둘레에 정확한 일정 전단변형장을 입력하고 드릴링 회전은 자유롭게 둔다.
            ops.sp(tag, 1, gamma*coords[tag][1])
            ops.sp(tag, 2, 0.)
    solve()
    measured = sum(ops.nodeReaction(nodes[i, ny], 1) for i in range(nx+1))
    theory = e/(2*(1+NU))*THICKNESS*length*gamma
    return {"test": "B_shear", "nx": nx, "ny": ny, "E": e, "Ep_modifier": modifier,
            "gamma": gamma, "theory_kN": theory, "FE_kN": measured,
            "error_pct": 100*(measured/theory-1)}


def cantilever(e, n, out_of_plane=False, modifier=0.):
    nx, ny, length, width, force = 6*n, n, 6., 1., .001
    nodes, coords = rectangle(e, nx, ny, length, width, modifier)
    for (i, j), tag in nodes.items():
        if out_of_plane:
            # 전 폭 Rx=0으로 원통형 판 휨을 만들고 Uy/Ux/Rz를 제거한다.
            # 이 경계에서는 EI=Db이며, 자유 측변의 보 휨 EI와 구분한다.
            ops.fix(tag, 1, 1, int(i == 0), 1, int(i == 0), 1)
        else:
            ops.fix(tag, int(i == 0), int(i == 0), 1, 1, 1, int(i == 0))
    load_pattern()
    dof = 3 if out_of_plane else 2
    for j, weight in enumerate(edge_weights(ny)):
        load = [0.]*6
        load[dof-1] = force*weight
        ops.load(nodes[nx, j], *load)
    solve()
    measured = sum(w*ops.nodeDisp(nodes[nx, j], dof) for j, w in enumerate(edge_weights(ny)))
    if out_of_plane:
        bending = modifier*e*THICKNESS**3/(12*(1-NU**2))*width
        shear_rigidity = (5/6)*modifier*e/(2*(1+NU))*THICKNESS*width
    else:
        bending = e*THICKNESS*width**3/12
        shear_rigidity = (5/6)*e/(2*(1+NU))*THICKNESS*width
    theory = force*(length**3/(3*bending)+length/shear_rigidity)
    balance = sum(ops.nodeReaction(nodes[0, j], dof) for j in range(ny+1))+force
    return {"test": "D_plate_bending" if out_of_plane else "C_inplane_bending",
            "nx": nx, "ny": ny, "E": e, "Ep_modifier": modifier,
            "load_kN": force, "theory_m": theory, "FE_m": measured,
            "error_pct": 100*(measured/theory-1), "equilibrium_error_kN": balance}


def zero_plate_check(e):
    nodes, coords = rectangle(e, 6, 1, modifier=0.)
    for tag in coords:
        # 0이 아닌 면외 변위를 강제하고 반력 0을 확인한다. 드릴링 자유도는 남긴다.
        ops.fix(tag, 1, 1, 0, 1, 1, 0)
    load_pattern()
    for tag, xyz in coords.items():
        ops.sp(tag, 3, .001*(xyz[0]/6)**2)
    solve()
    maximum = max(abs(ops.nodeReaction(tag, d)) for tag in coords for d in (3, 4, 5))
    return {"E": e, "Ep_modifier": 0., "max_out_of_plane_reaction": maximum,
            "force_control_solution": "면외 강성이 0이므로 단독 판의 하중제어 휨해는 존재하지 않음"}


def drilling_check(e, modifier):
    """병진을 고정한 한 요소에 Rz 모멘트를 가해 수치 안정화 강성을 측정한다."""
    nodes, coords = rectangle(e,1,1,modifier=modifier)
    for tag in coords:
        ops.fix(tag,1,1,1,1,1,0)
    load_pattern()
    target=nodes[1,1]
    ops.load(target,0,0,0,0,0,.001)
    solve()
    rotation=ops.nodeDisp(target,6)
    return {"E":e,"Ep_modifier":modifier,"applied_moment_kNm":.001,
            "rotation_rad":rotation,"drilling_effective_stiffness_kNm":.001/rotation,
            "interpretation":"드릴링 안정화 응답이며 물리적 슬래브 면외 휨강성이 아님"}


def mass_check(rho):
    nodes, coords = rectangle(2e7, 1, 1, length=2., width=3., rho=rho)
    # 질량행렬 추출 시 특이행렬을 피하려고 모든 DOF에 단위 보조질량을 준 뒤 뺀다.
    # 본모델 질량에는 이 보조질량 및 단위 밀도를 사용하지 않는다.
    for tag in coords:
        ops.mass(tag, *([1.]*6))
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.algorithm("Linear")
    ops.integrator("GimmeMCK", 1., 0., 0.)
    ops.analysis("Transient")
    if ops.analyze(1, 0.):
        raise RuntimeError("질량행렬 추출 실패")
    matrix = np.array(ops.printA("-ret")).reshape(24, 24)
    ordering = np.array([d for tag in sorted(coords) for d in ops.nodeDOFs(tag)])
    matrix = matrix[np.ix_(ordering, ordering)]-np.eye(24)
    expected = np.zeros((24, 24))
    # 절점 태그 순서는 좌하/우하/좌상/우상이며 요소 순서와 다르다.
    scalar = rho*THICKNESS*6/36*np.array([[4,2,2,1],[2,4,1,2],[2,1,4,2],[1,2,2,4]])
    for d in range(3):
        indices = np.arange(d, 24, 6)
        expected[np.ix_(indices, indices)] = scalar
    return {"rho_diagnostic_only": rho, "mass_expected_per_translation": rho*THICKNESS*6,
            "translation_mass_sums": [float(matrix[d::6, d::6].sum()) for d in range(3)],
            "max_matrix_error": float(np.max(abs(matrix-expected))),
            "max_rotational_mass": float(np.max(abs(matrix[[i for i in range(24) if i%6 >= 3]]))),
            "matrix": matrix.tolist()}


def audit_models():
    rows = []
    weights = original.mass_cases()["U"]
    for d in ("R", "S", "F"):
        model = original.prepare(d, weights)
        tags = ops.getNodeTags()
        mass = [sum(ops.nodeMass(t, k) for t in tags) for k in range(1, 7)]
        shell_tags = [t for t in ops.getEleTags() if t >= 1000000]
        normal_errors = []
        shared = set(t for edge in model["edges"] for t in edge)
        for tag in shell_tags:
            positions = np.array([ops.nodeCoord(n) for n in ops.eleNodes(tag)])
            normal = np.cross(positions[1]-positions[0], positions[3]-positions[0])
            normal_errors.append(normal[2] <= 0)
        initial = {t: [ops.nodeDisp(t, k) for k in (3, 4, 5)] for t in model["levels"][-1]}
        forces = [np.array(ops.eleForce(t)) for t in shell_tags]
        out_plane = [abs(f.reshape(4,6)[:,[2,3,4]]).max() for f in forces]
        rows.append({"case": d, "nodes": len(tags), "frame_elements": len(model["edges"]),
                     "shell_elements": len(shell_tags), "shell_bad_normals": int(sum(normal_errors)),
                     "shell_beam_shared_nodes": len(shared.intersection(t for cs in model.get("shell_elements",[]) for t in cs)),
                     "mass_X": mass[0], "mass_Y": mass[1], "mass_Z": mass[2],
                     "explicit_rotation_mass": sum(mass[3:]),
                     "gravity_reaction_kN": sum(ops.nodeReaction(t,3) for t in model["levels"][0]),
                     "max_shell_gravity_outplane_force": max(out_plane, default=0.),
                     "roof_gravity_Uz_Rx_Ry": initial})
    return rows


def reproduce():
    rows = []
    for name in ("U", "X99", "C99"):
        weights = original.mass_cases()[name]
        for d in ("R", "S", "F"):
            key = f"{d}_{name}"
            print("재현", key, flush=True)
            saved = json.loads((original.OUT/f"case_{key}.json").read_text(encoding="utf-8"))
            modal = original.modal(d, weights)
            value, before = modal["periods_s"][0], saved["modal"]["periods_s"][0]
            rows.append({"case":key, "metric":"T1_s", "old":before,"new":value,
                         "difference":value-before,"difference_pct":100*(value/before-1)})
            fresh = {"modal":modal}
            if name == "C99":
                for direction, label in enumerate(("X", "Y")):
                    static = original.static(d,weights,direction)
                    fresh[f"static_{label}"] = static
                    value = float(np.max(abs(np.array(static[-1]["joint_displacements_mm"])[:,direction])))
                    before = float(np.max(abs(np.array(saved[f"static_{label}"][-1]["joint_displacements_mm"])[:,direction])))
                    rows.append({"case":key,"metric":f"roof_{label}_joint_max_mm","old":before,
                                 "new":value,"difference":value-before,"difference_pct":100*(value/before-1)})
            save_json(OUT/f"reproduced_{key}.json",fresh)
    save_csv(OUT/"reproduction.csv",rows)
    return rows


def plots(tests, reproduction):
    fig, axes = plt.subplots(1,2,figsize=(11,4.3))
    for ax, test in zip(axes,("C_inplane_bending","D_plate_bending")):
        for kind,e in E_VALUES.items():
            rows=[r for r in tests if r["test"]==test and r["E"]==e]
            ax.plot([r["ny"] for r in rows],[r["error_pct"] for r in rows],"o-",label=kind)
        ax.axhline(0,color="gray",lw=.7)
        ax.set(xlabel="Elements across width (nx = 6 ny)",ylabel="Displacement error vs theory (%)",
               title="In-plane bending: Timoshenko approximation" if test.startswith("C") else "Out-of-plane bending: Ep_modifier = 1")
        ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"shell_bending_convergence.png",dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4.2))
    for ax,metric in zip(axes,("T1_s","roof_Y_joint_max_mm")):
        rows=[r for r in reproduction if r["metric"]==metric]
        x=np.arange(len(rows))
        ax.bar(x-.18,[r["old"] for r in rows],.36,label="Saved STEP10")
        ax.bar(x+.18,[r["new"] for r in rows],.36,label="Reproduced STEP12")
        ax.set(xticks=x,xticklabels=[r["case"] for r in rows],
               ylabel="First period (s)" if metric=="T1_s" else "Roof max |Uy| (mm)",
               ylim=(0,1.25*max(max(r["old"],r["new"]) for r in rows)))
        ax.tick_params(axis="x",rotation=35);ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"representative_reproduction.png",dpi=170);plt.close(fig)


def export_tables(data):
    """단위시험별 단위가 다른 결과를 한 CSV에 넣되 응답 단위를 명시한다."""
    rows=[]
    for item in data["tests"]:
        shear_test=item["test"]=="B_shear"
        rows.append({"test":item["test"],"E_kN_m2":item["E"],"Ep_modifier":item["Ep_modifier"],
            "nx":item["nx"],"ny":item["ny"],"response_unit":"kN" if shear_test else "m",
            "theory":item["theory_kN" if shear_test else "theory_m"],
            "FE":item["FE_kN" if shear_test else "FE_m"],"error_pct":item["error_pct"]})
    save_csv(OUT/"shell_unit_tests.csv",rows)
    ratios=[]
    for test in ("A_tension","B_shear","C_inplane_bending","D_plate_bending"):
        selected=[r for r in rows if r["test"]==test]
        ny=max(r["ny"] for r in selected)
        modifier=1. if test.startswith("D") else 0.
        a=next(r for r in selected if r["E_kN_m2"]==2e7 and r["ny"]==ny and r["Ep_modifier"]==modifier)
        b=next(r for r in selected if r["E_kN_m2"]==2e4 and r["ny"]==ny and r["Ep_modifier"]==modifier)
        ratio=a["FE"]/b["FE"] if test=="B_shear" else b["FE"]/a["FE"]
        ratios.append({"test":test,"Ep_modifier":modifier,"expected_S_over_F_stiffness":1000.,
                       "measured_S_over_F_stiffness":ratio,"error_pct":100*(ratio/1000-1)})
    save_csv(OUT/"stiffness_scaling.csv",ratios)
    audits=[{k:v for k,v in r.items() if k!="roof_gravity_Uz_Rx_Ry"} for r in data["audit"]]
    save_csv(OUT/"model_audit.csv",audits)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    before=preservation_manifest()
    save_json(OUT/"original_files_before.json",before)
    environment={"python":sys.version,"platform":platform.platform(),"OpenSees_core":ops.version(),
                 "packages":{n:importlib.metadata.version(n) for n in ("openseespy","openseespywin","numpy","scipy")}}
    tests=[]
    for e in E_VALUES.values():
        for modifier in (0.,1.):
            tests.extend([tension(e,modifier),tension(e,modifier,-1),shear(e,modifier)])
        for n in (2,4,8,16):
            tests.append(cantilever(e,n))
            tests.append(cantilever(e,n,True,1.))
    patch_errors=[abs(r["error_pct"]) for r in tests if r["test"].startswith(("A","B"))]
    if max(patch_errors)>1e-5:
        raise AssertionError("일정 변형률 패치시험 실패")
    zero=[zero_plate_check(e) for e in E_VALUES.values()]
    drilling=[drilling_check(e,m) for e in E_VALUES.values() for m in (0.,1.)]
    mass=[mass_check(rho) for rho in (0.,1.)]
    if max(r["max_matrix_error"] for r in mass)>1e-12:
        raise AssertionError("셸 질량행렬 검증 실패")
    audit=audit_models()
    data={"environment":environment,"tests":tests,"zero_plate":zero,
          "drilling":drilling,"mass":mass,"audit":audit}
    save_json(OUT/"unit_tests.json",data)
    export_tables(data)
    reproduction=reproduce()
    plots(tests,reproduction)
    after=preservation_manifest()
    changed=[p for p,h in before.items() if after.get(p)!=h]
    save_json(OUT/"preservation_check.json",{"files_checked":len(before),"changed":changed})
    if changed: raise AssertionError(f"기존 파일 변경 감지: {changed}")
    print("STEP12 완료", "patch error %",max(patch_errors),"reproduction max %",
          max(abs(r["difference_pct"]) for r in reproduction),flush=True)


if __name__=="__main__":
    run()
