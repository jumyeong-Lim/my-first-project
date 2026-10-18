"""STEP15: 중앙/외곽 골조선의 부호 있는 층전단력 분배.

기존 총질량·균등 중력·PDelta·면내 전용 셸 조건을 사용한다.
추가 관성력은 질량×1 m/s²이며 지진 시간이력 최대응답은 아니다.
기둥 하단 전역 요소력의 중력 전후 차이에서 상부 자유물체의 전단력을 구한다.
실행: python -m verification.step15_frame_shear_distribution
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import openseespy.opensees as ops

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verification.step14_material_based_diaphragm import (
    ROOT, ORDER, SLABS, use_material, original,
)
from verification.step12_shell_audit import save_json, save_csv
from verification.step08_validation import configure_sparse

OUT = ROOT / "results/verification/step15_frame_shear_distribution"
MESH = 8
FORCE_TOL = 1e-7


def protected_manifest():
    """STEP01~14의 코드/결과/시험을 해시로 보존 확인한다."""
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (ROOT/"verification", ROOT/"results/verification", ROOT/"tests")
            for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and "step15" not in p.as_posix().lower()}


def column_catalog(model):
    """요소 번호 추정 대신 좌표로 수직기둥과 하단 i/j 위치를 식별한다."""
    columns = []
    for tag in ops.getEleTags():
        nodes = list(ops.eleNodes(tag))
        if len(nodes) != 2:
            continue
        coords = np.array([model["coords"][t] for t in nodes])
        if not np.allclose(coords[0,:2], coords[1,:2]) or np.isclose(coords[0,2],coords[1,2]):
            continue
        lower = int(np.argmin(coords[:,2]))
        upper = 1-lower
        floor = int(round(coords[upper,2]/original.P.story_height))
        columns.append({"element":int(tag), "nodes":nodes, "lower_index":lower,
                        "floor":floor, "x_m":float(coords[lower,0]), "y_m":float(coords[lower,1])})
    expected = original.P.stories*len(original.XY)
    if len(columns) != expected:
        raise AssertionError(f"수직기둥 수 {len(columns)} != {expected}")
    return columns


def lower_force(column, values):
    """[Fi,Mi,Fj,Mj] 전역 요소력에서 실제 기둥 하단 6성분을 뽑는다."""
    return np.asarray(values).reshape(2,6)[column["lower_index"]]


def axis_probe(transformation="PDelta", direction=0, reverse=False):
    """단일 캔틸레버의 알려진 1 kN으로 하단 부호와 PDelta 차이를 검증한다."""
    ops.wipe()
    ops.model("basic","-ndm",3,"-ndf",6)
    ops.node(1,0,0,0); ops.node(2,0,0,4)
    ops.fix(1,1,1,1,1,1,1)
    ops.geomTransf(transformation,1,0,1,0)
    nodes=[2,1] if reverse else [1,2]
    ops.element("elasticBeamColumn",1,*nodes,.02,2e8,2e8/2.6,8e-6,2e-4,4e-4,1)
    ops.timeSeries("Linear",1); ops.pattern("Plain",1,1)
    ops.load(2,0,0,-50,0,0,0)
    configure_sparse(1.)
    if ops.analyze(1): raise RuntimeError("단일기둥 중력 검증 실패")
    before=np.array(ops.eleForce(1));ops.loadConst("-time",0)
    ops.timeSeries("Linear",2);ops.pattern("Plain",2,2)
    force=[0.]*6;force[direction]=1.
    ops.load(2,*force)
    configure_sparse(1.)
    if ops.analyze(1): raise RuntimeError("단일기둥 횡력 검증 실패")
    ops.reactions()
    lower=nodes.index(1)
    measured=-(np.array(ops.eleForce(1))-before).reshape(2,6)[lower,direction]
    reaction=ops.nodeReaction(1,direction+1)
    # localForce는 PDelta의 전역 평형 전단력과 다를 수 있음을 수치로 기록한다.
    local=np.array(ops.eleResponse(1,"localForce")).reshape(2,6)[lower,:3]
    ex=np.array([0.,0.,-1. if reverse else 1.])
    ey=np.cross([0.,1.,0.],ex); ez=np.cross(ex,ey)
    transformed=local[0]*ex+local[1]*ey+local[2]*ez
    np.testing.assert_allclose([measured,reaction],[1.,-1.],atol=1e-10)
    return {"transformation":transformation,"direction":"XY"[direction],"reversed_i_j":reverse,
            "load_kN":1.,"lower_global_cut_shear_kN":float(measured),"base_reaction_kN":reaction,
            "mapped_local_shear_kN":float(-transformed[direction])}


def calculate_case(code, name, direction, mesh=MESH, weights_override=None):
    """중력 전후 기둥 전역력 차이 → 층/골조선 전단력 및 자유물체 평형."""
    weights=original.mass_cases()[name] if weights_override is None else np.asarray(weights_override)
    np.testing.assert_allclose(sum(weights),1.,atol=1e-12)
    with use_material(code):
        model=original.prepare(code,weights,mesh)
    columns=column_catalog(model)
    before={c["element"]:np.array(ops.eleForce(c["element"])) for c in columns}
    base_before={t:np.array(ops.nodeReaction(t)) for t in model["levels"][0]}
    disp_before={t:np.array([ops.nodeDisp(t,d) for d in (1,2)])
                 for nodes in model["levels"][1:] for t in nodes}
    ops.timeSeries("Linear",2);ops.pattern("Plain",2,2)
    force_per_floor=original.P.floor_mass
    applied=[]
    for floor,nodes in enumerate(model["levels"][1:],1):
        for tag,w in zip(nodes,weights):
            load=np.zeros(6);load[direction]=force_per_floor*w
            ops.load(tag,*load)
            applied.append({"floor":floor,"xy":model["coords"][tag][:2],"load":load[:3]})
    configure_sparse(1.)
    if ops.analyze(1): raise RuntimeError(f"정적해석 실패 {code}/{name}/{'XY'[direction]}")
    ops.reactions()
    column_rows=[]
    for c in columns:
        increment=np.array(ops.eleForce(c["element"]))-before[c["element"]]
        # 음수 부호: 상부 자유물체가 하부에 요구하는 방향을 양수로 통일한다.
        cut=-lower_force(c,increment)
        column_rows.append({**c,"cut_Fx_kN":float(cut[0]),"cut_Fy_kN":float(cut[1]),
            "cut_Mz_kNm":float(cut[5]),"V_direction_kN":float(cut[direction]),
            "gravity_global_lower_force":lower_force(c,before[c["element"]]).tolist(),
            "increment_global_lower_force":(-cut).tolist()})
    floor_rows=[];line_rows=[];group_rows=[];checks=[]
    lines=np.unique(original.XY[:,1-direction])
    edge_value=max(abs(lines))
    for floor in range(1,original.P.stories+1):
        cs=[c for c in column_rows if c["floor"]==floor]
        expected=np.zeros(3);expected_torque=0.
        for p in applied:
            if p["floor"]>=floor:
                expected+=p["load"]
                x,y=p["xy"];fx,fy,_=p["load"]
                expected_torque+=x*fy-y*fx
        measured=np.array([sum(c[k] for c in cs) for k in ("cut_Fx_kN","cut_Fy_kN")])
        measured_torque=sum(c["x_m"]*c["cut_Fy_kN"]-c["y_m"]*c["cut_Fx_kN"]+c["cut_Mz_kNm"] for c in cs)
        np.testing.assert_allclose(measured,expected[:2],atol=FORCE_TOL,rtol=1e-8)
        np.testing.assert_allclose(measured_torque,expected_torque,atol=1e-6,rtol=1e-8)
        checks.append({"floor":floor,"force_error_kN":float(np.max(abs(measured-expected[:2]))),
            "expected_shear_kN":float(expected[direction]),"measured_shear_kN":float(measured[direction]),
            "expected_torque_kNm":float(expected_torque),"measured_torque_kNm":float(measured_torque),
            "torque_error_kNm":float(measured_torque-expected_torque)})
        abs_sum=sum(abs(c["V_direction_kN"]) for c in cs)
        layer=[]
        for line in lines:
            subset=[c for c in cs if np.isclose(c["y_m" if direction==0 else "x_m"],line)]
            value=sum(c["V_direction_kN"] for c in subset)
            layer.append({"floor":floor,"line_axis":"Y" if direction==0 else "X","line_coord_m":float(line),
                "group":"exterior" if np.isclose(abs(line),edge_value) else "interior",
                "column_count":len(subset),"V_kN":float(value),"signed_share_pct":float(100*value/expected[direction]),
                "column_abs_share_pct":float(100*sum(abs(c["V_direction_kN"]) for c in subset)/abs_sum)})
        line_rows.extend(layer)
        for group in ("interior","exterior"):
            subset=[r for r in layer if r["group"]==group]
            value=sum(r["V_kN"] for r in subset)
            group_rows.append({"floor":floor,"group":group,"line_count":len(subset),"V_kN":value,
                "signed_share_sum_pct":100*value/expected[direction],
                "signed_share_per_line_mean_pct":100*value/expected[direction]/len(subset)})
        negative=sum(min(r["V_kN"],0.) for r in layer)
        floor_rows.append({"floor":floor,"total_shear_kN":float(measured[direction]),
            "largest_line_share_pct":max(r["signed_share_pct"] for r in layer),
            "smallest_line_share_pct":min(r["signed_share_pct"] for r in layer),
            "reverse_line_count":sum(r["V_kN"] < -1e-8 for r in layer),
            "reverse_shear_ratio_pct":float(-100*negative/expected[direction])})
    # 기초반력은 전체 자유물체의 평형으로 독립 대조한다.
    base_increments={t:np.array(ops.nodeReaction(t))-base_before[t] for t in model["levels"][0]}
    base_total=np.sum(list(base_increments.values()),axis=0)
    np.testing.assert_allclose(base_total[:2],-expected_base(direction),atol=FORCE_TOL,rtol=1e-8)
    base_error=max(abs(base_total[d]+sum(c["cut_Fx_kN" if d==0 else "cut_Fy_kN"]
                   for c in column_rows if c["floor"]==1)) for d in (0,1))
    joints=np.array([[np.array([ops.nodeDisp(t,d) for d in (1,2)])-disp_before[t] for t in nodes]
                     for nodes in model["levels"][1:]])
    return {"model":code,"mass_case":name,"direction":"XY"[direction],"mesh_per_bay":mesh,
        "floor_mass":original.P.floor_mass,"center_xy_m":(weights@original.XY).tolist(),
        "line_results":line_rows,"group_results":group_rows,"floor_summary":floor_rows,
        "columns":column_rows,"equilibrium":checks,"base_reaction_increment":base_total.tolist(),
        "base_cut_reaction_error_kN":float(base_error),"joint_displacements_mm":(joints*1000).tolist()}


def expected_base(direction):
    value=np.zeros(2);value[direction]=original.P.stories*original.P.floor_mass
    return value


def flatten(cases,key):
    return [{"model":c["model"],"mass_case":c["mass_case"],"direction":c["direction"],**r}
            for c in cases.values() for r in c[key]]


def validate_symmetry(cases):
    """U의 대칭선 일치와 R의 동일 중심·다른 질량분포의 정적 동등성을 검사한다."""
    error=0.
    for code in ORDER:
        for direction in ("X","Y"):
            c=cases[f"{code}_U_{direction}"]
            for floor in (1,2,3):
                rows=[r for r in c["line_results"] if r["floor"]==floor]
                for a,b in zip(rows,rows[::-1]): error=max(error,abs(a["signed_share_pct"]-b["signed_share_pct"]))
    if error>1e-6: raise AssertionError(f"대칭선 불일치 {error}")
    # 균일 가속도에서 R의 하중은 총질량·중심으로 결정되며 극관성은 정적 하중에 들어가지 않는다.
    inertia_error=0.
    for direction in ("X","Y"):
        reference=cases[f"R_C50_{direction}"]["line_results"]
        for name in ("C50_Jlo","C50_Jhi"):
            values=cases[f"R_{name}_{direction}"]["line_results"]
            inertia_error=max(inertia_error,max(abs(a["V_kN"]-b["V_kN"]) for a,b in zip(reference,values)))
    if inertia_error>FORCE_TOL: raise AssertionError("R의 동일 중심 대조군 불일치")
    return {"max_uniform_mirror_share_error_percentage_points":error,
            "max_rigid_same_center_inertia_control_difference_kN":inertia_error}


def run(representative_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    before=protected_manifest()
    probes=[axis_probe(t,d,r) for t in ("Linear","PDelta") for d in (0,1) for r in (False,True)]
    save_json(OUT/"single_column_axis_probes.json",probes)
    source=json.loads((ROOT/"results/verification/step14_material_based_diaphragm/step14_results.json").read_text(encoding="utf-8"))["results"]
    names=["U","C99"] if representative_only else ["U","C99"]+[n for n in original.mass_cases() if n not in ("U","C99")]
    cases={};reproduction=[]
    for name in names:
        for code in ORDER:
            for d in (0,1):
                key=f"{code}_{name}_{'XY'[d]}"
                print("STEP15",key,flush=True)
                result=calculate_case(code,name,d)
                reference=np.array(source[f"{code}_{name}"][f"static_{'XY'[d]}"][-1]["joint_displacements_mm"])
                actual=np.array(result["joint_displacements_mm"])[-1]
                error=float(np.max(abs(actual-reference)))
                np.testing.assert_allclose(actual,reference,atol=1e-7,rtol=1e-7)
                reproduction.append({"case":key,"max_roof_joint_difference_mm":error})
                cases[key]=result
                save_json(OUT/f"case_{key}.json",result)
    save_csv(OUT/"step15_line_shares.csv",flatten(cases,"line_results"))
    save_csv(OUT/"step15_group_shares.csv",flatten(cases,"group_results"))
    save_csv(OUT/"step15_floor_summary.csv",flatten(cases,"floor_summary"))
    save_csv(OUT/"step15_equilibrium.csv",flatten(cases,"equilibrium"))
    save_csv(OUT/"step15_reproduction.csv",reproduction)
    validations=validate_symmetry(cases) if not representative_only else {}
    # 대표 C99에서 선별 분담률의 메시 민감도를 확인한다. R에는 셸이 없다.
    mesh_rows=[]
    if not representative_only:
        for code in ORDER[1:]:
            for d in (0,1):
                for mesh in (4,12):
                    print("mesh",code,"XY"[d],mesh,flush=True)
                    fine=calculate_case(code,"C99",d,mesh)
                    reference=cases[f"{code}_C99_{'XY'[d]}"]
                    for a,b in zip(reference["line_results"],fine["line_results"]):
                        mesh_rows.append({"model":code,"direction":"XY"[d],"mesh":mesh,"floor":a["floor"],
                            "line_coord_m":a["line_coord_m"],"reference_share_pct":a["signed_share_pct"],
                            "share_pct":b["signed_share_pct"],"difference_percentage_points":b["signed_share_pct"]-a["signed_share_pct"]})
        save_csv(OUT/"step15_mesh_check.csv",mesh_rows)
    # 같은 질량배치의 R 대비 차이와 U를 뺀 상호작용을 부호 있는 %p로 저장한다.
    interaction=[]
    for c in cases.values():
        ref=cases[f"R_{c['mass_case']}_{c['direction']}"]
        uniform=cases[f"{c['model']}_U_{c['direction']}"]
        rigid_u=cases[f"R_U_{c['direction']}"]
        for a,b,u,ru in zip(c["line_results"],ref["line_results"],uniform["line_results"],rigid_u["line_results"]):
            delta=a["signed_share_pct"]-b["signed_share_pct"]
            interaction.append({"model":c["model"],"mass_case":c["mass_case"],"direction":c["direction"],
                "floor":a["floor"],"line_coord_m":a["line_coord_m"],"diaphragm_effect_percentage_points":delta,
                "interaction_percentage_points":delta-(u["signed_share_pct"]-ru["signed_share_pct"])})
    save_csv(OUT/"step15_interactions.csv",interaction)
    data={"cases":cases,"metadata":{"models":ORDER,"mass_cases":names,"mesh_per_bay":MESH,
        "acceleration_m_s2":1.,"floor_weight_kN":original.P.floor_weight,"transformation":"PDelta",
        "force_definition":"minus gravity-subtracted lower-end global resisting force",
        "scope":"탄성 모멘트골조/면내 전용 다이어프램/동일 질량/단위가속도 정적 관성력",
        "sources":["https://openseespydoc.readthedocs.io/en/latest/src/eleForce.html",
                   "https://openseespydoc.readthedocs.io/en/latest/src/pdelta.html"]}}
    save_json(OUT/"step15_results.json",data)
    after=protected_manifest();changed=[p for p,h in before.items() if after.get(p)!=h]
    checks={"case_count":len(cases),"model_mass_combinations":len(cases)//2,"original_files_checked":len(before),"changed":changed,
        "max_force_equilibrium_error_kN":max(r["force_error_kN"] for c in cases.values() for r in c["equilibrium"]),
        "max_torque_equilibrium_error_kNm":max(abs(r["torque_error_kNm"]) for c in cases.values() for r in c["equilibrium"]),
        "max_base_cut_reaction_error_kN":max(c["base_cut_reaction_error_kN"] for c in cases.values()),
        "max_roof_reproduction_error_mm":max(r["max_roof_joint_difference_mm"] for r in reproduction),
        "mesh12_max_share_change_percentage_points":max((abs(r["difference_percentage_points"]) for r in mesh_rows if r["mesh"]==12),default=0),
        **validations}
    save_json(OUT/"verification_checks.json",checks)
    if changed: raise AssertionError(f"기존 파일 변경 {changed}")
    if not representative_only:
        from verification.step15_reporting import run as report
        report()
    print("STEP15 completed",json.dumps(checks,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--representative-only",action="store_true")
    run(parser.parse_args().representative_only)
