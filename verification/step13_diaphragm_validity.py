"""STEP13: 기존 R/S/F를 보존한 상태에서 연결·보 격자·면외강성을 진단한다.

기존 생성기를 호출한 뒤 독립 OpenSees 도메인만 변경한다. 원본 소스는 수정하지 않는다.
추가 진단모델은 실제 바닥판의 대체 물성을 주장하는 모델이 아니다.
"""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verification import step08_diaphragm_inplane_sensitivity as base
from verification import step10_diaphragm_mass_interaction as original
from verification.step12_shell_audit import ROOT, save_json, save_csv, preservation_manifest

OUT = ROOT / "results/verification/step13_diaphragm_validity"


def diagnostic_builder(kind, p=base.PARAMETERS, subdivisions=8, shell_e=None,
                       member_scale=1., transformation="PDelta", variant="baseline"):
    """기존 모델에서 진단에 필요한 항목만 명시적으로 바꾼다.

    full_plate: Ep=1이며 내부 절점 Uz/Rx/Ry 고정을 해제한다.
    released_drill: 보와 셸의 Ux/Uy만 같게 하고 셸 드릴링 회전 Rz는 분리한다.
    no_shell: 셸을 제거하고 보 격자를 유지한다. 무연결 내부 절점도 제거한다.
    """
    model = base.build_finite_case(kind,p,subdivisions,shell_e,member_scale,transformation)
    if variant == "baseline":
        return model
    if variant not in ("full_plate", "released_drill", "no_shell"):
        raise ValueError(variant)
    elements = [(tag,list(ops.eleNodes(tag))) for tag in ops.getEleTags() if tag >= 1000000]
    beam_nodes = {t for edge in model["edges"] for t in edge}
    shell_nodes = {t for _,corners in elements for t in corners}
    for tag,_ in elements:
        ops.remove("element",tag)
    if variant == "no_shell":
        for tag in shell_nodes-beam_nodes:
            ops.remove("node",tag)
            model["coords"].pop(tag)
        # 기존 공통 골조 절점의 변위만 비교한다. 셸 없는 면적 평균은 정의하지 않는다.
        model["shell_levels"] = [list(nodes) for nodes in model["levels"][1:]]
        model["shell_elements"] = []
        return model
    mapping = {}
    if variant == "full_plate":
        # Ep=0용 내부 고정을 유지하면 판을 지반에 고정하는 오류가 생기므로 해제한다.
        for tag in shell_nodes-beam_nodes:
            for dof in (3,4,5):
                ops.remove("sp",tag,dof)
        e = base.SHELL_E_KN_M2[kind] if shell_e is None else shell_e
        ops.section("ElasticMembranePlateSection",2,e,base.SHELL_POISSON,base.THICKNESS_M,0.,1.)
    else:
        for tag in sorted(shell_nodes & beam_nodes):
            duplicate = tag+2000000
            coords = model["coords"][tag]
            ops.node(duplicate,*coords)
            ops.fix(duplicate,0,0,1,1,1,0)
            # 기존 골조 절점을 retained로 두어 질량은 원래 위치에 한 번만 둔다.
            ops.equalDOF(tag,duplicate,1,2)
            mapping[tag] = duplicate
            model["coords"][duplicate] = coords
    model["shell_elements"] = []
    for tag,corners in elements:
        new_nodes = [mapping.get(t,t) for t in corners]
        ops.element("ShellMITC4",tag,*new_nodes,2 if variant=="full_plate" else 1)
        model["shell_elements"].append(new_nodes)
    # 출력 표본은 원래 골조/셸 격자를 유지한다. 분리 절점의 Ux/Uy는 정확히 같다.
    return model


def evaluate(d, name, variant, mesh=8):
    weights = original.mass_cases()[name]
    with patch.object(original,"build_finite_case",partial(diagnostic_builder,variant=variant)):
        modal = original.modal(d,weights,mesh)
        static_x = original.static(d,weights,0,mesh)
        static_y = original.static(d,weights,1,mesh)
    row = {"case":f"{d}_{name}","variant":variant,"mesh":mesh,"T1_s":modal["periods_s"][0]}
    for i,(label,floors) in enumerate((("X",static_x),("Y",static_y))):
        row[f"roof_{label}_joint_max_mm"] = float(np.max(abs(np.array(floors[-1]["joint_displacements_mm"])[:,i])))
        row[f"roof_{label}_inplane_RMS_mm"] = floors[-1]["inplane_RMS_mm"] if variant!="no_shell" else None
    return row,{"modal":modal,"static_X":static_x,"static_Y":static_y}


def vertical_probe(variant):
    """중력 후 수직 유연도를 비교하는 진단이다. 실제 중력배치는 유지한다."""
    weights=original.mass_cases()["U"]
    with patch.object(original,"build_finite_case",partial(diagnostic_builder,variant=variant)):
        model=original.prepare("S",weights)
    node=model["levels"][-1][5]
    initial=ops.nodeDisp(node,3)
    ops.timeSeries("Linear",2);ops.pattern("Plain",2,2)
    ops.load(node,0,0,-.001,0,0,0)
    original.configure_sparse(1)
    if ops.analyze(1): raise RuntimeError("수직 진단 실패")
    return {"variant":variant,"node":node,"vertical_probe_force_kN":-.001,
            "vertical_compliance_m_per_kN":(ops.nodeDisp(node,3)-initial)/(-.001)}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    before=preservation_manifest()
    rows=[]
    for name in ("U","C99"):
        for d in ("S","F"):
            for variant in ("baseline","released_drill","full_plate"):
                print("진단",name,d,variant,flush=True)
                row,data=evaluate(d,name,variant)
                rows.append(row)
                save_json(OUT/f"diagnostic_{d}_{name}_{variant}.json",data)
        for mesh in (1,8):
            print("보 격자 대조",name,mesh,flush=True)
            row,data=evaluate("F",name,"no_shell",mesh)
            rows.append(row)
            save_json(OUT/f"diagnostic_grid_{name}_{mesh}.json",data)
    for row in rows:
        ref=next(r for r in rows if r["case"]==row["case"] and r["variant"]=="baseline")
        for metric in ("T1_s","roof_X_joint_max_mm","roof_Y_joint_max_mm"):
            row[f"{metric}_change_pct"]=100*(row[metric]/ref[metric]-1)
    probes=[vertical_probe(v) for v in ("baseline","released_drill","full_plate","no_shell")]
    save_csv(OUT/"diagnostic_comparison.csv",rows)
    save_json(OUT/"step13_results.json",{"diagnostics":rows,"vertical_probes":probes,
        "scope":"공유 드릴링, 면외 판강성, 셸 없는 보 격자의 별도 진단. 원본 36조합은 변경하지 않음."})
    plots(rows)
    after=preservation_manifest()
    changed=[p for p,h in before.items() if after.get(p)!=h]
    save_json(OUT/"preservation_check.json",{"files_checked":len(before),"changed":changed})
    if changed: raise AssertionError("기존 파일 변경 감지")
    print("STEP13 진단 완료",flush=True)


def plots(rows):
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    chosen=[r for r in rows if r["case"].endswith("C99") and not(r["variant"]=="no_shell" and r["mesh"]==1)]
    labels=["Beam grid only" if r["variant"]=="no_shell" else r["case"][:1]+" / "+r["variant"] for r in chosen]
    for ax,metric,ylabel in zip(axes,("T1_s","roof_X_joint_max_mm","roof_Y_joint_max_mm"),
                              ("First period (s)","Roof max |Ux| (mm)","Roof max |Uy| (mm)")):
        colors=["#666666" if r["variant"]=="no_shell" else "#3274a1" if r["case"].startswith("S") else "#c45b52" for r in chosen]
        ax.bar(range(len(chosen)),[r[metric] for r in chosen],color=colors)
        ax.set(xticks=range(len(chosen)),xticklabels=labels,ylabel=ylabel)
        ax.tick_params(axis="x",rotation=70);ax.grid(axis="y",alpha=.2)
    fig.suptitle("C99: diagnostic variants (original R/S/F cases are preserved)")
    fig.tight_layout();fig.savefig(OUT/"connection_and_plate_diagnostics.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    run()
