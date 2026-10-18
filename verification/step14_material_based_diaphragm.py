"""STEP14: 바닥 형식과 물성에 연결되는 면내강성 기준모델.

RC150은 실험 보정 모델이 아니라 가정한 균질 RC 슬래브의 물성 기반 후보다.
NIST의 유효강성 범위를 참고하되 저감계수는 현재 하중으로 식별한 값이 아니다.
기존 STEP01~13의 코드/결과는 변경하지 않고 동일 질량·중력 조건으로 비교한다.
실행: python -m verification.step14_material_based_diaphragm
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verification import step08_diaphragm_inplane_sensitivity as base
from verification import step10_diaphragm_mass_interaction as original
from verification.step12_shell_audit import ROOT, save_json, save_csv, solve, edge_weights

OUT = ROOT / "results/verification/step14_material_based_diaphragm"


@dataclass(frozen=True)
class Slab:
    """E·nu·두께는 연구 가정이며 저감계수는 문헌 범위를 참고한다."""
    code: str
    reduction: float
    concrete_E_kN_m2: float = 30e6
    thickness_m: float = .15
    poisson: float = .20

    def __post_init__(self):
        if not (self.concrete_E_kN_m2 > 0 and self.thickness_m > 0
                and 0 < self.reduction <= 1 and -1 < self.poisson < .5):
            raise ValueError("물성·두께·저감계수 범위를 확인하세요")

    @property
    def E(self):
        return self.reduction * self.concrete_E_kN_m2

    @property
    def A(self):
        # 공학 전단변형률 gamma_xy를 쓰는 평면응력 막 구성행렬 (kN/m).
        nu = self.poisson
        return self.E*self.thickness_m/(1-nu**2)*np.array(
            [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    def record(self):
        return {**asdict(self), "E_effective_kN_m2": self.E,
                "Et_kN_m": self.E*self.thickness_m,
                "Gt_kN_m": self.A[2, 2], "A_kN_m": self.A.tolist(),
                "status": "가정 물성 기반 후보; 실험 보정/실건물 설계값 아님"}


SLABS = {s.code: s for s in (Slab("RC150_a10", .1), Slab("RC150_a30", .3),
                            Slab("RC150_a100", 1.))}
ORDER = ["R", "S", "F", *SLABS]
COLORS = ["#333333", "#377eb8", "#e07a37", "#9853a5", "#3c9c70", "#b69a24"]


def material_builder(spec, kind, p=base.PARAMETERS, subdivisions=8, shell_e=None,
                     member_scale=1., transformation="PDelta"):
    """기존 보/기둥 격자를 생성하고 셸 단면만 물성 기반 단면으로 교체한다.

    보와 셸은 격자 절점을 공유한다(완전 부착). 면외강성과 셸질량은 0이다.
    기존 단면 번호 1은 제거하지 않고 새 단면 14를 모든 셸에 연결한다.
    """
    model = base.build_finite_case(kind, p, subdivisions, spec.E,
                                   member_scale, transformation)
    elements = [(t, list(ops.eleNodes(t))) for t in ops.getEleTags() if t >= 1000000]
    for tag, _ in elements:
        ops.remove("element", tag)
    ops.section("ElasticMembranePlateSection", 14, spec.E, spec.poisson,
                spec.thickness_m, 0., 0.)
    for tag, nodes in elements:
        ops.element("ShellMITC4", tag, *nodes, 14)
    model["material_basis"] = spec.record()
    return model


@contextmanager
def use_material(code):
    """패치 범위 종료 후 원래 생성기와 사전이 자동 복구된다."""
    if code not in SLABS:
        yield
        return
    spec = SLABS[code]
    from functools import partial
    with patch.dict(original.DIAPHRAGMS, {code: spec.E}), patch.object(
            original, "build_finite_case", partial(material_builder, spec)):
        yield


def panel(spec, n):
    """2 m x 1 m 작은 패널. 실제 바닥 경간이 아닌 구성식 검증용 치수다."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.section("ElasticMembranePlateSection", 14, spec.E, spec.poisson,
                spec.thickness_m, 0., 0.)
    nx, ny = 2*n, n
    nodes = {}
    for j in range(ny+1):
        for i in range(nx+1):
            tag = 1+j*(nx+1)+i
            nodes[i, j] = tag
            ops.node(tag, i/n, j/n, 0.)
    for j in range(ny):
        for i in range(nx):
            ops.element("ShellMITC4", 1+j*nx+i, nodes[i,j], nodes[i+1,j],
                        nodes[i+1,j+1], nodes[i,j+1], 14)
    return nodes, nx, ny


def axial_test(spec, n, direction):
    """자유 포아송 수축 상태에서 P/delta=Et*b/L을 X/Y 각각 검증한다."""
    nodes, nx, ny = panel(spec, n)
    for (i,j), tag in nodes.items():
        fixed = (int(i == 0), int(i == 0 and j == 0)) if direction == 0 else (
            int(i == 0 and j == 0), int(j == 0))
        ops.fix(tag, *fixed, 1, 1, 1, 0)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    boundary = [nodes[nx,j] for j in range(ny+1)] if direction == 0 else [nodes[i,ny] for i in range(nx+1)]
    weights = edge_weights(len(boundary)-1)
    for tag, w in zip(boundary, weights):
        force = [0.]*6
        force[direction] = w  # 전체 인장력 1 kN
        ops.load(tag, *force)
    solve()
    delta = sum(w*ops.nodeDisp(t, direction+1) for t,w in zip(boundary, weights))
    length, width = (2., 1.) if direction == 0 else (1., 2.)
    theory = spec.E*spec.thickness_m*width/length
    return {"model": spec.code, "mesh_n": n, "test": "tension_X" if direction == 0 else "tension_Y",
            "theory_kN_m": theory, "FE_kN_m": 1/delta,
            "error_pct": 100*((1/delta)/theory-1), "displacement_m": delta,
            "equilibrium_error_kN": 1+sum(ops.nodeReaction(t,direction+1) for t in nodes.values())}


def constitutive_test(spec, n):
    """세 독립 일정변형률 시험으로 A11/A12/A22/A66을 반력에서 복원한다."""
    recovered = np.zeros((3,3))
    strain = 1e-6
    for col in range(3):
        nodes, nx, ny = panel(spec, n)
        for tag in nodes.values():
            ops.fix(tag, 0, 0, 1, 1, 1, 0)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for (i,j), tag in nodes.items():
            if i in (0,nx) or j in (0,ny):
                x,y,_ = ops.nodeCoord(tag)
                ops.sp(tag, 1, strain*(x if col == 0 else y if col == 2 else 0.))
                ops.sp(tag, 2, strain*y if col == 1 else 0.)
        solve()
        resultants = np.zeros(3)
        for tag in nodes.values():
            x,y,_ = ops.nodeCoord(tag)
            rx,ry = ops.nodeReaction(tag,1), ops.nodeReaction(tag,2)
            resultants += [rx*x, ry*y, rx*y]
        recovered[:,col] = resultants/(2.*strain)
    error = np.max(abs(recovered-spec.A))/np.max(abs(spec.A))
    return {"model": spec.code, "mesh_n": n, "A_theory_kN_m": spec.A.tolist(),
            "A_FE_kN_m": recovered.tolist(), "max_scaled_error": float(error),
            "Gt_FE_kN_m": recovered[2,2]}


def evaluate(code, name, mesh=8):
    weights = original.mass_cases()[name]
    with use_material(code):
        modal = original.modal(code, weights, mesh)
        sx = original.static(code, weights, 0, mesh)
        sy = original.static(code, weights, 1, mesh)
    return {"diaphragm": code, "mass_case": name, "weights": weights.tolist(),
            "mesh_per_bay": mesh, "modal": modal, "static_X": sx, "static_Y": sy}


def file_manifest():
    files = [p for parent in (ROOT/"verification", ROOT/"tests", ROOT/"results/verification")
             for p in parent.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and "step14" not in p.as_posix().lower()]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def plots(properties, axial, results, matches):
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    finite = [r for r in properties if r["model"] != "R"]
    for ax, key, title in zip(axes, ("Et_kN_m", "Gt_kN_m"), ("Free-contraction axial rigidity Et", "Shear rigidity Gt")):
        ax.bar([r["model"] for r in finite], [r[key] for r in finite], color=COLORS[1:])
        ax.set(yscale="log", ylabel="Rigidity per unit width (kN/m)", title=title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/"step14_membrane_rigidities.png",dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(14,4.5))
    for ax, name in zip(axes, ("U","C50","C99")):
        for code,color in zip(ORDER,COLORS):
            case = results[f"{code}_{name}"]
            periods = case["modal"]["periods_s"][:9] if code == "R" else case["match_to_rigid"]["periods_s"]
            ax.plot(range(1,10),periods,"o-",label=code,color=color,ms=3)
        ax.set(title=name, xlabel="Matched rigid mode", ylabel="Period (s)", ylim=(0,None))
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT/"step14_matched_periods.png",dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for row,d in enumerate(("X","Y")):
        for code,color in zip(ORDER,COLORS):
            floors = results[f"{code}_C99"][f"static_{d}"]
            axes[row,0].plot([f["max_directional_mm"] for f in floors],range(1,4),"o-",color=color,label=code)
            axes[row,1].plot([f["inplane_RMS_mm"] for f in floors],range(1,4),"o-",color=color,label=code)
        axes[row,0].set(xlabel=f"Max |U{d.lower()}| (mm)",ylabel="Floor",title=f"C99 / {d}: unit acceleration")
        axes[row,1].set(xlabel="In-plane residual RMS (mm)",ylabel="Floor",title=f"C99 / {d}: rigid motion removed")
        for ax in axes[row]: ax.set_yticks([1,2,3]); ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT/"step14_all_floor_displacements.png",dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for ax,d in zip(axes,("X","Y")):
        for code,color in zip(ORDER,COLORS):
            floor=results[f"{code}_C99"][f"static_{d}"][-1]
            disp=np.array(floor["joint_displacements_mm"])[:,0 if d=="X" else 1]
            ax.plot(range(1,13),disp,"o-",color=color,label=code,ms=3)
        ax.set(xlabel="Roof joint: X increases first, then Y",ylabel=f"U{d.lower()} (mm)",title=f"C99 / {d}")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/"step14_roof_positions.png",dpi=180);plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    before = file_manifest()
    properties = []
    for code,e,t in [("S",2e7,.12),("F",2e4,.12)]+[(c,s.E,s.thickness_m) for c,s in SLABS.items()]:
        properties.append({"model":code,"E_kN_m2":e,"t_m":t,"nu":.2,
                           "Et_kN_m":e*t,"Gt_kN_m":e*t/2.4,
                           "equivalent_alpha_for_RC150":e*t/(30e6*.15)})
    save_csv(OUT/"material_properties.csv",properties)
    axial, constitutive = [], []
    for spec in SLABS.values():
        for n in (1,2,4):
            axial.extend(axial_test(spec,n,d) for d in (0,1))
            constitutive.append(constitutive_test(spec,n))
    assert max(abs(r["error_pct"]) for r in axial) < 1e-5
    assert max(r["max_scaled_error"] for r in constitutive) < 1e-7
    assert max(abs(r["equilibrium_error_kN"]) for r in axial) < 1e-7
    save_csv(OUT/"panel_axial_tests.csv",axial)
    save_json(OUT/"panel_constitutive_tests.json",constitutive)
    # 기존 36개 결과는 읽기 전용으로 사용하고 대표 6개를 새로 계산해 재현한다.
    saved = json.loads((original.OUT/"step10_results.json").read_text(encoding="utf-8"))
    assert saved["metadata"]["mesh"] == 8
    results = saved["results"]
    reproduction = []
    for name in ("U","C99"):
        for code in ("R","S","F"):
            print("reproduce",code,name,flush=True)
            fresh = evaluate(code,name)
            oldv,newv = original.scalar_results(results[f"{code}_{name}"]),original.scalar_results(fresh)
            for metric in ("T1_s","X_roof_max_mm","Y_roof_max_mm"):
                error = 100*(newv[metric]/oldv[metric]-1)
                reproduction.append({"case":f"{code}_{name}","metric":metric,"difference_pct":error})
                assert abs(error)<1e-5
    save_csv(OUT/"baseline_reproduction.csv",reproduction)
    for name,weights in original.mass_cases().items():
        for code in SLABS:
            print("physical",code,name,flush=True)
            case = evaluate(code,name)
            results[f"{code}_{name}"] = case
            save_json(OUT/f"case_{code}_{name}.json",case)
    matches, summary, floors = [], [], []
    for name,weights in original.mass_cases().items():
        md = np.tile(np.repeat(original.P.floor_mass*weights,2),original.P.stories)
        rigid = results[f"R_{name}"]["modal"]
        a = np.array(rigid["vectors"])
        for code in ORDER:
            case = results[f"{code}_{name}"]
            summary.append({"model":code,"mass_case":name,**original.scalar_results(case)})
            if code != "R":
                b = np.array(case["modal"]["vectors"])
                scores = ((a*md)@b.T)**2
                rr,cc = linear_sum_assignment(-scores)
                idx = cc[np.argsort(rr)]
                case["match_to_rigid"] = {"orders":(idx+1).tolist(),"MAC":scores[np.arange(9),idx].tolist(),
                    "periods_s":np.array(case["modal"]["periods_s"])[idx].tolist()}
                for i,j in enumerate(idx):
                    matches.append({"model":code,"mass_case":name,"rigid_mode":i+1,"matched_mode":int(j+1),
                        "MAC":float(scores[i,j]),"period_s":case["modal"]["periods_s"][j],
                        "period_change_pct":100*(case["modal"]["periods_s"][j]/rigid["periods_s"][i]-1),
                        "participation_X":case["modal"]["participation_X_Y_Rz"][j][0],
                        "participation_Y":case["modal"]["participation_X_Y_Rz"][j][1],
                        "participation_Rz":case["modal"]["participation_X_Y_Rz"][j][2]})
            for d in ("X","Y"):
                for f in case[f"static_{d}"]:
                    floors.append({"model":code,"mass_case":name,"direction":d,"floor":f["floor"],
                        "max_displacement_mm":f["max_directional_mm"],"inplane_RMS_mm":f["inplane_RMS_mm"]})
    convergence=[]
    for code in SLABS:
        for mesh in (4,12):
            print("mesh",code,mesh,flush=True)
            fine=evaluate(code,"C99",mesh)
            ref=original.scalar_results(results[f"{code}_C99"])
            for metric,value in original.scalar_results(fine).items():
                if metric == "first_mode_roof_rotation_share": continue
                convergence.append({"model":code,"mass_case":"C99","mesh":mesh,"reference_mesh":8,
                    "metric":metric,"value":value,"reference":ref[metric],
                    "change_pct":100*(value/ref[metric]-1) if abs(ref[metric])>1e-10 else None})
            save_json(OUT/f"mesh_{code}_C99_n{mesh}.json",fine)
    contrasts=original.contrasts(results)
    save_csv(OUT/"step14_summary.csv",summary)
    save_csv(OUT/"step14_matched_modes.csv",matches)
    save_csv(OUT/"step14_floor_displacements.csv",floors)
    save_csv(OUT/"step14_interactions.csv",contrasts)
    save_csv(OUT/"step14_mesh_check.csv",convergence)
    data={"specifications":{c:s.record() for c,s in SLABS.items()},"results":results,
          "metadata":{"mesh":8,"OpenSees":ops.version(),"shell_density":0,"Ep_modifier":0,
          "floor_weight_kN":original.P.floor_weight,"gravity":"uniform fixed", "mass":"same nodal distribution for each comparison",
          "interpretation":"물성 기반의 가정 RC 후보. 연결/철근/균열 실험 보정과 실제 자중 재현은 포함하지 않음"}}
    save_json(OUT/"step14_results.json",data)
    plots(properties,axial,results,matches)
    after=file_manifest()
    changed=[p for p,h in before.items() if after.get(p)!=h]
    save_json(OUT/"verification_checks.json",{"original_files_checked":len(before),"changed":changed,
        "max_panel_axial_error_pct":max(abs(r["error_pct"]) for r in axial),
        "max_panel_A_scaled_error":max(r["max_scaled_error"] for r in constitutive),
        "max_baseline_reproduction_error_pct":max(abs(r["difference_pct"]) for r in reproduction),
        "physical_case_count":36,"baseline_case_count":36,
        "min_physical_matched_MAC":min(r["MAC"] for r in matches if r["model"] in SLABS),
        "mesh12_max_change_pct":max(abs(r["change_pct"]) for r in convergence if r["mesh"]==12 and r["change_pct"] is not None)})
    if changed: raise AssertionError(f"기존 파일 변경: {changed}")
    print("STEP14 completed",flush=True)
    from verification.step14_mode_audit import run as audit_modes
    audit_modes()
    from verification.step14_reporting import run as write_report
    write_report()


if __name__ == "__main__":
    run()
