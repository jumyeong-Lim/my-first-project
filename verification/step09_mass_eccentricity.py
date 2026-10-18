"""STEP09: 총질량을 보존한 3D 강체 바닥의 질량편심 단독 민감도.

중력하중과 골조강성은 STEP07 기준을 유지한다. 각 층의 수평 집중질량만
기하학적 중심에서 +X 끝선, +Y 끝선 또는 +X,+Y 모서리로 재배치한다.
"""
from __future__ import annotations

import csv
import json
import sys
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
    PROJECT_ROOT, Parameters3D, build_model, gravity_analysis,
)

P = Parameters3D()
OUT = PROJECT_ROOT / "results/verification/step09_mass_eccentricity"
LEVELS = (0.99, 0.90, 0.75, 0.50, 0.25, 0.10, 0.0)
DIRECTIONS = ("corner", "x_edge", "y_edge")


def mass_weights(xy, direction, concentration):
    """균등질량과 편심 목표 배치를 선형 혼합한다. 총합은 항상 1이다."""
    xy = np.asarray(xy)
    if not 0 <= concentration <= 1:
        raise ValueError("집중률은 0~1이어야 합니다")
    if direction == "corner":
        target = (np.isclose(xy[:, 0], np.max(xy[:, 0])) &
                  np.isclose(xy[:, 1], np.max(xy[:, 1])))
        if concentration == 1:
            raise ValueError("한 절점 100% 집중은 층 회전관성이 0입니다; 0.99 이하를 사용하세요")
    elif direction == "x_edge":
        target = np.isclose(xy[:, 0], np.max(xy[:, 0]))
    elif direction == "y_edge":
        target = np.isclose(xy[:, 1], np.max(xy[:, 1]))
    else:
        raise ValueError(direction)
    target = target.astype(float)/np.count_nonzero(target)
    return (1-concentration)/len(xy)+concentration*target


def mass_matrix(weights, xy, floor_mass):
    """각 층 [Ux,Uy,Rz] 질량행렬: 비대각항이 병진–비틀림 관성결합이다."""
    nodal = floor_mass*np.asarray(weights)
    ex, ey = nodal@xy/floor_mass
    j0 = np.sum(nodal*np.sum(xy**2, axis=1))
    matrix = np.array([[floor_mass, 0, -floor_mass*ey],
                       [0, floor_mass, floor_mass*ex],
                       [-floor_mass*ey, floor_mass*ex, j0]])
    jc = j0-floor_mass*(ex**2+ey**2)
    return matrix, (float(ex),float(ey)), float(jc)


def build_and_load(direction, concentration, p=P):
    model = build_model(p, transformation="PDelta")
    weights = mass_weights(model["xy"],direction,concentration)
    for nodes in model["levels"][1:]:
        for tag,weight in zip(nodes,weights):
            m = p.floor_mass*weight
            ops.mass(tag,m,m,0,0,0,0)
    gravity = gravity_analysis(model,p,1.0)
    if gravity["relative_error"] > 1e-8:
        raise AssertionError("중력 반력 평형 실패")
    return model,weights,gravity


def solve_case(direction, concentration, p=P):
    model,weights,gravity = build_and_load(direction,concentration,p)
    matrix,center,jc = mass_matrix(weights,model["xy"],p.floor_mass)
    if np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise AssertionError("층 질량행렬이 양의 정부호가 아닙니다")
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    eigenvalues = np.asarray(ops.eigen("-fullGenLapack",3*p.stories),float)
    if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
        raise AssertionError("양의 유한 고유치 확인 실패")
    raw = np.array([[ops.nodeEigenvector(tag,n+1,d) for tag in model["masters"]
                     for d in (1,2,6)] for n in range(len(eigenvalues))])
    mglobal = np.kron(np.eye(p.stories),matrix)
    normalized = raw/np.sqrt(np.einsum("ni,ij,nj->n",raw,mglobal,raw))[:,None]
    orthogonality = np.max(np.abs(normalized@mglobal@normalized.T-np.eye(len(eigenvalues))))
    if orthogonality > 1e-7:
        raise AssertionError(f"질량 직교성 오류: {orthogonality}")
    # 질량중심을 기준으로 병진과 비틀림 에너지를 분해한다.
    ex,ey = center
    # q_center = [Ux-ey*Rz, Uy+ex*Rz, Rz]
    energy = []
    for v in normalized:
        floor = v.reshape(p.stories,3)
        ux_c = floor[:,0]-ey*floor[:,2]
        uy_c = floor[:,1]+ex*floor[:,2]
        energy.append([p.floor_mass*np.sum(ux_c**2),p.floor_mass*np.sum(uy_c**2),
                       jc*np.sum(floor[:,2]**2)])
    energy = np.asarray(energy)
    np.testing.assert_allclose(energy.sum(axis=1),1,atol=1e-8)
    # 고정된 평면 길이 척도로 지붕의 회전 유래 수평변위 비중을 표현한다.
    # 회전 운동에너지 비율은 편심 시 Jc가 작아져 응답 회전을 가릴 수 있다.
    radius = float(np.sqrt(np.mean(np.sum(model["xy"]**2,axis=1))))
    rotation_share = []
    for v in normalized:
        ux,uy,theta=v[-3:]
        translation=np.hypot(ux-ey*theta,uy+ex*theta)
        rotation=abs(theta)*radius
        rotation_share.append(float(rotation/np.hypot(translation,rotation)))
    # X/Y 균등 지반가속도와 질량중심 기준 회전 영향벡터의 유효 참여비.
    influence = np.zeros((3*p.stories,3))
    for floor in range(p.stories):
        influence[3*floor:3*floor+3,:] = [[1,0,ey],[0,1,-ex],[0,0,1]]
    influence_mass=influence.T@mglobal@influence
    np.testing.assert_allclose(influence_mass,
        np.diag([p.floor_mass*p.stories,p.floor_mass*p.stories,jc*p.stories]),atol=1e-8)
    denominator = np.einsum("ik,ij,jk->k",influence,mglobal,influence)
    participation = (normalized@mglobal@influence)**2/denominator
    # 회전 영향벡터는 질량중심 기준 물리 회전: q_origin=[ey,-ex,1].
    np.testing.assert_allclose(participation.sum(axis=0),1,atol=1e-7)
    props = ops.modalProperties("-return")
    np.testing.assert_allclose(props["totalFreeMass"][:2],p.floor_mass*p.stories,rtol=1e-9)
    np.testing.assert_allclose(props["centerOfMass"][:2],center,atol=1e-9)
    # modalProperties의 Rz 총질량은 전체 질량중심을 기준으로 한 극관성이다.
    np.testing.assert_allclose(props["totalFreeMass"][5],jc*p.stories,rtol=1e-9)
    return {"direction":direction,"concentration":concentration,"weights":weights.tolist(),
            "center_xy_m":center,"eccentricity_over_halfspan":[ex/(p.lengths[0]/2),ey/(p.lengths[1]/2)],
            "floor_mass":p.floor_mass,"floor_mass_matrix":matrix.tolist(),
            "polar_inertia_about_origin":matrix[2,2],"polar_inertia_about_com":jc,
            "periods_s":(2*np.pi/np.sqrt(eigenvalues)).tolist(),
            "modes":normalized.tolist(),"energy_fraction_X_Y_Rz":energy.tolist(),
            "roof_rotation_displacement_share":rotation_share,
            "effective_participation_X_Y_Rz":participation.tolist(),
            "cumulative_participation_X_Y_Rz":participation.sum(axis=0).tolist(),
            "mass_orthogonality_error":float(orthogonality),"gravity":gravity,
            "opensees_total_free_mass":props["totalFreeMass"],
            "opensees_center_of_mass":props["centerOfMass"]}


def mode_mac(a,b,ma):
    """질량배치가 달라도 같은 기하학적 좌표에서 형상을 비교한다.

    기준 질량행렬을 고정해 MAC을 계산한다. 이 대응은 모드 순번이 바뀌어도
    형상을 찾기 위한 것이며, 두 모델의 서로 다른 질량으로 각각 정규화하지 않는다.
    """
    a,b=np.asarray(a),np.asarray(b)
    metric=np.asarray(ma)
    numerator=(a@metric@b.T)**2
    denominator=np.einsum("ni,ij,nj->n",a,metric,a)[:,None]*np.einsum("ni,ij,nj->n",b,metric,b)[None,:]
    return numerator/denominator


def summarize_branch(reference,cases,p=P):
    metric=np.kron(np.eye(p.stories),np.asarray(reference["floor_mass_matrix"]))
    baseline=np.asarray(reference["modes"])
    for case in cases:
        scores=mode_mac(baseline,case["modes"],metric)
        rows,cols=linear_sum_assignment(-scores)
        order=cols[np.argsort(rows)]
        case["reference_mode_order"]=(order+1).tolist()
        case["reference_mac"]=scores[np.arange(len(order)),order].tolist()


def track_direction(reference,direction,concentrations,p=P):
    """대칭 모델에서 편심을 올리며 이웃 조건 MAC으로 가지를 추적한다."""
    metric=np.kron(np.eye(p.stories),np.asarray(reference["floor_mass_matrix"]))
    cache={0.0:reference}
    rows=[]
    def sample(alpha):
        key=round(alpha,12)
        if key not in cache:
            cache[key]=solve_case(direction,alpha,p)
        return cache[key]
    def advance(low,high,previous,depth=0):
        case=sample(high)
        scores=mode_mac(previous,case["modes"],metric)
        rr,cc=linear_sum_assignment(-scores)
        order=cc[np.argsort(rr)]
        chosen=scores[np.arange(len(order)),order]
        if np.min(chosen)<.90 and depth<6:
            mid=(low+high)/2
            middle=advance(low,mid,previous,depth+1)
            return advance(mid,high,middle,depth+1)
        rows.append({"concentration":high,"actual_mode_order":(order+1).tolist(),
                     "adjacent_MAC":chosen.tolist(),
                     "periods_s":np.asarray(case["periods_s"])[order].tolist(),
                     "roof_rotation_share":np.asarray(case["roof_rotation_displacement_share"])[order].tolist()})
        return np.asarray(case["modes"])[order]
    previous=np.asarray(reference["modes"])
    low=0.0
    for high in sorted(concentrations):
        previous=advance(low,high,previous)
        low=high
    return rows


def run():
    reference=solve_case("corner",0)
    cases=[]
    for direction in DIRECTIONS:
        for concentration in LEVELS:
            if concentration == 0:
                continue
            print(f"STEP09 {direction} {concentration:g}",flush=True)
            cases.append(solve_case(direction,concentration))
        if direction != "corner":
            print(f"STEP09 {direction} 1",flush=True)
            cases.append(solve_case(direction,1.0))
    summarize_branch(reference,cases)
    tracking={direction:track_direction(reference,direction,
        [c["concentration"] for c in cases if c["direction"]==direction])
        for direction in DIRECTIONS}
    return {"assumptions":{"floor_type":"미정","diaphragm":"강체 기준",
            "gravity":"STEP07 균등 중력하중 유지; 수평 질량만 재배치",
            "eccentricity":"모든 3개 층에 동일 배치; 코너 최대 집중률 0.99",
            "mass":"기존 12개 골조 절점의 수평질량만 사용"},
            "opensees_version":ops.version(),"reference":reference,"cases":cases,
            "tracking":tracking}


def write(data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"step09_results.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"step09_modes.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.writer(file)
        writer.writerow(["direction","concentration","center_x_m","center_y_m","mode","period_s",
                         "energy_X","energy_Y","energy_Rz","roof_rotation_share","effective_X","effective_Y","effective_Rz",
                         "reference_mode","reference_MAC"])
        for case in [data["reference"],*data["cases"]]:
            for n,period in enumerate(case["periods_s"]):
                writer.writerow([case["direction"],case["concentration"],*case["center_xy_m"],n+1,period,
                    *case["energy_fraction_X_Y_Rz"][n],case["roof_rotation_displacement_share"][n],
                    *case["effective_participation_X_Y_Rz"][n],
                    n+1 if case is data["reference"] else case["reference_mode_order"].index(n+1)+1,
                    1 if case is data["reference"] else case["reference_mac"][case["reference_mode_order"].index(n+1)]])
    with (OUT/"step09_tracked_modes.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.writer(file)
        writer.writerow(["direction","concentration","rigid_origin_mode","actual_mode",
                         "period_s","roof_rotation_share","adjacent_MAC"])
        for direction,rows in data["tracking"].items():
            for row in rows:
                for origin in range(9):
                    writer.writerow([direction,row["concentration"],origin+1,
                        row["actual_mode_order"][origin],row["periods_s"][origin],
                        row["roof_rotation_share"][origin],row["adjacent_MAC"][origin]])
    xy=np.array([(i*P.bay_length-P.lengths[0]/2,j*P.bay_length-P.lengths[1]/2)
                 for j in range(P.bays_y+1) for i in range(P.bays_x+1)])
    selected=[data["reference"]]+[next(c for c in data["cases"] if c["direction"]==d and
        np.isclose(c["concentration"],a)) for d,a in (("corner",.99),("x_edge",1),("y_edge",1))]
    fig,axes=plt.subplots(1,4,figsize=(14,3.8),sharex=True,sharey=True)
    for ax,case,title in zip(axes,selected,("Uniform","99% corner","100% +X edge","100% +Y edge")):
        weights=np.array(case["weights"])
        ax.scatter(xy[:,0],xy[:,1],s=30+900*weights,color="#2c73a9",alpha=.75)
        ax.scatter(*case["center_xy_m"],marker="x",s=95,color="#c14539",linewidths=2)
        for (x,y),w in zip(xy,weights):
            if w>.05: ax.text(x,y+0.45,f"{w:.0%}",ha="center",fontsize=7)
        ax.set(title=title,xlabel="X (m)",xlim=(-11,11),ylim=(-8,8),xticks=[-9,-3,3,9],yticks=[-6,0,6])
        ax.set_aspect("equal");ax.grid(alpha=.2)
    axes[0].set_ylabel("Y (m)")
    fig.suptitle("STEP09: floor mass distribution (each of three floors); red x = mass center")
    fig.tight_layout();fig.savefig(OUT/"step09_mass_layouts.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for direction in DIRECTIONS:
        rows=[data["reference"]]+sorted((c for c in data["cases"] if c["direction"]==direction),key=lambda c:c["concentration"])
        x=[r["concentration"] for r in rows]
        axes[0].plot(x,[r["periods_s"][0] for r in rows],"o-",label=direction)
        axes[1].plot(x,[r["roof_rotation_displacement_share"][0] for r in rows],"o-",label=direction)
    axes[0].set(xlabel="Mass concentration fraction",ylabel="First period (s)",title="First-mode period")
    axes[1].set(xlabel="Mass concentration fraction",ylabel="Roof rotation displacement share",title="First-mode translation–torsion coupling")
    for ax in axes: ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/"step09_extreme_to_symmetric.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,case,title in zip(axes,selected[1:],("99% corner","100% +X edge","100% +Y edge")):
        energy=np.array(case["energy_fraction_X_Y_Rz"][:9])*100
        bottom=np.zeros(9)
        for j,label in enumerate(("X translation","Y translation","torsion")):
            ax.bar(np.arange(1,10),energy[:,j],bottom=bottom,label=label)
            bottom+=energy[:,j]
        ax.set(title=title,xlabel="Mode",ylabel="Kinetic-energy share (%)",xticks=np.arange(1,10),ylim=(0,105))
    axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/"step09_modal_coupling.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,case,title in zip(axes.flat,selected,("Uniform","99% corner","100% +X edge","100% +Y edge")):
        ux,uy,theta=case["modes"][0][-3:]
        motion=np.column_stack((ux-theta*xy[:,1],uy+theta*xy[:,0]))
        motion*=1.4/np.max(np.linalg.norm(motion,axis=1))
        moved=xy+motion
        for j in range(P.bays_y+1):
            idx=np.arange(j*(P.bays_x+1),(j+1)*(P.bays_x+1))
            ax.plot(*xy[idx].T,"--",color=".7")
            ax.plot(*moved[idx].T,color="#286ba5")
        for i in range(P.bays_x+1):
            ax.plot(*xy[i::P.bays_x+1].T,"--",color=".7")
            ax.plot(*moved[i::P.bays_x+1].T,color="#286ba5")
        ax.set(title=f"{title}: T1={case['periods_s'][0]:.3f} s",xlabel="X (m)",ylabel="Y (m)",
               xlim=(-11,11),ylim=(-8,8));ax.set_aspect("equal")
    fig.suptitle("STEP09 first-mode roof plan: gray original, blue mode shape\nEach panel independently amplified for shape comparison")
    fig.subplots_adjust(top=.84,bottom=.08,hspace=.55,wspace=.24)
    fig.savefig(OUT/"step09_first_mode_roof_shapes.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for direction in DIRECTIONS:
        rows=data["tracking"][direction]
        for origin in (0,1,2):
            x=[0]+[r["concentration"] for r in rows]
            label=f"{direction}: ref {origin+1}"
            axes[0].plot(x,[data["reference"]["periods_s"][origin]]+
                [r["periods_s"][origin] for r in rows],label=label)
            axes[1].plot(x,[origin+1]+[r["actual_mode_order"][origin] for r in rows],label=label)
            axes[2].plot(x,[data["reference"]["roof_rotation_displacement_share"][origin]]+
                [r["roof_rotation_share"][origin] for r in rows],label=label)
    axes[0].set(xlabel="Mass concentration",ylabel="Period (s)",title="Tracked branches: periods")
    axes[1].set(xlabel="Mass concentration",ylabel="Actual eigenmode order",title="Mode ordering changes")
    axes[2].set(xlabel="Mass concentration",ylabel="Roof rotation displacement share",title="Tracked branch coupling")
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(OUT/"step09_tracked_modes.png",dpi=170);plt.close(fig)


if __name__ == "__main__":
    result=run();write(result)
    for case in (result["reference"],*result["cases"]):
        if case["concentration"] in (0,.99,1):
            print(case["direction"],case["concentration"],case["center_xy_m"],case["periods_s"][:3],flush=True)
