"""STEP10: 다이어프램 면내강성 × 질량배치의 개별 효과와 상호작용.

재료 탄성, 중력 균등 선행하중, 세 층 동일 질량배치를 사용한다.
절대 변위는 질량×1 m/s²의 정적 관성력으로 비교한다. 지진 최대응답이 아니다.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from scipy.optimize import linprog, linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from verification.step08_diaphragm_inplane_sensitivity import (
    build_case, build_finite_case, joint_horizontal_array, SHELL_E_KN_M2,
)
from verification.step08_validation import configure_sparse, floor_geometry, floor_metrics
from verification.step09_mass_eccentricity import P, mass_weights, mass_matrix, PROJECT_ROOT

OUT=PROJECT_ROOT/"results/verification/step10_diaphragm_mass_interaction"
DIAPHRAGMS={"R":None,"S":SHELL_E_KN_M2["semirigid"],"F":SHELL_E_KN_M2["flexible"]}
LABELS={"R":"Rigid","S":"Finite E=2e7","F":"Low E=2e4"}
XY=np.array([(i*P.bay_length-P.lengths[0]/2,j*P.bay_length-P.lengths[1]/2)
             for j in range(P.bays_y+1) for i in range(P.bays_x+1)])
RADIUS=np.sqrt(np.mean(np.sum(XY**2,axis=1)))
MESH=8


def mass_cases():
    """같은 질량중심의 회전관성 대조는 비음수 절점질량 제약 아래 구한다."""
    cases={"U":np.full(len(XY),1/len(XY))}
    for prefix,direction in (("X","x_edge"),("Y","y_edge"),("C","corner")):
        for alpha in (.10,.50,.99):
            cases[f"{prefix}{int(alpha*100):02d}"]=mass_weights(XY,direction,alpha)
    # C50과 같은 질량중심 (4.5,3.0)에서 극관성을 최소/최대로 설정한다.
    # 약간의 분산 질량을 남겨 극단적인 무관성 자유도를 만들지 않는다.
    constraint=np.vstack((np.ones(len(XY)),XY.T))
    target=constraint@cases["C50"]
    radius2=np.sum(XY**2,axis=1)
    for name,sign in (("C50_Jlo",1),("C50_Jhi",-1)):
        result=linprog(sign*radius2,A_eq=constraint,b_eq=target,
                       bounds=[(.01/len(XY),None)]*len(XY),method="highs")
        if not result.success: raise RuntimeError(result.message)
        cases[name]=result.x
    return cases


def prepare(diaphragm,weights,mesh=MESH,member_scale=1.0,transformation="PDelta"):
    # STEP11에서는 질량·중력하중은 그대로 두고 가정 단면과 기둥 기하변환만 바꾼다.
    model=build_case("rigid",member_scale=member_scale,transformation=transformation) if diaphragm=="R" else build_finite_case(
        "flexible",subdivisions=mesh,shell_e=DIAPHRAGMS[diaphragm],
        member_scale=member_scale,transformation=transformation)
    for nodes in model["levels"][1:]:
        for tag,w in zip(nodes,weights):
            m=P.floor_mass*w
            ops.mass(tag,m,m,0,0,0,0)
    ops.timeSeries("Linear",1);ops.pattern("Plain",1,1)
    for nodes in model["levels"][1:]:
        for tag in nodes: ops.load(tag,0,0,-P.floor_weight/len(nodes),0,0,0)
    configure_sparse(.1)
    if ops.analyze(10): raise RuntimeError("중력 선행해석 실패")
    ops.reactions()
    total=sum(ops.nodeReaction(t,3) for t in model["levels"][0])
    np.testing.assert_allclose(total,P.floor_weight*P.stories,rtol=1e-8)
    ops.loadConst("-time",0)
    return model


def characterize(vector,weights):
    """질량중심의 강체 병진·회전·면내 잔차로 운동에너지를 분해한다."""
    _,center,jc=mass_matrix(weights,XY,P.floor_mass)
    centered=XY-np.array(center)
    geometry=floor_geometry(centered)
    root=np.sqrt(np.repeat(P.floor_mass*weights,2))
    energy=np.zeros(4);distortions=[];rotations=[]
    for floor in np.asarray(vector).reshape(P.stories,-1,2):
        fit=np.linalg.lstsq(geometry*root[:,None],floor.ravel()*root,rcond=None)[0]
        residual=floor.ravel()-geometry@fit
        energy+=np.array([P.floor_mass*fit[0]**2,P.floor_mass*fit[1]**2,
                          jc*fit[2]**2,np.sum((root*residual)**2)])
        total=np.sum((root*floor.ravel())**2)
        distortions.append(float(np.sqrt(np.sum((root*residual)**2)/total)))
        rot=abs(fit[2])*RADIUS
        rotations.append(float(rot/np.sqrt(fit[0]**2+fit[1]**2+rot**2)))
    np.testing.assert_allclose(sum(energy),1,atol=1e-7)
    return {"energy_X_Y_Rz_distortion":energy.tolist(),
            "floor_modal_distortion_ratio":distortions,"floor_rotation_share":rotations}


def modal(diaphragm,weights,mesh=MESH,member_scale=1.0,transformation="PDelta"):
    model=prepare(diaphragm,weights,mesh,member_scale,transformation)
    ops.wipeAnalysis();ops.constraints("Transformation");ops.numberer("RCM");ops.system("UmfPack")
    count=9 if diaphragm=="R" else 48
    eigenvalues=np.asarray(ops.eigen("-fullGenLapack" if diaphragm=="R" else "-genBandArpack",count))
    if np.any(eigenvalues<=0) or not np.all(np.isfinite(eigenvalues)): raise AssertionError("고유치 오류")
    md=np.tile(np.repeat(P.floor_mass*weights,2),P.stories)
    vectors=np.array([joint_horizontal_array(model,lambda t,d:ops.nodeEigenvector(t,n+1,d)).ravel()
                      for n in range(count)])
    vectors/=np.sqrt(np.sum(vectors**2*md,axis=1))[:,None]
    orthogonality=float(np.max(np.abs((vectors*md)@vectors.T-np.eye(count))))
    if orthogonality>1e-6: raise AssertionError("질량 직교성 실패")
    _,center,jc=mass_matrix(weights,XY,P.floor_mass)
    influence=np.tile(floor_geometry(XY-np.array(center)),(P.stories,1))
    participation=((vectors*md)@influence)**2/np.sum(md[:,None]*influence**2,axis=0)
    props=ops.modalProperties("-return")
    np.testing.assert_allclose(props["totalFreeMass"][:2],P.floor_mass*P.stories,rtol=1e-8)
    np.testing.assert_allclose(props["centerOfMass"][:2],center,atol=1e-8)
    np.testing.assert_allclose(props["totalFreeMass"][5],jc*P.stories,rtol=1e-8)
    if np.min(participation.sum(axis=0))<.99: raise AssertionError("누적 참여비 99% 미달: 모드 추가 필요")
    return {"periods_s":(2*np.pi/np.sqrt(eigenvalues)).tolist(),"vectors":vectors.tolist(),
            "participation_X_Y_Rz":participation.tolist(),
            "cumulative_participation":participation.sum(axis=0).tolist(),
            "orthogonality_error":orthogonality,
            "character": [characterize(v,weights) for v in vectors]}


def static(diaphragm,weights,direction,mesh=MESH,acceleration=1.0,
           member_scale=1.0,transformation="PDelta"):
    """세 층 모든 절점에 F=m×1 m/s²를 가한 추가 정적 변위를 저장한다."""
    model=prepare(diaphragm,weights,mesh,member_scale,transformation)
    floors=model.get("shell_levels",model["levels"][1:])
    initial=[np.array([[ops.nodeDisp(t,d) for d in (1,2)] for t in nodes]) for nodes in floors]
    ops.timeSeries("Linear",2);ops.pattern("Plain",2,2)
    for nodes in model["levels"][1:]:
        for tag,w in zip(nodes,weights):
            force=[0.0]*6;force[direction]=P.floor_mass*w*acceleration
            ops.load(tag,*force)
    configure_sparse(1)
    if ops.analyze(1): raise RuntimeError("단위가속도 정적해석 실패")
    ops.reactions()
    horizontal=sum(ops.nodeReaction(t,direction+1) for t in model["levels"][0])
    np.testing.assert_allclose(horizontal,-P.floor_mass*P.stories*acceleration,rtol=1e-7)
    records=[]
    for level,(nodes,base) in enumerate(zip(floors,initial),1):
        xy=np.array([model["coords"][t][:2] for t in nodes])
        displacement=np.array([[ops.nodeDisp(t,d) for d in (1,2)] for t in nodes])-base
        idx=[nodes.index(t) for t in model["levels"][level]]
        area=np.ones(len(nodes))
        if diaphragm!="R":
            area[np.isclose(abs(xy[:,0]),P.lengths[0]/2)]*=.5
            area[np.isclose(abs(xy[:,1]),P.lengths[1]/2)]*=.5
        metrics=floor_metrics(xy,displacement,area)
        records.append({"floor":level,"joint_displacements_mm":(1000*displacement[idx]).tolist(),
            "xy":xy.tolist(),"field_displacements_mm":(1000*displacement).tolist(),
            "max_directional_mm":float(1000*np.max(abs(displacement[:,direction]))),
            "inplane_RMS_mm":1000*metrics["distortion_rms"],
            "rigid_fit_Ux_Uy_Rz":metrics["rigid_motion"]})
    return records


def scalar_results(case):
    return {"T1_s":case["modal"]["periods_s"][0],
            "first_mode_roof_rotation_share":case["modal"]["character"][0]["floor_rotation_share"][-1],
            "X_roof_max_mm":case["static_X"][-1]["max_directional_mm"],
            "Y_roof_max_mm":case["static_Y"][-1]["max_directional_mm"],
            "X_roof_distortion_mm":case["static_X"][-1]["inplane_RMS_mm"],
            "Y_roof_distortion_mm":case["static_Y"][-1]["inplane_RMS_mm"]}


def contrasts(results):
    """상호작용 = (D,E)-(R,E)-(D,U)+(R,U). 기준이 0이면 절대차를 쓴다."""
    output=[]
    base=scalar_results(results["R_U"])
    for key,case in results.items():
        d,e=case["diaphragm"],case["mass_case"]
        values=scalar_results(case)
        rigid=scalar_results(results[f"R_{e}"])
        uniform=scalar_results(results[f"{d}_U"])
        for metric,value in values.items():
            delta=value-rigid[metric]
            interaction=delta-(uniform[metric]-base[metric])
            # 균등 강체의 왜곡은 0이므로 백분율을 만들지 않는다.
            usable=abs(base[metric])>1e-8
            output.append({"case":key,"metric":metric,"value":value,
                "mass_effect_at_rigid":rigid[metric]-base[metric],
                "diaphragm_effect_at_uniform":uniform[metric]-base[metric],
                "diaphragm_effect_at_same_mass":delta,"interaction":interaction,
                "diaphragm_change_pct_same_mass":100*delta/rigid[metric] if abs(rigid[metric])>1e-8 else None,
                "interaction_pct_of_R_U":100*interaction/base[metric] if usable else None})
    return output


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    distributions=mass_cases();results={}
    for name,weights in distributions.items():
        for d in DIAPHRAGMS:
            key=f"{d}_{name}";print(key,flush=True)
            mat,center,jc=mass_matrix(weights,XY,P.floor_mass)
            result={"diaphragm":d,"mass_case":name,"weights":weights.tolist(),
                "center_xy_m":center,"Jc":jc,"floor_mass":P.floor_mass,
                "mesh_per_bay":1 if d=="R" else MESH,
                "modal":modal(d,weights),"static_X":static(d,weights,0),"static_Y":static(d,weights,1)}
            results[key]=result
            (OUT/f"case_{key}.json").write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
    for name,weights in distributions.items():
        md=np.tile(np.repeat(P.floor_mass*weights,2),P.stories)
        a=np.asarray(results[f"R_{name}"]["modal"]["vectors"])
        for d in ("S","F"):
            case=results[f"{d}_{name}"]
            b=np.asarray(case["modal"]["vectors"])
            scores=((a*md)@b.T)**2
            rr,cc=linear_sum_assignment(-scores)
            idx=cc[np.argsort(rr)]
            case["match_to_rigid"]={"orders":(idx+1).tolist(),
                "MAC":scores[np.arange(9),idx].tolist(),
                "periods_s":np.asarray(case["modal"]["periods_s"])[idx].tolist()}
    return {"results":results,"contrasts":contrasts(results),
            "metadata":{"acceleration_m_s2":1,"gravity":"균등 유지",
                "diaphragm_E":DIAPHRAGMS,"mesh":MESH,"opensees":ops.version()}}


def write(data):
    results=data["results"];names=list(mass_cases())
    (OUT/"step10_results.json").write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    with (OUT/"step10_case_catalog.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["mass_case","center_x_m","center_y_m","floor_mass","Jc"]+[f"node_{i}_mass_fraction" for i in range(1,13)])
        for name in names:
            case=results[f"R_{name}"]
            writer.writerow([name,*case["center_xy_m"],case["floor_mass"],case["Jc"],*case["weights"]])
    fig,axes=plt.subplots(3,4,figsize=(14,10))
    for ax,name in zip(axes.flat,names):
        case=results[f"R_{name}"]
        ax.scatter(XY[:,0],XY[:,1],s=20+500*np.asarray(case["weights"]),color="#3274a1",alpha=.75)
        ax.scatter(*case["center_xy_m"],s=65,marker="x",color="#bf4b45")
        ax.set(title=f"{name}: e=({case['center_xy_m'][0]:.2f},{case['center_xy_m'][1]:.2f}) m",
               xlim=(-11,11),ylim=(-8,8),xticks=[-9,-3,3,9],yticks=[-6,0,6],xlabel="X (m)",ylabel="Y (m)")
        ax.set_aspect("equal");ax.grid(alpha=.2)
    fig.suptitle("STEP10 mass cases: blue circle area indicates mass; red x = mass center\nSame mass layout on all three floors; paired with R / S / F diaphragms")
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/"step10_mass_case_catalog.png",dpi=160);plt.close(fig)
    with (OUT/"step10_interactions.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data["contrasts"][0]));writer.writeheader();writer.writerows(data["contrasts"])
    with (OUT/"step10_floor_displacements.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["case","load","floor","x_m","y_m","ux_mm","uy_mm"])
        for key,case in results.items():
            for load in ("X","Y"):
                for floor in case[f"static_{load}"]:
                    for xy,u in zip(XY,floor["joint_displacements_mm"]):writer.writerow([key,load,floor["floor"],*xy,*u])
    with (OUT/"step10_modal_comparison.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["case","rigid_reference_mode","actual_mode","period_s","same_mass_rigid_period_s",
                         "period_change_pct","MAC","MAC_at_least_0p8","participation_X","participation_Y","participation_Rz"])
        for key,case in results.items():
            ref=results[f"R_{case['mass_case']}"]["modal"]
            for n in range(9):
                j=n if case["diaphragm"]=="R" else case["match_to_rigid"]["orders"][n]-1
                mac=1 if case["diaphragm"]=="R" else case["match_to_rigid"]["MAC"][n]
                period=case["modal"]["periods_s"][j];base=ref["periods_s"][n]
                writer.writerow([key,n+1,j+1,period,base,100*(period/base-1),mac,mac>=.8,
                                 *case["modal"]["participation_X_Y_Rz"][j]])
    fig,axes=plt.subplots(3,1,figsize=(14,12))
    for ax,metric,title in zip(axes,("T1_s","X_roof_max_mm","Y_roof_max_mm"),
        ("First period (s)","Roof max |Ux| (mm), ax=1 m/s²","Roof max |Uy| (mm), ay=1 m/s²")):
        for i,d in enumerate(DIAPHRAGMS):
            vals=[scalar_results(results[f"{d}_{n}"])[metric] for n in names]
            ax.bar(np.arange(len(names))+(i-1)*.24,vals,.24,label=LABELS[d])
        ax.set(xticks=np.arange(len(names)),xticklabels=names,ylabel=title);ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"step10_by_mass_case.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,metric in zip(axes,("T1_s","X_roof_max_mm","Y_roof_max_mm")):
        vals=np.array([[next(c["interaction_pct_of_R_U"] for c in data["contrasts"]
                    if c["case"]==f"{d}_{n}" and c["metric"]==metric) for n in names] for d in ("S","F")])
        limit=max(np.max(abs(vals)),1e-8)
        im=ax.imshow(vals,aspect="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
        for i in range(2):
            for j in range(len(names)):ax.text(j,i,f"{vals[i,j]:.2f}",ha="center",va="center",fontsize=7,
                color="white" if abs(vals[i,j])>.55*limit else "black")
        ax.set(xticks=range(len(names)),xticklabels=names,yticks=[0,1],yticklabels=["S","F"],title=metric)
        ax.tick_params(axis="x",rotation=65);fig.colorbar(im,ax=ax,label="Interaction / R_U (%)")
    fig.tight_layout();fig.savefig(OUT/"step10_interaction_heatmap.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,load in zip(axes,("X","Y")):
        direction="XY".index(load)
        for d in DIAPHRAGMS:
            case=results[f"{d}_C99"]
            u=np.array(case[f"static_{load}"][-1]["joint_displacements_mm"])
            ax.plot(range(1,13),u[:,direction],"o-",label=LABELS[d])
        ax.set(xlabel="Roof joint (X ascending, then Y rows)",ylabel=f"U{load.lower()} (mm)",
               title=f"C99: {load} unit-acceleration load",xticks=range(1,13));ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"step10_roof_position_comparison.png",dpi=170);plt.close(fig)
    for direction,load in enumerate(("X","Y")):
        fig,axes=plt.subplots(3,4,figsize=(16,11))
        all_values=np.concatenate([np.asarray(case[f"static_{load}"][-1]["joint_displacements_mm"])[:,direction]
                                   for case in results.values()])
        limits=(min(0,float(np.min(all_values)))*1.05,max(0,float(np.max(all_values)))*1.05)
        for ax,name in zip(axes.flat,names):
            for d in DIAPHRAGMS:
                floor=results[f"{d}_{name}"][f"static_{load}"][-1]
                u=np.asarray(floor["joint_displacements_mm"])
                ax.plot(range(1,13),u[:,direction],"o-",markersize=3,label=d)
            ax.set(title=name,xlabel="Roof joint",ylabel=f"U{load.lower()} (mm)",xticks=[1,4,5,8,9,12],ylim=limits);ax.grid(alpha=.2)
            for divider in (4.5,8.5):ax.axvline(divider,color=".6",linestyle=":",linewidth=.6)
        axes[0,0].legend()
        fig.suptitle(f"All mass layouts: roof positions under {load} unit acceleration\nR=rigid, S=finite E2e7, F=low E2e4; joint rows Y=-6,0,+6; X increases within each row")
        fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/f"step10_roof_{load}_all_cases.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,load in zip(axes,("X","Y")):
        for d in DIAPHRAGMS:
            floors=results[f"{d}_C99"][f"static_{load}"]
            ax.plot([f["inplane_RMS_mm"] for f in floors],[1,2,3],"o-",label=LABELS[d])
        ax.set(title=f"C99, {load} acceleration",xlabel="Area-weighted in-plane RMS (mm)",ylabel="Floor",yticks=[1,2,3]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"step10_all_floor_distortion.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for d in DIAPHRAGMS:
        axes[0].plot(names,[results[f"{d}_{n}"]["modal"]["character"][0]["floor_rotation_share"][-1] for n in names],"o-",label=LABELS[d])
        axes[1].plot(names,[results[f"{d}_{n}"]["modal"]["character"][0]["floor_modal_distortion_ratio"][-1] for n in names],"o-",label=LABELS[d])
    axes[0].set(ylabel="First-mode roof rotation displacement share")
    axes[1].set(ylabel="First-mode roof mass-weighted distortion ratio")
    for ax in axes: ax.tick_params(axis="x",rotation=60);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"step10_mode_coupling.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    data=run();write(data)
    print("STEP10 completed",len(data["results"]),"cases",flush=True)
