"""STEP16: 저장된 STEP14/15 72조합의 동특성·변형·횡력 분배 통합.

이 스크립트는 새 OpenSees 해석을 수행하지 않는다. 공통 절점, 질량가중치,
표본 위치와 단위를 명시하여 기존 결과를 연결하고 대조군·상호작용을 재계산한다.
실행: python -m verification.step16_integrated_elastic_conclusion
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from verification.step16_modal_tools import (
    weighted_mac, match_modes, mode_clusters, subspace_similarity,
)

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results/verification/step16_integrated_elastic_conclusion"
DIR14=ROOT/"results/verification/step14_material_based_diaphragm"
DIR15=ROOT/"results/verification/step15_frame_shear_distribution"
MODELS=["R","S","F","RC150_a10","RC150_a30","RC150_a100"]
NAMES=["U","X10","X50","X99","Y10","Y50","Y99","C10","C50","C99","C50_Jlo","C50_Jhi"]
XY=np.array([(x,y) for y in (-6.,0.,6.) for x in (-9.,-3.,3.,9.)])
SOURCES=[DIR14/"step14_results.json",DIR15/"step15_results.json",
         DIR14/"step14_mesh_check.csv",DIR14/"higher_mode_mesh_check.csv",
         DIR15/"step15_mesh_check.csv",DIR14/"verification_checks.json",
         DIR14/"mode_audit_checks.json",DIR15/"verification_checks.json"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def save_csv(path,rows):
    if not rows: return
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def preservation_manifest():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (ROOT/"verification",ROOT/"tests",ROOT/"results/verification")
            for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and "step16" not in p.as_posix().lower()}


def rigid_geometry(xy=XY):
    a=np.zeros((2*len(xy),3))
    a[0::2,0]=1.;a[1::2,1]=1.
    a[0::2,2]=-xy[:,1];a[1::2,2]=xy[:,0]
    return a


def joint_distortion(displacement):
    """전체 모델 공통 12절점의 균등 표본 RMS. 원 결과의 면적가중 RMS와 구분한다."""
    vector=np.array(displacement).ravel()
    a=rigid_geometry();fit=np.linalg.lstsq(a,vector,rcond=None)[0]
    residual=(vector-a@fit).reshape(-1,2)
    return float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))


def geometry_rotation_fraction(vector):
    """기하학적 중심·균등 기준으로 회전변위 성분을 비교한다. 운동에너지는 아니다."""
    radius2=float(np.mean(np.sum(XY**2,axis=1)))
    motions=np.asarray(vector).reshape(3,12,2)
    fit=np.array([np.linalg.lstsq(rigid_geometry(),u.ravel(),rcond=None)[0] for u in motions])
    rot=np.sum(fit[:,2]**2)*radius2
    total=np.sum(fit[:,:2]**2)+rot
    return float(rot/total) if total>1e-30 else 0.


def physical_energy(vector,weights,floor_mass):
    """질량중심 기준의 강체 병진·회전·잔차 운동에너지를 원 모드에서 재검산한다."""
    center=weights@XY
    a=rigid_geometry(XY-center)
    md=np.repeat(floor_mass*weights,2);root=np.sqrt(md)
    jc=float(np.sum(floor_mass*weights*np.sum((XY-center)**2,axis=1)))
    energy=np.zeros(4)
    for u in np.asarray(vector).reshape(3,24):
        fit=np.linalg.lstsq(a*root[:,None],u*root,rcond=None)[0]
        residual=u-a@fit
        energy+=np.array([floor_mass*fit[0]**2,floor_mass*fit[1]**2,
                          jc*fit[2]**2,np.sum(md*residual**2)])
    return energy


def validate_sources(d14,d15):
    """통합 전에 사례키·좌표·질량·중심·저장 변위의 일치를 대조한다."""
    cases=d14["results"];forces=d15["cases"]
    assert set(cases)=={f"{c}_{n}" for c in MODELS for n in NAMES}
    assert set(forces)=={f"{c}_{n}_{d}" for c in MODELS for n in NAMES for d in ("X","Y")}
    assert d14["metadata"]["mesh"]==d15["metadata"]["mesh_per_bay"]==8
    assert d14["metadata"]["floor_weight_kN"]==d15["metadata"]["floor_weight_kN"]==200.
    assert d15["metadata"]["acceleration_m_s2"]==1.
    assert d15["metadata"]["transformation"]=="PDelta"
    max_disp=0.;max_energy=0.;max_orth=0.
    for name in NAMES:
        weights=np.array(cases[f"R_{name}"]["weights"])
        np.testing.assert_allclose(sum(weights),1.,atol=1e-12)
        assert np.min(weights)>0.
        for code in MODELS:
            case=cases[f"{code}_{name}"]
            np.testing.assert_allclose(case["weights"],weights,atol=1e-12)
            modal=case["modal"];v=np.array(modal["vectors"])
            mass=forces[f"{code}_{name}_X"]["floor_mass"]
            md=np.tile(np.repeat(mass*weights,2),3)
            assert v.shape[1]==72
            assert np.min(modal["periods_s"])>0.
            assert min(modal["cumulative_participation"])>=.99
            orth=np.max(abs((v*md)@v.T-np.eye(len(v))))
            max_orth=max(max_orth,float(orth))
            np.testing.assert_allclose((v*md)@v.T,np.eye(len(v)),atol=1e-6)
            for i in range(9):
                energy=physical_energy(v[i],weights,mass)
                expected=modal["character"][i]["energy_X_Y_Rz_distortion"]
                max_energy=max(max_energy,float(np.max(abs(energy-expected))))
                np.testing.assert_allclose(energy,expected,atol=1e-8)
            for direction in ("X","Y"):
                force=forces[f"{code}_{name}_{direction}"]
                assert force["direction"]==direction and force["model"]==code
                np.testing.assert_allclose(force["center_xy_m"],weights@XY,atol=1e-10)
                for floor in (1,2,3):
                    static=case[f"static_{direction}"][floor-1]
                    xy=np.array(static["xy"])
                    for index,position in enumerate(XY):
                        found=np.flatnonzero(np.all(np.isclose(xy,position),axis=1))
                        assert len(found)==1
                        np.testing.assert_allclose(np.array(static["field_displacements_mm"])[found[0]],
                            static["joint_displacements_mm"][index],atol=1e-8)
                    diff=np.array(static["joint_displacements_mm"])-np.array(force["joint_displacements_mm"])[floor-1]
                    max_disp=max(max_disp,float(np.max(abs(diff))))
                    np.testing.assert_allclose(diff,0.,atol=1e-7)
    return {"case_count":len(cases),"directional_case_count":len(forces),
            "max_shared_displacement_difference_mm":max_disp,
            "max_physical_energy_recalculation_error":max_energy,
            "max_mass_orthogonality_error":max_orth}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=preservation_manifest()
    d14=read_json(SOURCES[0]);d15=read_json(SOURCES[1])
    checks=validate_sources(d14,d15)
    cases=d14["results"];forces=d15["cases"]
    floor_mass=forces["R_U_X"]["floor_mass"]
    fixed_weight=np.full(72,floor_mass/12)
    modal_rows=[];mass_mode_rows=[];cluster_rows=[];summary=[];floors=[];positions=[];lines=[];catalog=[]
    for name in NAMES:
        weights=np.array(cases[f"R_{name}"]["weights"])
        center=weights@XY
        jc=float(np.sum(floor_mass*weights*np.sum((XY-center)**2,axis=1)))
        md=np.tile(np.repeat(floor_mass*weights,2),3)
        ref=cases[f"R_{name}"]["modal"]
        for code in MODELS:
            case=cases[f"{code}_{name}"];modal=case["modal"]
            vectors=np.array(modal["vectors"])
            same=match_modes(np.array(ref["vectors"]),vectors,md)
            uniform=cases[f"{code}_U"]["modal"]
            cross=match_modes(np.array(uniform["vectors"])[:9],vectors,fixed_weight)
            family="numerical_stiffness_cases" if code in ("R","S","F") else "assumed_RC_material_cases"
            catalog.append({"model":code,"mass_case":name,"model_group":family,"floor_mass":floor_mass,
                "center_x_m":float(center[0]),"center_y_m":float(center[1]),"ex_over_Lx":float(center[0]/18),
                "ey_over_Ly":float(center[1]/12),"J_about_CM":jc,"mass_weights":json.dumps(weights.tolist())})
            for i,j in enumerate(same["indices"]):
                p=np.array(modal["participation_X_Y_Rz"])[j]
                e=modal["character"][j]["energy_X_Y_Rz_distortion"]
                modal_rows.append({"model":code,"mass_case":name,"rigid_reference_mode":i+1,
                    "matched_mode":int(j+1),"MAC_actual_mass":float(same["MAC"][i]),
                    "period_s":modal["periods_s"][j],"reference_period_s":ref["periods_s"][i],
                    "period_change_pct":100*(modal["periods_s"][j]/ref["periods_s"][i]-1),
                    "participation_X":float(p[0]),"participation_Y":float(p[1]),"participation_Rz_CM":float(p[2]),
                    "energy_X":e[0],"energy_Y":e[1],"energy_rotation_CM":e[2],"energy_distortion":e[3],
                    "geometry_rotation_fraction":geometry_rotation_fraction(vectors[j]),
                    "low_single_mode_MAC":bool(same["MAC"][i]<.9)})
            for i,j in enumerate(cross["indices"]):
                mass_mode_rows.append({"model":code,"mass_case":name,"uniform_reference_mode":i+1,
                    "matched_mode":int(j+1),"MAC_fixed_uniform_weight":float(cross["MAC"][i]),
                    "period_s":modal["periods_s"][j],"uniform_period_s":uniform["periods_s"][i],
                    "diagnostic_period_difference_pct":100*(modal["periods_s"][j]/uniform["periods_s"][i]-1),
                    "MAC_below_0p9":bool(cross["MAC"][i]<.9),
                    "weight_definition":"fixed uniform reference; shape similarity only"})
            # 근접 모드는 단일 벡터의 MAC만으로 판단하지 않는다.
            for indices in mode_clusters(ref["periods_s"],relative_gap=.03):
                if len(indices)<2: continue
                selected=np.array(same["indices"])[indices]
                similarity=subspace_similarity(np.array(ref["vectors"])[indices],vectors[selected],md)
                cluster_rows.append({"model":code,"mass_case":name,"comparison":"same_mass_rigid_reference",
                    "reference_modes":",".join(str(i+1) for i in indices),
                    "candidate_modes":",".join(str(i+1) for i in selected),
                    "min_cosine_squared":float(similarity["minimum_cosine_squared"]),
                    "max_principal_angle_deg":float(np.degrees(max(similarity["principal_angles_rad"]))),
                    "relative_period_gap_threshold":.03})
            energy=modal["character"][0]["energy_X_Y_Rz_distortion"]
            scalar={"model":code,"mass_case":name,"model_group":family,"T1_s":modal["periods_s"][0],
                "T1_change_from_R_pct":100*(modal["periods_s"][0]/ref["periods_s"][0]-1),
                "frequency1_Hz":1/modal["periods_s"][0],"J_about_CM":jc,
                "mode1_energy_X":energy[0],"mode1_energy_Y":energy[1],"mode1_energy_Rz_CM":energy[2],
                "mode1_energy_distortion":energy[3],"mode1_geometry_rotation_fraction":geometry_rotation_fraction(vectors[0]),
                "cum_participation_X":modal["cumulative_participation"][0],
                "cum_participation_Y":modal["cumulative_participation"][1],
                "cum_participation_Rz_CM":modal["cumulative_participation"][2]}
            for direction,d in (("X",0),("Y",1)):
                force=forces[f"{code}_{name}_{direction}"]
                fref=forces[f"R_{name}_{direction}"]
                max_share=0.;largest_location=None
                for row,refrow in zip(force["line_results"],fref["line_results"]):
                    assert (row["floor"],row["line_coord_m"])==(refrow["floor"],refrow["line_coord_m"])
                    delta=row["signed_share_pct"]-refrow["signed_share_pct"]
                    if abs(delta)>max_share:
                        max_share=abs(delta);largest_location=f"story{row['floor']}/line{row['line_coord_m']:+g}"
                    lines.append({"model":code,"mass_case":name,"direction":direction,**row,
                        "same_mass_R_difference_pp":delta})
                scalar[f"{direction}_max_line_share_difference_pp"]=max_share
                scalar[f"{direction}_max_line_difference_location"]=largest_location
                for floor in (1,2,3):
                    st=case[f"static_{direction}"][floor-1]
                    joint=np.array(st["joint_displacements_mm"])
                    jointmax=float(np.max(abs(joint[:,d])))
                    floors.append({"model":code,"mass_case":name,"direction":direction,"floor":floor,
                        "common_joint_max_mm":jointmax,"field_max_mm":st["max_directional_mm"],
                        "joint_distortion_RMS_mm":joint_distortion(joint),"field_distortion_RMS_mm":st["inplane_RMS_mm"]})
                    for index,((x,y),u) in enumerate(zip(XY,joint),1):
                        positions.append({"model":code,"mass_case":name,"direction":direction,"floor":floor,
                            "joint_index":index,"x_m":x,"y_m":y,"Ux_mm":float(u[0]),"Uy_mm":float(u[1]),
                            "along_loading_mm":float(u[d]),"cross_loading_mm":float(u[1-d])})
                    if floor==3:
                        rjoint=np.array(cases[f"R_{name}"][f"static_{direction}"][-1]["joint_displacements_mm"])
                        rmax=float(np.max(abs(rjoint[:,d])))
                        scalar[f"{direction}_roof_common_max_mm"]=jointmax
                        scalar[f"{direction}_roof_max_change_from_R_pct"]=100*(jointmax/rmax-1)
                        scalar[f"{direction}_roof_joint_RMS_mm"]=joint_distortion(joint)
                        scalar[f"{direction}_roof_field_RMS_mm"]=st["inplane_RMS_mm"]
            summary.append(scalar)
    # 같은 위치를 유지한 네 사례의 차분. 최대값 위치 변경과 구분한다.
    def add_interaction(rows,identity,value_fields):
        lookup={(r["model"],r["mass_case"],*(r[k] for k in identity)):r for r in rows}
        output=[]
        for r in rows:
            suffix=tuple(r[k] for k in identity)
            ref=lookup[("R",r["mass_case"],*suffix)]
            u=lookup[(r["model"],"U",*suffix)];ru=lookup[("R","U",*suffix)]
            for field,unit in value_fields:
                output.append({"model":r["model"],"mass_case":r["mass_case"],**{k:r[k] for k in identity},
                    "metric":field,"value_unit":"percent" if unit=="percentage_points" else unit,
                    "difference_unit":unit,"value":r[field],
                    "mass_effect_at_R":ref[field]-ru[field],"diaphragm_effect_same_mass":r[field]-ref[field],
                    "diaphragm_effect_at_U":u[field]-ru[field],
                    "interaction":r[field]-ref[field]-u[field]+ru[field]})
        return output
    point_interactions=add_interaction(positions,["direction","floor","joint_index"],[("along_loading_mm","mm"),("cross_loading_mm","mm")])
    line_interactions=add_interaction(lines,["direction","floor","line_coord_m"],[("signed_share_pct","percentage_points")])
    floor_interactions=add_interaction(floors,["direction","floor"],[("common_joint_max_mm","mm"),
        ("joint_distortion_RMS_mm","mm"),("field_distortion_RMS_mm","mm")])
    case_interactions=add_interaction(summary,[],[("T1_s","s")])
    for r in floor_interactions:
        r["extremum_switch_possible"]=r["metric"]=="common_joint_max_mm"
    # 관측된 메시 차이를 같은 위치의 효과와 비교한다. 모든 조건의 엄밀한 오차 상한은 아니다.
    mesh=read_csv(DIR15/"step15_mesh_check.csv")
    mesh12={(r["model"],r["direction"],int(r["floor"]),float(r["line_coord_m"])):
            abs(float(r["difference_percentage_points"])) for r in mesh if int(r["mesh"])==12}
    for r in lines:
        shift=mesh12.get((r["model"],r["direction"],r["floor"],r["line_coord_m"])) if r["mass_case"]=="C99" else None
        r["observed_mesh8_to12_shift_pp"]=shift
        r["effect_vs_observed_mesh_shift"]=("at_or_below_observed_shift" if abs(r["same_mass_R_difference_pp"])<=shift+1e-10
            else "above_observed_shift") if shift is not None else "not_checked_this_case"
    raw_tables={"case_catalog":catalog,"case_summary":summary,"same_mass_modes":modal_rows,
        "mass_variation_modes":mass_mode_rows,"near_mode_subspaces":cluster_rows,"floor_metrics":floors,
        "position_metrics":positions,"line_metrics":lines,"point_interactions":point_interactions,
        "line_interactions":line_interactions,"floor_interactions":floor_interactions,
        "case_interactions":case_interactions}
    for name,rows in raw_tables.items(): save_csv(OUT/f"step16_{name}.csv",rows)
    source_records=[{"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCES]
    metadata={"models":MODELS,"mass_cases":NAMES,"common_joint_xy_m":XY.tolist(),"common_modal_components":72,
        "assumed_RC_specifications":d14["specifications"],
        "same_mass_MAC_weight":"actual nodal mass of that mass case",
        "cross_mass_MAC_weight":"fixed uniform floor_mass/12 in each horizontal component",
        "kinematic_rotation_definition":"geometric-center rigid-motion fit; uniform reference; not kinetic energy",
        "RMS_definition":"common joint RMS is equal-weight 12 joint fit; field RMS uses original area weighting",
        "interaction_definition":"Q(D,E)-Q(R,E)-Q(D,U)+Q(R,U), fixed location where applicable",
        "modal_cluster_relative_gap":.03,"new_OpenSees_analyses":0,"source_files":source_records,
        "scope":"3-story assumed elastic steel moment frame, perfect attachment, in-plane-only diaphragm, fixed floor mass and gravity",
        "prior_mesh_and_validation":{p.parent.name+"/"+p.name:read_json(p) for p in SOURCES if p.name.endswith("checks.json")}}
    save_json(OUT/"step16_integrated_results.json",{"metadata":metadata,**raw_tables})
    after=preservation_manifest();changed=[p for p,h in before.items() if after.get(p)!=h]
    checks.update({"source_files_checked":len(SOURCES),"existing_files_checked":len(before),"changed":changed,
        "modal_same_mass_matches":len(modal_rows),"modal_cross_mass_matches":len(mass_mode_rows),
        "common_position_records":len(positions),"line_records":len(lines),"floor_records":len(floors),
        "new_OpenSees_analyses":0,
        "low_MAC_actual_mass_count":sum(r["low_single_mode_MAC"] for r in modal_rows),
        "low_MAC_fixed_weight_count":sum(r["MAC_fixed_uniform_weight"]<.9 for r in mass_mode_rows)})
    assert len(summary)==72 and len(positions)==5184 and len(lines)==1512 and len(floors)==432
    if changed: raise AssertionError(f"기존 파일 변경 {changed}")
    save_json(OUT/"verification_checks.json",checks)
    from verification.step16_reporting import run as reporting
    reporting()
    print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
