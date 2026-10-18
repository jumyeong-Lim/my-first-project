"""STEP10 편심 셸 모델의 메시 및 하중 크기 민감도를 별도 점검한다."""
import json
import numpy as np
from verification.step10_diaphragm_mass_interaction import OUT,mass_cases,modal,static


def run():
    data=json.loads((OUT/"step10_results.json").read_text(encoding="utf-8"))
    results=data["results"];weights=mass_cases();checks=[]
    for d in ("S","F"):
        for name in ("C99","C50_Jlo"):
            print("mesh check",d,name,flush=True)
            base=results[f"{d}_{name}"]
            coarse=modal(d,weights[name],mesh=4)
            record={"diaphragm":d,"mass_case":name,
                "first_period_4_to_8_pct":100*(base["modal"]["periods_s"][0]/coarse["periods_s"][0]-1),
                "static":{}}
            for direction,load in enumerate(("X","Y")):
                fine=static(d,weights[name],direction,mesh=16)
                old=base[f"static_{load}"]
                record["static"][load]={
                    "max_floor_peak_8_to_16_pct":max(100*abs(b["max_directional_mm"]/a["max_directional_mm"]-1) for a,b in zip(old,fine)),
                    "max_floor_RMS_8_to_16_pct":max(100*abs(b["inplane_RMS_mm"]/a["inplane_RMS_mm"]-1) for a,b in zip(old,fine)),
                    "floors_16":[{k:f[k] for k in ("floor","max_directional_mm","inplane_RMS_mm")} for f in fine]}
            checks.append(record)
    # 동일 기준 모델에서 하중 절반의 응답이 절반인지 확인해 단위응답 해석 범위를 점검한다.
    half=static("F",weights["C99"],0,mesh=8,acceleration=.5)
    full=results["F_C99"]["static_X"]
    scale_error=max(np.max(abs(2*np.asarray(a["joint_displacements_mm"])-np.asarray(b["joint_displacements_mm"]))) /
                    np.max(abs(np.asarray(b["joint_displacements_mm"]))) for a,b in zip(half,full))
    rigid_control_error=max(np.max(abs(np.asarray(results[f"R_{name}"][f"static_{load}"][floor]["joint_displacements_mm"])-
        np.asarray(results["R_C50"][f"static_{load}"][floor]["joint_displacements_mm"])))
        for name in ("C50_Jlo","C50_Jhi") for load in ("X","Y") for floor in range(3))
    result={"mesh_checks":checks,"half_load_relative_scale_error":float(scale_error),
            "same_center_rigid_static_max_difference_mm":float(rigid_control_error)}
    (OUT/"step10_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":run()
