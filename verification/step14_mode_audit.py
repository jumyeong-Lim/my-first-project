"""STEP14 결과에서 발견한 근접 모드와 고차 주기 차이를 추가 확인한다."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from verification.step14_material_based_diaphragm import OUT, SLABS, use_material, original
from verification.step12_shell_audit import save_json, save_csv


def run():
    results=json.loads((OUT/"step14_results.json").read_text(encoding="utf-8"))["results"]
    clusters=[]
    weights=original.mass_cases()["C99"]
    md=np.tile(np.repeat(original.P.floor_mass*weights,2),3)
    a=np.array(results["R_C99"]["modal"]["vectors"])[3:5]
    qa=np.linalg.qr((a*np.sqrt(md)).T)[0]
    for code in SLABS:
        case=results[f"{code}_C99"]
        idx=np.array(case["match_to_rigid"]["orders"])[3:5]-1
        b=np.array(case["modal"]["vectors"])[idx]
        qb=np.linalg.qr((b*np.sqrt(md)).T)[0]
        singular=np.linalg.svd(qa.T@qb,compute_uv=False)
        clusters.append({"model":code,"mass_case":"C99","rigid_modes":"4,5",
            "finite_modes":",".join(str(i+1) for i in idx),
            "min_principal_cosine_squared":float(min(singular**2)),
            "max_principal_angle_deg":float(np.degrees(np.arccos(np.clip(min(singular),-1,1))))})
    save_csv(OUT/"near_mode_subspace_check.csv",clusters)
    # C99 검증에 더해 가장 큰 고차 주기 차이가 나온 X99를 12분할로 재해석한다.
    rows=[]
    for name in ("C99","X99"):
        weights=original.mass_cases()[name]
        md=np.tile(np.repeat(original.P.floor_mass*weights,2),3)
        for code in SLABS:
            if name=="C99":
                fine=json.loads((OUT/f"mesh_{code}_C99_n12.json").read_text(encoding="utf-8"))["modal"]
            else:
                print("high-mode mesh",code,name,flush=True)
                with use_material(code): fine=original.modal(code,weights,12)
                save_json(OUT/f"modal_{code}_X99_n12.json",fine)
            case=results[f"{code}_{name}"]
            idx=np.array(case["match_to_rigid"]["orders"])-1
            a=np.array(case["modal"]["vectors"])[idx]
            b=np.array(fine["vectors"])
            scores=((a*md)@b.T)**2
            rr,cc=linear_sum_assignment(-scores)
            cc=cc[np.argsort(rr)]
            for i,j in enumerate(cc):
                rows.append({"model":code,"mass_case":name,"rigid_mode":i+1,
                    "mesh8_mode":int(idx[i]+1),"mesh12_mode":int(j+1),"MAC_8_to_12":float(scores[i,j]),
                    "period8_s":case["modal"]["periods_s"][idx[i]],"period12_s":fine["periods_s"][j],
                    "period_change_pct":100*(fine["periods_s"][j]/case["modal"]["periods_s"][idx[i]]-1)})
    save_csv(OUT/"higher_mode_mesh_check.csv",rows)
    md=np.tile(np.repeat(original.P.floor_mass*original.mass_cases()["C99"],2),3)
    fig=plt.figure(figsize=(13,8))
    for col,code in enumerate(("R","RC150_a30","RC150_a100")):
        case=results[f"{code}_C99"]
        for row,i in enumerate((3,4)):
            j=i if code=="R" else case["match_to_rigid"]["orders"][i]-1
            vector=np.array(case["modal"]["vectors"])[j]
            rigid=np.array(results["R_C99"]["modal"]["vectors"])[i]
            if np.dot(vector*md,rigid)<0: vector=-vector
            vector=vector.reshape(3,12,2)
            vector*=1.5/np.max(np.linalg.norm(vector,axis=2))
            ax=fig.add_subplot(2,3,row*3+col+1,projection="3d")
            for floor in range(3):
                xy=original.XY;deformed=xy+vector[floor];z=np.full(12,(floor+1)*4.)
                for indices in ([list(range(k,k+4)) for k in (0,4,8)]+[list(range(k,12,4)) for k in range(4)]):
                    ax.plot(xy[indices,0],xy[indices,1],z[indices],color="gray",alpha=.3,lw=.7)
                    ax.plot(deformed[indices,0],deformed[indices,1],z[indices],color="#3274a1",lw=1)
            ax.set(title=f"{code}: mode {j+1}, T={case['modal']['periods_s'][j]:.4f}s",
                   xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)")
            ax.set_box_aspect((18,12,12));ax.view_init(elev=23,azim=-65)
    fig.suptitle("C99 near modes 4/5: horizontal shapes only; each normalized to max 1.5 m")
    fig.tight_layout();fig.savefig(OUT/"step14_near_mode_shapes.png",dpi=180);plt.close(fig)
    save_json(OUT/"mode_audit_checks.json",{
        "minimum_cluster_cosine_squared":min(r["min_principal_cosine_squared"] for r in clusters),
        "minimum_mesh_MAC":min(r["MAC_8_to_12"] for r in rows),
        "maximum_higher_mode_mesh_period_change_pct":max(abs(r["period_change_pct"]) for r in rows)})
    print("STEP14 mode audit completed",flush=True)


if __name__ == "__main__": run()
