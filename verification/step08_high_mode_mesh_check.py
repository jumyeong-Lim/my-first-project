"""STEP08 고차 모드 계통의 4×4/8×8 메시 재현 검증."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verification.step08_validation import OUT, MD, modal, track


def run():
    old=json.loads((OUT/"validation_results.json").read_text(encoding="utf-8"))
    reference=modal(None,1)
    path8=track(reference,8)
    path4=old["path"]
    a,b=path4[-1],path8[-1]
    records=[]
    for n in range(9):
        va=np.asarray(a["vectors"][n]);vb=np.asarray(b["vectors"][n])
        mac=float(np.sum(MD*va*vb)**2)
        records.append({"rigid_origin_mode":n+1,"mode_4":a["mode_orders"][n],
            "mode_8":b["mode_orders"][n],"period_4_s":a["periods"][n],
            "period_8_s":b["periods"][n],
            "period_difference_pct":100*(b["periods"][n]/a["periods"][n]-1),
            "cross_mesh_MAC":mac,"origin_MAC_4":a["direct_rigid_mac"][n],
            "origin_MAC_8":b["direct_rigid_mac"][n]})
    return {"mesh_4_points":len(path4),"mesh_8_points":len(path8),
        "mesh_8_min_adjacent_MAC":min(min(row["adjacent_mac"]) for row in path8),
        "records":records}


def write(result):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"high_mode_mesh_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"high_mode_mesh_check.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.DictWriter(file,fieldnames=list(result["records"][0]))
        writer.writeheader();writer.writerows(result["records"])
    rows=result["records"]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar([r["rigid_origin_mode"] for r in rows],
                [abs(r["period_difference_pct"]) for r in rows])
    axes[1].bar([r["rigid_origin_mode"] for r in rows],
                [r["cross_mesh_MAC"] for r in rows])
    axes[0].set(xlabel="Rigid origin mode",ylabel="Abs period difference (%)",
                title="Tracked low-stiffness endpoint: 4 vs 8 mesh")
    axes[1].set(xlabel="Rigid origin mode",ylabel="Cross-mesh MAC",ylim=(0,1.05),
                title="Tracked mode-shape consistency")
    for ax in axes: ax.set_xticks(range(1,10));ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(OUT/"high_mode_mesh_check.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    result=run();write(result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
