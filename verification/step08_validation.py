"""STEP08 보강: 층별 변형, 메시 수렴, 강성 경로의 모드 추적과 참여질량.

바닥판 형식 미정, 완전 부착을 사용자가 선택했다. 유효강성은 매개변수이며
실제 바닥판의 인증된 강성으로 해석하지 않는다. 질량은 원래 골조 절점에 유지한다.
"""
from __future__ import annotations
import json
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verification.step08_diaphragm_inplane_sensitivity import (
    PARAMETERS as P, OUTPUT_DIR, SHELL_E_KN_M2, build_case, build_finite_case,
    horizontal_mass_vector, joint_horizontal_array,
)
OUT = OUTPUT_DIR / "validation"
MD = horizontal_mass_vector()


def floor_geometry(xy):
    """강체 평면 병진 두 개와 회전 한 개의 기하변환이다."""
    a = np.zeros((2*len(xy), 3))
    a[0::2, 0], a[1::2, 1] = 1, 1
    a[0::2, 2], a[1::2, 2] = -xy[:, 1], xy[:, 0]
    return a


def floor_metrics(xy, motion, weights=None):
    """병진·회전을 제거한 잔차를 계산한다. 메시 표본에는 면적가중치를 쓴다."""
    if weights is None:
        weights = np.ones(len(xy))
    weights = np.asarray(weights)/np.sum(weights)
    a = floor_geometry(xy)
    root = np.sqrt(np.repeat(weights, 2))
    fit = np.linalg.lstsq(a*root[:, None], motion.ravel()*root, rcond=None)[0]
    residual = motion-(a@fit).reshape(-1, 2)
    rms = np.sqrt(np.sum(weights*np.sum(residual**2, axis=1)))
    total = np.sqrt(np.sum(weights*np.sum(motion**2, axis=1)))
    return {"rigid_motion": fit.tolist(), "distortion_rms": float(rms),
            "distortion_ratio": float(rms/total) if total > 1e-20 else 0.0,
            "range_xy": np.ptp(motion, axis=0).tolist()}


def prepare(e, mesh):
    model = build_case("rigid") if e is None else build_finite_case(
        "flexible", subdivisions=mesh, shell_e=e)
    # 세분 메시에는 희소 행렬 풀이를 사용하되 기존 10단계 중력 절차를 유지한다.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nodes in model["levels"][1:]:
        for tag in nodes:
            ops.load(tag, 0, 0, -P.floor_weight/len(nodes), 0, 0, 0)
    configure_sparse(.1)
    if ops.analyze(10):
        raise RuntimeError("중력해석 실패")
    ops.reactions()
    reaction = sum(ops.nodeReaction(t, 3) for t in model["levels"][0])
    if abs(reaction-P.floor_weight*P.stories)/(P.floor_weight*P.stories) > 1e-8:
        raise AssertionError("중력 평형 실패")
    ops.loadConst("-time", 0)
    return model


def configure_sparse(step):
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-12, 40)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", step)
    ops.analysis("Static")


def modal(e, mesh, count=48, solver="-genBandArpack"):
    model = prepare(e, mesh)
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    if e is None:
        count, solver = 9, "-fullGenLapack"
    values = np.asarray(ops.eigen(solver, count))
    if np.min(values) <= 0 or not np.all(np.isfinite(values)):
        raise AssertionError("양의 유한 고유치 확인 실패")
    vectors = np.array([joint_horizontal_array(model,
        lambda t, d: ops.nodeEigenvector(t, n+1, d)).ravel() for n in range(count)])
    vectors /= np.sqrt(np.sum(vectors**2*MD, axis=1))[:, None]
    xy = model["xy"]
    influence = np.tile(floor_geometry(xy), (P.stories, 1))
    denominator = np.sum(MD[:, None]*influence**2, axis=0)
    participation = ((vectors*MD)@influence)**2/denominator
    orthogonality = np.max(np.abs((vectors*MD)@vectors.T-np.eye(count)))
    if orthogonality > 1e-6:
        raise AssertionError(f"질량 직교성 실패: {orthogonality}")
    properties = ops.modalProperties("-return")
    np.testing.assert_allclose(properties["totalFreeMass"][:2], P.floor_mass*P.stories, rtol=1e-9)
    np.testing.assert_allclose(properties["centerOfMass"][:2], 0, atol=1e-9)
    np.testing.assert_allclose(properties["totalFreeMass"][5], denominator[2], rtol=1e-9)
    ratios = [[floor_metrics(xy, floor)["distortion_ratio"] for floor in v.reshape(3, -1, 2)]
              for v in vectors]
    return {"E": e, "mesh": mesh, "periods": (2*np.pi/np.sqrt(values)).tolist(),
            "vectors": vectors.tolist(), "participation_X_Y_Rz": participation.tolist(),
            "cumulative_participation": participation.sum(axis=0).tolist(),
            "floor_modal_distortion": ratios, "mass_orthogonality_error": float(orthogonality),
            "polar_mass_inertia_total": float(denominator[2])}


def match(a, b):
    scores = (np.asarray(a)*MD)@np.asarray(b).T
    scores **= 2
    rows, cols = linear_sum_assignment(-scores)
    order = np.argsort(rows)
    cols = cols[order]
    chosen = scores[np.arange(len(cols)), cols]
    second = np.array([np.max(np.delete(scores[i], j)) for i, j in enumerate(cols)])
    return cols, chosen, chosen-second


def static(e, mesh, direction=0, loading="roof"):
    model = prepare(e, mesh)
    floors = model.get("shell_levels", model["levels"][1:])
    initial = [np.array([[ops.nodeDisp(t, d) for d in (1, 2)] for t in nodes]) for nodes in floors]
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    loaded = model["levels"][1:] if loading == "uniform_floors" else [model["levels"][-1]]
    for nodes in loaded:
        for tag in nodes:
            load = [0.0]*6
            load[direction] = 12.0/(len(loaded)*len(nodes))
            ops.load(tag, *load)
    configure_sparse(1)
    if ops.analyze(1):
        raise RuntimeError("층별 정적해석 실패")
    output = []
    previous = 0.0
    for level, (nodes, base) in enumerate(zip(floors, initial), 1):
        xy = np.array([model["coords"][t][:2] for t in nodes])
        motion = np.array([[ops.nodeDisp(t, d) for d in (1, 2)] for t in nodes])-base
        weights = np.ones(len(nodes))
        # 직사각형 균일 격자의 경계는 절반, 모서리는 1/4의 지배면적이다.
        if e is not None:
            weights[np.isclose(np.abs(xy[:, 0]), P.lengths[0]/2)] *= .5
            weights[np.isclose(np.abs(xy[:, 1]), P.lengths[1]/2)] *= .5
        field = floor_metrics(xy, motion, weights)
        joint_indices = [nodes.index(t) for t in model["levels"][level]]
        joint = floor_metrics(model["xy"], motion[joint_indices])
        average = field["rigid_motion"][direction]
        drift = average-previous
        previous = average
        output.append({"floor": level, "field": field, "joints": joint,
                       "mean_translation_mm": average*1000, "interstory_mm": drift*1000,
                       "field_distortion_over_drift": field["distortion_rms"]/abs(drift) if abs(drift)>1e-15 else None,
                       "xy": xy.tolist(), "motion": motion.tolist()})
    return output


def track(reference, mesh=4):
    """강성이 큰 쪽부터 인접 강성의 MAC으로 추적하고 낮은 MAC 구간을 세분한다."""
    cache = {}
    def sample(e):
        key = f"{e:.12g}"
        if key not in cache:
            cache[key] = modal(e, mesh)
        return cache[key]
    rows = []
    previous_vectors = np.array(reference["vectors"])
    def advance(high, low, vectors, depth=0):
        current = sample(low)
        indices, mac, margin = match(vectors, current["vectors"])
        if high is not None and np.min(mac)<.95 and depth<5:
            midpoint = np.sqrt(high*low)
            intermediate = advance(high, midpoint, vectors, depth+1)
            return advance(midpoint, low, intermediate, depth+1)
        selected = np.array(current["vectors"])[indices]
        direct = np.sum(selected*MD*np.array(reference["vectors"]), axis=1)**2
        rows.append({"E": low, "mode_orders": (indices+1).tolist(),
            "periods": np.array(current["periods"])[indices].tolist(),
            "adjacent_mac": mac.tolist(), "assignment_margin": margin.tolist(),
            "direct_rigid_mac": direct.tolist(),
            "participation": np.array(current["participation_X_Y_Rz"])[indices].tolist(),
            "floor_distortion": np.array(current["floor_modal_distortion"])[indices].tolist(),
            "vectors": selected.tolist(),
            "ambiguous": bool(np.min(mac)<.95 or np.min(margin)<.05)})
        return selected
    high = None
    for e in np.geomspace(2e10, 2e4, 25):
        previous_vectors = advance(high, float(e), previous_vectors)
        high = float(e)
    return rows


def run():
    reference = modal(None, 1)
    mesh_results, comparisons = {}, []
    for kind in ("semirigid", "flexible"):
        for mesh in (1, 2, 4, 8):
            e = SHELL_E_KN_M2[kind]
            print(f"mesh: {kind}, {mesh} x {mesh} per bay", flush=True)
            result = modal(e, mesh)
            result["static"] = {f"{load}_{axis}": static(e, mesh, axis, load)
                for load in ("roof", "uniform_floors") for axis in (0, 1)}
            mesh_results[f"{kind}_{mesh}"] = result
        for coarse, fine in ((1, 2), (2, 4), (4, 8)):
            a, b = mesh_results[f"{kind}_{coarse}"], mesh_results[f"{kind}_{fine}"]
            selected, _, _ = match(reference["vectors"], a["vectors"])
            indices, mac, _ = match(np.array(a["vectors"])[selected], b["vectors"])
            change = np.array(b["periods"])[indices]/np.array(a["periods"])[selected]-1
            field_changes = []
            for key in a["static"]:
                for fa, fb in zip(a["static"][key], b["static"][key]):
                    field_changes.append(abs(fb["field"]["distortion_rms"]/fa["field"]["distortion_rms"]-1))
            comparisons.append({"kind":kind,"coarse":coarse,"fine":fine,
                "period_change_pct": (100*change).tolist(), "mesh_mac":mac.tolist(),
                "max_field_rms_change_pct":100*max(field_changes)})
    print("tracking stiffness path", flush=True)
    path = track(reference)
    # 추출 개수를 늘려 추적 대상 모드 및 참여질량이 누락되지 않았는지 확인한다.
    # 관성 자유도 수에 가까운 모드 요청은 ARPACK의 부분공간 생성이 실패할 수 있어
    # 이 교차검증 한 번은 전체 행렬 풀이로 수행한다.
    extended = modal(2e4, 4, count=66, solver="-fullGenLapack")
    endpoint = np.array(path[-1]["vectors"])
    indices, mac, _ = match(endpoint, extended["vectors"])
    count_check = {"48_to_66_mac":mac.tolist(), "orders_66":(indices+1).tolist(),
        "cumulative_66":extended["cumulative_participation"],
        "cumulative_48":mesh_results["flexible_4"]["cumulative_participation"]}
    result = {"reference":reference,"meshes":mesh_results,"mesh_comparisons":comparisons,
            "tracking_mesh":4,"path":path,"mode_count_check":count_check,
            "assumptions":{"floor_type":"미정", "connection":"완전 부착",
            "mass":"기존 12개 절점/층에 고정; 셸과 추가 절점은 무질량",
            "mesh":"보도 분할하여 셸 경계와 공유; 막 내부 면외 자유도 제거"}}
    refine_static(result)
    return result


def refine_static(data):
    """주기와 별개로 작은 면내변형의 8→16 메시 민감도를 추가 확인한다."""
    checks = {}
    for kind in ("semirigid", "flexible"):
        print(f"static refinement: {kind}, 16 x 16", flush=True)
        cases = {f"{load}_{axis}": static(SHELL_E_KN_M2[kind],16,axis,load)
                 for load in ("roof","uniform_floors") for axis in (0,1)}
        differences = []
        for key, floors in cases.items():
            for a,b in zip(data["meshes"][f"{kind}_8"]["static"][key], floors):
                differences.append(100*abs(b["field"]["distortion_rms"]/a["field"]["distortion_rms"]-1))
        checks[kind] = {"static":cases, "max_field_rms_change_pct":max(differences)}
    data["static_8_to_16"] = checks


def write(data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"validation_results.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"all_floor_metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["case","loading_axis","floor","field_RMS_mm","joint_RMS_mm",
                         "mean_translation_mm","interstory_mm","field_RMS_over_drift"])
        static_sources = dict(data["meshes"])
        static_sources.update({f"{kind}_16":case for kind,case in data.get("static_8_to_16",{}).items()})
        for key,case in static_sources.items():
            for loading,floors in case["static"].items():
                for f in floors:
                    writer.writerow([key,loading,f["floor"],1000*f["field"]["distortion_rms"],
                        1000*f["joints"]["distortion_rms"],f["mean_translation_mm"],
                        f["interstory_mm"],f["field_distortion_over_drift"]])
    with (OUT/"tracked_modes.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["E","rigid_origin_mode","actual_mode","period_s","adjacent_MAC",
                         "rigid_origin_MAC","assignment_margin","mass_X","mass_Y","inertia_Rz"])
        for row in data["path"]:
            for n in range(9):
                writer.writerow([row["E"],n+1,row["mode_orders"][n],row["periods"][n],
                    row["adjacent_mac"][n],row["direct_rigid_mac"][n],row["assignment_margin"][n],
                    *row["participation"][n]])
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    for kind in ("semirigid","flexible"):
        entries=[v for v in data["mesh_comparisons"] if v["kind"]==kind]
        labels=[f"{v['coarse']}→{v['fine']}" for v in entries]
        axes[0].plot(labels,[max(abs(np.array(v["period_change_pct"]))) for v in entries],"o-",label=kind)
        distortion_changes=[v["max_field_rms_change_pct"] for v in entries]
        if "static_8_to_16" in data:
            labels=labels+["8→16"]
            distortion_changes.append(data["static_8_to_16"][kind]["max_field_rms_change_pct"])
        axes[1].plot(labels,distortion_changes,"o-",label=kind)
    axes[0].set(ylabel="Max matched period change (%)",title="Mesh refinement: nine reference modes")
    axes[1].set(ylabel="Max field distortion RMS change (%)",title="All floors and both load patterns")
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT/"mesh_convergence.png",dpi=170); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4.5))
    for kind in ("semirigid","flexible"):
        for axis in (0,1):
            floors=data["meshes"][f"{kind}_8"]["static"][f"roof_{axis}"]
            axes[axis].plot([v["field"]["distortion_rms"]*1000 for v in floors],[1,2,3],"o-",label=kind)
    for i,ax in enumerate(axes):
        ax.set(xlabel="Area-weighted in-plane RMS (mm)",ylabel="Floor",yticks=[1,2,3],title=f"Roof {'XY'[i]} load; 8×8 mesh per bay")
        ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/"all_floor_distortion.png",dpi=170);plt.close(fig)
    path=data["path"]; e=[r["E"] for r in path]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for n in (1,4,7):
        label=f"Rigid reference mode {n+1}"
        axes[0,0].semilogx(e,[r["periods"][n] for r in path],label=label)
        axes[0,1].semilogx(e,[r["mode_orders"][n] for r in path],label=label)
        axes[1,0].semilogx(e,[r["direct_rigid_mac"][n] for r in path],label=label)
        axes[1,1].semilogx(e,[r["adjacent_mac"][n] for r in path],label=label)
    for ax,title,ylabel in zip(axes.flat,["Tracked periods","Changing eigenmode order","Similarity to rigid origin","Similarity to previous stiffness"],["Period (s)","Mode order","MAC","Adjacent MAC"]):
        ax.set(title=title,ylabel=ylabel,xlabel="Shell E (kN/m²)");ax.invert_xaxis();ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/"stiffness_mode_tracking.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,n in zip(axes,(1,4,7)):
        for d,label in enumerate(("X","Y","Rz")):
            ax.semilogx(e,[100*r["participation"][n][d] for r in path],label=label)
        ax.set(title=f"Tracked reference mode {n+1}",xlabel="Shell E (kN/m²)",ylabel="Effective mass / inertia (%)")
        ax.invert_xaxis();ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/"tracked_participation.png",dpi=170);plt.close(fig)
    # 저 MAC 모드의 실제 질량절점 평면형상과 층별 왜곡을 추적 전후로 비교한다.
    xy=np.array([(i*P.bay_length-P.lengths[0]/2,j*P.bay_length-P.lengths[1]/2)
                 for j in range(P.bays_y+1) for i in range(P.bays_x+1)])
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for col,n in enumerate((1,4,7)):
        origin=np.array(data["reference"]["vectors"][n])
        endpoint=np.array(path[-1]["vectors"][n])
        if np.sum(origin*MD*endpoint)<0: endpoint=-endpoint
        for row,vector in enumerate((origin,endpoint)):
            motion=vector[-len(xy)*2:].reshape(-1,2).copy()
            motion*=1.5/np.max(np.linalg.norm(motion,axis=1))
            moved=xy+motion
            ax=axes[row,col]
            for j in range(P.bays_y+1):
                idx=np.arange(j*(P.bays_x+1),(j+1)*(P.bays_x+1))
                ax.plot(*xy[idx].T,"--",color=".7");ax.plot(*moved[idx].T,color="#286ba5")
            for i in range(P.bays_x+1):
                ax.plot(*xy[i::P.bays_x+1].T,"--",color=".7")
                ax.plot(*moved[i::P.bays_x+1].T,color="#286ba5")
            title=f"Rigid origin {n+1}" if row==0 else f"Tracked mode {path[-1]['mode_orders'][n]}, origin MAC={path[-1]['direct_rigid_mac'][n]:.4g}"
            ax.set(title=title,xlabel="X (m)",ylabel="Y (m)",xlim=(-11,11),ylim=(-8,8));ax.set_aspect("equal")
    fig.suptitle("Tracked torsional branches: roof frame-joint samples\nEach mode independently amplified; gray original, blue modal shape")
    fig.subplots_adjust(top=.85,bottom=.08,hspace=.55,wspace=.3)
    fig.savefig(OUT/"tracked_low_mac_shapes.png",dpi=170);plt.close(fig)


if __name__ == "__main__":
    data=run()
    write(data)
    print(json.dumps(data["mesh_comparisons"],indent=2))
    print("path points",len(data["path"]),"ambiguous",sum(r["ambiguous"] for r in data["path"]))
    print(data["mode_count_check"])
