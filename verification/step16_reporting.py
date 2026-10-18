"""STEP16 통합 결과의 과학 그래프와 한국어 최종 결론을 생성한다."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verification.step16_integrated_elastic_conclusion import (
    OUT, ROOT, MODELS, NAMES, read_json, save_csv,
)

COLORS={"R":"#333333","S":"#377eb8","F":"#e07a37","RC150_a10":"#9853a5",
        "RC150_a30":"#3c9c70","RC150_a100":"#b69a24"}


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers)]+
        ["| "+" | ".join(str(v) for v in row)+" |" for row in rows])


def finish(fig,name,legend_axes=None):
    if legend_axes is not None:
        handles,labels=legend_axes.get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",ncol=6,fontsize=8)
        fig.tight_layout(rect=(0,0,1,.93))
    else: fig.tight_layout()
    fig.savefig(OUT/name,dpi=180);plt.close(fig)


def plots(data,envelopes):
    summary=data["case_summary"];lookup={(r["model"],r["mass_case"]):r for r in summary}
    modes=data["same_mass_modes"]
    fields=("max_T1_change_pct","max_X_roof_change_pct","max_Y_roof_change_pct","max_line_share_change_pp")
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,title,unit in zip(axes.ravel(),fields,
            ("Sorted first period","Roof X displacement: common joints","Roof Y displacement: common joints","Frame-line shear share"),
            ("Change from same-mass R (%)",)*3+("Change from same-mass R (percentage points)",)):
        values=[r[key] for r in envelopes]
        ax.bar([r["model"] for r in envelopes],values,color=[COLORS[r["model"]] for r in envelopes])
        for i,v in enumerate(values): ax.text(i,v+.025*max(values),f"{v:.3f}",ha="center",fontsize=8)
        ax.set(title=title,ylabel=unit,ylim=(0,max(values)*1.18));ax.tick_params(axis="x",rotation=25)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Maximum absolute difference across 12 mass layouts (not all extrema occur in one case)")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/"step16_effect_overview.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,prefix in enumerate(("X","Y","C")):
        names=["U",prefix+"10",prefix+"50",prefix+"99"]
        for code in MODELS:
            values=[lookup[code,n] for n in names]
            axes[0,col].plot([0,.1,.5,.99],[v["T1_s"] for v in values],"o-",label=code,color=COLORS[code],ms=4)
            axes[1,col].plot([0,.1,.5,.99],[max(v["X_max_line_share_difference_pp"],v["Y_max_line_share_difference_pp"])
                            for v in values],"o-",label=code,color=COLORS[code],ms=4)
        axes[0,col].set(title=f"{prefix} mass-layout sequence",ylabel="Sorted first period (s)")
        axes[1,col].set(ylabel="Max frame-share difference from R (pp)",xlabel="Mass concentration mixing coefficient")
        for ax in axes[:,col]: ax.grid(alpha=.2)
    finish(fig,"step16_mass_eccentricity_trends.png",axes[0,0])
    # 동일 중심: 질량관성의 동적 영향과 국부 하중 위치의 정적 영향을 구별한다.
    names=["C50","C50_Jlo","C50_Jhi"];fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].bar(names,[lookup["R",n]["J_about_CM"] for n in names],color="#777777")
    axes[0,0].set(ylabel="J about mass center (kN s^2 m)",title="Same mass and center; different inertia")
    for code in MODELS:
        for ax,key,title,ylabel in ((axes[0,1],"T1_s","Dynamic: sorted first period","Period (s)"),
                (axes[1,0],"X_roof_common_max_mm","Static: unit X acceleration","Roof max |Ux| (mm)"),
                (axes[1,1],"Y_roof_common_max_mm","Static: unit Y acceleration","Roof max |Uy| (mm)")):
            ax.plot(names,[lookup[code,n][key] for n in names],"o-",color=COLORS[code],label=code,ms=4)
            ax.set(title=title,ylabel=ylabel);ax.grid(alpha=.2)
    finish(fig,"step16_same_center_controls.png",axes[0,1])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    names=["U","C10","C50","C99"]
    for code in ("R","F","RC150_a30"):
        for ax,key in zip(axes,("mode1_energy_Rz_CM","mode1_geometry_rotation_fraction")):
            ax.plot([0,.1,.5,.99],[100*lookup[code,n][key] for n in names],"o-",color=COLORS[code],label=code)
    axes[0].set(title="Actual mass: rotation energy about CM",ylabel="First-mode energy fraction (%)")
    axes[1].set(title="Fixed geometry: rotation displacement component",ylabel="First-mode kinematic fraction (%)")
    for ax in axes: ax.set_xlabel("Corner mass concentration mixing coefficient");ax.grid(alpha=.2)
    finish(fig,"step16_rotation_definitions.png",axes[0])
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    finite=MODELS[1:]
    counts=[sum(r["MAC_actual_mass"]<.9 for r in modes if r["model"]==c) for c in finite]
    maxima=[max(abs(r["period_change_pct"]) for r in modes if r["model"]==c and r["MAC_actual_mass"]>=.9) for c in finite]
    for ax,values,title,label in ((axes[0],counts,"Weak single-mode matches out of 108 per model","Count with MAC < 0.9"),
            (axes[1],maxima,"Period changes for matches with MAC >= 0.9","Max absolute matched-period change (%)")):
        ax.bar(finite,values,color=[COLORS[c] for c in finite]);ax.set(title=title,ylabel=label)
        for i,v in enumerate(values):ax.text(i,v+.025*max(values),f"{v:.2f}" if ax==axes[1] else str(v),ha="center",fontsize=8)
        ax.set_ylim(0,max(values)*1.15);ax.tick_params(axis="x",rotation=25);ax.grid(axis="y",alpha=.2)
    finish(fig,"step16_modal_matching_reliability.png")
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    names=[n for n in NAMES if n!="U"]
    matrices=[]
    matrices.append(np.array([[abs(next(r["interaction"] for r in data["case_interactions"] if r["model"]==c and r["mass_case"]==n)) for n in names] for c in finite]))
    for key in ("point_interactions","line_interactions"):
        matrices.append(np.array([[max(abs(r["interaction"]) for r in data[key] if r["model"]==c and r["mass_case"]==n
            and (key=="line_interactions" or r["metric"]=="along_loading_mm")) for n in names] for c in finite]))
    for ax,m,title,unit in zip(axes,matrices,("Sorted T1 interaction","Max pointwise displacement interaction","Max line-share interaction"),("s","mm","pp")):
        im=ax.imshow(m,cmap="YlOrRd",aspect="auto",vmin=0)
        ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(5),yticklabels=finite,title=title)
        ax.tick_params(axis="x",rotation=75)
        fig.colorbar(im,ax=ax,label=f"Absolute interaction ({unit})",shrink=.8)
    finish(fig,"step16_interaction_overview.png")
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,d in enumerate(("X","Y")):
        for col,codes in enumerate((("S","F"),("RC150_a10","RC150_a30","RC150_a100"))):
            ax=axes[row,col]
            for code in codes:
                records=[lookup[code,n] for n in NAMES]
                ax.scatter([r[f"{d}_roof_joint_RMS_mm"] for r in records],
                    [r[f"{d}_max_line_share_difference_pp"] for r in records],label=code,color=COLORS[code],s=32)
                r=lookup[code,"C99"]
                ax.annotate("C99",(r[f"{d}_roof_joint_RMS_mm"],r[f"{d}_max_line_share_difference_pp"]),fontsize=7,xytext=(3,3),textcoords="offset points")
            ax.set(title=f"{d}: "+("numerical stiffness cases" if col==0 else "assumed RC cases"),
                xlabel="Roof common-joint distortion RMS (mm)",ylabel="Max line-share difference from R (pp)")
            ax.legend(fontsize=8);ax.grid(alpha=.2)
    finish(fig,"step16_deformation_force_distribution.png")


def run():
    data=read_json(OUT/"step16_integrated_results.json");checks=read_json(OUT/"verification_checks.json")
    summary=data["case_summary"];lookup={(r["model"],r["mass_case"]):r for r in summary}
    envelopes=[]
    for code in MODELS[1:]:
        rows=[r for r in summary if r["model"]==code]
        envelopes.append({"model":code,"max_T1_change_pct":max(abs(r["T1_change_from_R_pct"]) for r in rows),
            "max_X_roof_change_pct":max(abs(r["X_roof_max_change_from_R_pct"]) for r in rows),
            "max_Y_roof_change_pct":max(abs(r["Y_roof_max_change_from_R_pct"]) for r in rows),
            "max_line_share_change_pp":max(max(r["X_max_line_share_difference_pp"],r["Y_max_line_share_difference_pp"]) for r in rows)})
    save_csv(OUT/"step16_effect_envelopes.csv",envelopes)
    plots(data,envelopes)
    effect_table=table(["모델","최대 T1 차이 (%)","최대 X변위 차이 (%)","최대 Y변위 차이 (%)","최대 선분담률 차이 (%p)"],
        [[r["model"]]+[f"{r[k]:.5f}" for k in ("max_T1_change_pct","max_X_roof_change_pct","max_Y_roof_change_pct","max_line_share_change_pp")] for r in envelopes])
    controls=table(["R 질량배치","Jc (kN·s²·m)","T1 (s)","지붕 X (mm)","지붕 Y (mm)"],
        [[n,f"{lookup['R',n]['J_about_CM']:.5f}",f"{lookup['R',n]['T1_s']:.9f}",
          f"{lookup['R',n]['X_roof_common_max_mm']:.9f}",f"{lookup['R',n]['Y_roof_common_max_mm']:.9f}"]
         for n in ("C50","C50_Jlo","C50_Jhi")])
    modes=data["same_mass_modes"]
    reliable=[r for r in modes if r["model"]=="RC150_a10" and r["mass_case"]=="X99" and r["rigid_reference_mode"]==9][0]
    weak=[r for r in modes if r["MAC_actual_mass"]<.9]
    modal_table=table(["모델","MAC<0.9 대응 수 /108","MAC≥0.9 대응의 최대 주기 차이 (%)"],
        [[c,sum(r["MAC_actual_mass"]<.9 for r in modes if r["model"]==c),
          f"{max(abs(r['period_change_pct']) for r in modes if r['model']==c and r['MAC_actual_mass']>=.9):.4f}"] for c in MODELS[1:]])
    cluster=[r for r in data["near_mode_subspaces"] if r["mass_case"]=="C99" and r["model"] in ("S","RC150_a30","RC150_a100") and r["reference_modes"]=="4,5"]
    cluster_table=table(["모델","C99 4·5차 묶음 최소 cos²"],[[r["model"],f"{r['min_cosine_squared']:.6f}"] for r in cluster])
    rc50=lookup["R","C50"];rc99=lookup["R","C99"]
    first_nine=data["metadata"]["source_files"][0]
    primary14=read_json(ROOT/first_nine["path"])
    first9=np.array(primary14["results"]["F_U"]["modal"]["participation_X_Y_Rz"])[:9].sum(axis=0)
    meshcheck=data["metadata"]["prior_mesh_and_validation"]
    m15=meshcheck["step15_frame_shear_distribution/verification_checks.json"]
    checked_lines=[r for r in data["line_metrics"] if r["effect_vs_observed_mesh_shift"]=="at_or_below_observed_shift"]
    critical=table(["검토 항목","현재 판단"],[
        ["작은 패널 구성식·질량·좌표·평형","수치 구현 검증 완료"],
        ["저차 주기·정적 변위·선별 힘 분배","72조합의 통제 비교 완료"],
        ["고차 형상 대응","낮은 MAC에 개별 수치 단정 금지; 근접 모드 묶음 병행"],
        ["RC 후보 물성","E·nu·두께 가정 + 문헌 범위 저감; 실험 보정 전"],
        ["실제 바닥 자중·중력 전달","기존 200 kN/층 유지; RC 실자중·면외작용 미반영"],
        ["항복·소성힌지·파괴·지진 최대응답","현재 연구 결론의 범위 밖"]])
    report=f'''# STEP16: 동특성·면내변형·횡력 분배 통합 및 탄성 연구 결론

## 1. 연구 질문과 완료 범위

연구명: **다이어프램 조건 및 질량 편심에 따른 강재 골조의 동특성 평가**.
STEP16은 STEP14의 고유치·정적 변위와 STEP15의 기둥 전단력 분담을 동일 사례로 통합했다.
R/S/F 36조합과 물성 기반 가정 RC 36조합, 총 72조합이다.
새 OpenSees 해석 횟수는 0회다. 입력 조건과 저장 결과를 재검산하고 후처리했다.

기본 골조는 3층, 18×12 m 평면의 탄성 강재 모멘트골조다.
층중량 200 kN, 세 층 동일 질량배치, 균등 중력 선행해석과 기둥 PDelta를 유지했다.
유한 바닥은 완전 부착·면내 전용 셸이며 면외강성과 셸질량은 0이다.
RC150의 두께 150 mm, Ec=30 GPa, nu=0.2 및 alpha=0.1/0.3/1.0은 가정·문헌 범위 기반 후보다.
가정한 RC 두께의 자중 810 kN/층(단위중량 25 kN/m³ 가정)과 유지한 총중량 200 kN/층의 불일치가 있다.
따라서 결과는 실제 건물의 설계안 검증보다, 동일 질량에서 강성 효과를 비교한 수치 연구다.

## 2. 통합 데이터와 지표 정의

- 72개 사례: 질량중심·회전관성·정렬된 1차 주기·주요 정적 지표.
- 공통 모드 성분: 층별 골조 12절점 Ux/Uy, 총 72성분.
- 동일 질량배치에서 9개 강체 모드를 기준으로 {checks['modal_same_mass_matches']}개 대응을 재계산했다.
- 각 모델의 U 기준 9개 모드를 다른 질량배치에 대응한 {checks['modal_cross_mass_matches']}개 형상 비교.
- 전 층 공통 위치 변위 {checks['common_position_records']}개, 골조선 기록 {checks['line_records']}개.
- 공통 12절점 RMS와 원 해석의 면적가중 전체 셸 RMS를 별도로 보존했다.

같은 질량배치의 모드 비교에는 그 배치의 실제 대각질량 W를 쓴다.
서로 다른 질량배치 간 비교는 고정 균등 Wref로 각각 다시 정규화한다.

MAC_W(a,b) = |aᵀWb|² / [(aᵀWa)(bᵀWb)].

Wref MAC은 형상 유사성이며 실제 참여질량이 아니다. 에너지·참여율은 각 실제 질량행렬로 계산한다.
질량변화 대응은 저장된 이산 사례 사이의 MAC 기반 대응이며 연속적인 고유치 경로 추적은 아니다.
근접 모드는 상대 주기 차이 3%를 후처리 분류값으로 사용하고, 묶음의 주각으로 비교했다.
3%와 MAC 0.9는 결과 해석용 점검 기준이며 설계기준의 합격 판정값은 아니다.

균등 U에서 다른 질량배치로의 고정 균등가중 대응은 648개 중
{checks['low_MAC_fixed_weight_count']}개(약 {100*checks['low_MAC_fixed_weight_count']/648:.2f}%)가 MAC 0.9 미만이다.
특히 C99·C50_Jlo·C50_Jhi는 각각 54개 대응이 모두 기준 미만이다.
이 표는 형상 재편과 대응 불확실성을 표시하는 진단자료다. 극소질량 절점도 균등하게
평가하는 Wref의 특성, 실제 모드 혼합·형상 변화 등이 함께 영향을 줄 수 있다.
`mass_variation_modes`의 `diagnostic_period_difference_pct`는 후보 쌍의 진단값으로만 저장했다.
이를 편심 변화에 따른 동일 모드의 확정 주기변화나 연속 고유치 경로로 사용하지 않았다.
편심 효과의 주 결론은 정렬된 주기, 실제 질량 기준 참여율·에너지와 정적 대조로 도출했다.

## 3. 질문 1: 1차 주기가 비슷하면 다른 거동도 같은가?

동일 질량배치의 R 대비 최대 차이는 다음과 같다. 서로 다른 지표의 최대치는 같은 사례일 필요가 없다.

{effect_table}

RC 후보의 1차 주기·정적 변위·선별 분배는 R에 가깝다. 그러나 이 결론이 고차 동특성까지
동일하다는 뜻은 아니다. RC150_a10/X99의 강체 9차 대응은 주기 차이
{reliable['period_change_pct']:.4f}%, MAC {reliable['MAC_actual_mass']:.5f}다.
이 모드의 X/Y 참여율은 {100*reliable['participation_X']:.4f}%/{100*reliable['participation_Y']:.4f}%,
질량중심 회전 참여율은 {100*reliable['participation_Rz_CM']:.4f}%다.
이는 높은 모드 주기 차이를 지진 전체 최대응답 증가율로 그대로 환산할 수 없다는 점도 보여준다.
기존 대표 고차 메시 검증의 8→12분할 주기 변화 최대 0.2321%보다 위 차이는 훨씬 크다.

## 4. 질문 2: 편심이 커질수록 비틀림이 단조 증가하는가?

U→C10→C50→C99에서 1차 주기는 증가하지만 회전 운동에너지 비율은 단조 증가하지 않는다.
R의 1차 질량중심 회전 에너지 비율은 C50에서 {100*rc50['mode1_energy_Rz_CM']:.4f}%,
C99에서 {100*rc99['mode1_energy_Rz_CM']:.4f}%다.
극단적 질량 집중으로 질량중심 주위 Jc가 줄어들기 때문이다.
같은 모드의 큰 회전변위와 작은 회전 에너지 분율은 서로 모순되지 않는다.

`step16_rotation_definitions.png`는 두 정의를 나란히 보여준다.
왼쪽은 실제 질량·질량중심 기준의 회전 운동에너지 비율이다.
오른쪽은 기하학적 중심에서 균등 표본으로 맞춘 회전변위 성분 비율이며 에너지가 아니다.
기존 STEP10의 지붕 회전변위 지표와도 기준점·층 합산 방식이 다르므로 숫자를 혼용하지 않는다.
정렬된 1차 모드의 성분을 비교한 그림이므로 모드 순서 변경 가능성도 함께 고려한다.

## 5. 질문 3: 같은 질량중심에서 정적·동적 차이는 어떻게 구분되는가?

아래 R 세 배치는 총질량과 중심 (4.5,3.0)m이 같다.

{controls}

R의 정적 관성력은 합력과 비틀림 모멘트가 같으므로 변위·골조선 분배도 같다.
동적 고유치는 달라진 질량행렬, 특히 강체 평면 운동의 회전관성에 영향을 받는다.
유한 바닥 F에서는 같은 중심이라도 국부 관성력 위치가 달라 정적 결과가 달라진다.
STEP15에서 이 대조군의 F 분담률 차이는 최대 약 12.905%p였다.
이 정적 차이는 회전관성의 동적 효과가 아니라 공간적으로 다른 하중벡터의 효과다.
고정 중력 조건에서는 같은 다이어프램 모델의 선행 중력 상태를 유지한 비교다.

## 6. 질문 4: 기존 F의 결론이 RC 후보에서도 유지되는가?

질량이 몰린 쪽 외곽 골조로 횡력이 집중되는 경향은 유지된다.
그러나 바닥 강성 변화가 이를 추가로 키우는 정도는 다르다.
C99·Y방향 1층 가까운 외곽선은 R 48.287%, F 63.666%, RC150_a30 48.549%다.
RC 후보군 전체의 R 대비 최대 분담률 차이는 0.609%p다.
F의 큰 차이를 일반 RC 슬래브의 실제 응답으로 확대할 근거는 없다.
F의 Et는 RC150 총단면 Et의 약 0.05333%인 수치 저강성 대조군이다.

현재 3D 골조는 직각 방향의 전단력 쌍과 기둥 비틀림도 함께 비틀림에 저항한다.
STEP15의 R/C99/X 1층에서 -yFx 47.692%, xFy 52.264%, 기둥 Mz 0.044%였다.
단방향 골조선만 고려한 간단한 비틀림 분배식을 현재 모델의 정답으로 사용하지 않는다.

## 7. 모드 비교의 신뢰 범위

{modal_table}

총 낮은 동일질량 MAC 대응은 {len(weak)}개다. 낮은 MAC이 곧 해석 오류를 의미하지는 않지만,
그 쌍의 주기차를 동일한 모드의 확정 차이로 해석하지 않는다.
F의 낮은 MAC은 54/108개이며, 전체 대응 최대 주기차 약 158.745%는 MAC 약 0.511이라
주요 정량 결론의 대표 최대치로 사용하지 않았다.

근접한 C99 4·5차 묶음은 다음과 같다.

{cluster_table}

각 벡터의 혼합에 비해 부분공간이 잘 유지되는 경우가 있다.
F의 낮은 MAC 전체를 이 근접 모드 현상 하나로 설명하지 않는다.
전체 모드의 참여질량과 잔차 성분도 별도로 보존했다.
특히 F/U 원 1~9차 누적 X/Y/Rz 참여율은
{100*first9[0]:.3f}%/{100*first9[1]:.3f}%/{100*first9[2]:.3f}%다.
99% 참여 조건은 저장된 전체 모드(R 9개, 유한 모델 48개)로 확인한 것이며
9개 대응 모드만의 누적 참여 조건이 아니다.

## 8. 두 변수의 개별 효과와 상호작용

I(D,E)=Q(D,E)-Q(R,E)-Q(D,U)+Q(R,U).

변위는 동일 절점·층·가력방향, 분담률은 동일 골조선·층·방향을 고정해 계산했다.
단위는 변위 mm, 주기 s, 분담률 %p다. 기준이 0인 회전·왜곡 지표에 백분율을 강제하지 않았다.
최대 변위 지표에는 최대값 위치 변경 가능성을 별도 표시하고 위치별 차분도 제공했다.
그림의 최대 절대 상호작용은 각 고정 위치의 차분을 구한 뒤 절댓값 최대치를 취했다.
상호작용은 수치 응답의 비가산성을 뜻하며, 재료 비선형성이나 통계적 유의성을 뜻하지 않는다.

## 9. 수치 검증과 작은 차이의 판단

- STEP14/15 공통 절점 변위 차이 최대 {checks['max_shared_displacement_difference_mm']:.3e} mm.
- 실제 질량 모달 에너지 재계산 오차 최대 {checks['max_physical_energy_recalculation_error']:.3e}.
- 전체 저장 모드 질량 직교성 오차 최대 {checks['max_mass_orthogonality_error']:.3e}.
- 이전 파일 {checks['existing_files_checked']}개 해시 보존, 변경 {len(checks['changed'])}개.
- STEP15 C99의 선분담률 8→12분할 변화 최대 {m15['mesh12_max_share_change_percentage_points']:.5f}%p.

RC 후보의 작은 차이는 크기와 검증 범위를 함께 제시한다.
대표 C99의 같은 모델·같은 층·같은 선에 대해서만 관측 메시 변화와 효과를 대조했다.
이 점검에서 효과가 메시 이동 이하인 기록은 {len(checked_lines)}개이며
`step16_line_metrics.csv`의 `effect_vs_observed_mesh_shift`에 표시했다.
다른 질량배치에는 ‘해당 사례 메시 미검증’으로 표시했다.
관측된 메시 차이를 모든 사례의 엄밀한 오차 상한으로 취급하지 않는다.
기존 RC 대표 T1 8→12분할 최대 0.00352%, 면내 RMS 상대변화 최대 0.75662% 역시
각 지표의 변화이며 전체 구조 응답 오차율이 아니다.

## 10. 탄성 연구의 최종 결론

1. 이 연구의 탄성 골조에서 질량배치와 바닥 면내강성은 주기·모드·변형·힘 전달에 서로 다른 크기로 영향을 준다.
2. 가정한 RC 후보는 저차 주기와 정적 변위·골조선 분담에서 강체 기준에 가깝다. 고차 동특성은 별도 판단이 필요하다.
3. 매우 낮은 수치 강성 F에서는 편심 조건의 면내변형과 외곽 힘 집중이 커졌다. 실제 RC 바닥의 일반 결과로 확대하지 않는다.
4. 질량중심만으로 유한 바닥의 거동을 설명할 수 없다. 질량분포와 회전관성을 구별하는 대조군이 필요하다.
5. 극단적 편심에서 회전변위·참여율·운동에너지는 서로 다른 지표다. 하나의 수치만으로 병진–비틀림 결합을 판정하지 않는다.
6. 강체 근사의 적절성은 지표별·하중별·모델별로 판단한다. 작은 T1 차이가 전체 거동 일치를 입증하지 않는다.

발표·논문용 표현:

> 가정한 3층 탄성 강재 모멘트골조에서, 바닥판 면내강성과 질량배치의 변화가 동특성 및 정적 관성력 전달에 미치는 영향을 비교하였다. 물성에 연결한 RC 후보는 저차 주기와 정적 횡력 분배에서 강체 기준과 가까웠으나, 일부 고차 모드에서는 차이가 남았다. 저강성 수치 모델에서는 편심과 바닥 면내변형에 따른 힘 집중이 두드러졌으며, 동일 질량중심에서도 국부 질량배치가 유한 바닥의 정적 응답을 바꿀 수 있음을 확인하였다.

## 11. 완료와 후속 연구의 경계

{critical}

다음 연구가 실건물 재현을 목표로 하면 실제 바닥·강재 단면·접합 상세와 자중을 먼저 확정한다.
지진 최대응답이 목표면 감쇠·지진입력과 시간이력 해석 검증이 필요하다.
항복·손상이 목표면 보정된 재료/힌지 모델과 손상 기준을 추가한다.
이 항목들을 현재 탄성 연구의 완료 여부와 혼동하지 않는다.

## 12. 실행과 재현

```powershell
python -m unittest tests.test_step16_modal_tools tests.test_step16_integration_metrics -v
python -m verification.step16_integrated_elastic_conclusion
```

`results/verification/step16_integrated_elastic_conclusion/`에 데이터 13개 CSV와 통합 JSON,
검증 요약, 그림 7개를 저장한다. 모든 입력 경로와 SHA256은 JSON metadata에 있다.
핵심 표는 `step16_effect_envelopes.csv`, 사례별 값은 `step16_case_summary.csv`다.
`same_mass_modes`와 `mass_variation_modes`의 가중치 정의를 바꾸어 해석하지 않는다.
`point_interactions`·`line_interactions`는 같은 위치의 차분형 상호작용이다.
그림·보고서만 재생성: `python -m verification.step16_reporting`.
'''
    (ROOT/"verification/STEP16_통합분석_및_탄성연구결론.md").write_text(report,encoding="utf-8")
    print("STEP16 figures and integrated conclusion written",flush=True)


if __name__=="__main__": run()
