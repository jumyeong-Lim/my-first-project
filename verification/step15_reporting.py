"""STEP15의 저장된 힘 분배 결과를 그림과 한국어 보고서로 정리한다."""
from __future__ import annotations
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verification.step15_frame_shear_distribution import OUT, ROOT, ORDER, SLABS, original
from verification.step14_material_based_diaphragm import COLORS
from verification.step12_shell_audit import save_csv


def markdown_table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers)] +
                     ["| "+" | ".join(str(v) for v in row)+" |" for row in rows])


def line_values(case,floor,field="signed_share_pct"):
    rows=[r for r in case["line_results"] if r["floor"]==floor]
    return np.array([r["line_coord_m"] for r in rows]), np.array([r[field] for r in rows])


def share_bars(ax,cases,name,direction,floor):
    coords,_=line_values(cases[f"R_{name}_{direction}"],floor)
    x=np.arange(len(coords));width=.13
    for i,(code,color) in enumerate(zip(ORDER,COLORS)):
        _,values=line_values(cases[f"{code}_{name}_{direction}"],floor)
        ax.bar(x+(i-2.5)*width,values,width,color=color,label=code)
    ax.axhline(0,color="black",lw=.7)
    ax.set(xticks=x,xticklabels=[f"{v:+g}" for v in coords],ylabel="Signed story-shear share (%)",
           xlabel=("Frame line Y (m)" if direction=="X" else "Frame line X (m)"),
           title=f"{name} / {direction} loading / story {floor}")
    ax.grid(axis="y",alpha=.2)


def plots(cases,interactions,torques):
    # 골조선 정의: 방향별 내부선의 개수가 다른 점을 명시한다.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,d in zip(axes,("X","Y")):
        for y in (-6,0,6): ax.plot([-9,9],[y,y],color="lightgray",lw=1)
        for x in (-9,-3,3,9): ax.plot([x,x],[-6,6],color="lightgray",lw=1)
        if d=="X":
            for y in (-6,0,6):
                color="#377eb8" if y==0 else "#e07a37"
                ax.plot([-9,9],[y,y],color=color,lw=3)
                ax.text(10,y,f"Y={y:+g}",va="center",fontsize=9)
            ax.annotate("X loading",xy=(3,-8),xytext=(-5,-8),arrowprops={"arrowstyle":"->"})
        else:
            for x in (-9,-3,3,9):
                color="#e07a37" if abs(x)==9 else "#377eb8"
                ax.plot([x,x],[-6,6],color=color,lw=3)
                ax.text(x,7,f"X={x:+g}",ha="center",fontsize=9)
            ax.annotate("Y loading",xy=(11,5),xytext=(11,-4),arrowprops={"arrowstyle":"->"})
        ax.scatter(original.XY[:,0],original.XY[:,1],color="black",s=14)
        ax.set(xlim=(-12,15),ylim=(-10,10),aspect="equal",xlabel="X (m)",ylabel="Y (m)",
               title=f"{d} loading: blue=interior, orange=exterior")
    fig.tight_layout();fig.savefig(OUT/"step15_frame_line_layout.png",dpi=180);plt.close(fig)
    for floor in (1,3):
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for row,d in enumerate(("X","Y")):
            for col,name in enumerate(("U","C99")): share_bars(axes[row,col],cases,name,d,floor)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",ncol=6,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/f"step15_line_shares_story{floor}.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for row,d in enumerate(("X","Y")):
        for col,floor in enumerate((1,2,3)): share_bars(axes[row,col],cases,"C99",d,floor)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=6,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/"step15_C99_all_story_shares.png",dpi=180);plt.close(fig)
    # 그룹의 선 개수 차이를 제거한 선당 평균을 제시한다. 그룹 합은 CSV에 별도 저장된다.
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,d in enumerate(("X","Y")):
        for col,name in enumerate(("U","C99")):
            ax=axes[row,col];x=np.arange(2);width=.13
            for i,(code,color) in enumerate(zip(ORDER,COLORS)):
                c=cases[f"{code}_{name}_{d}"]
                values=[next(r["signed_share_per_line_mean_pct"] for r in c["group_results"]
                             if r["floor"]==1 and r["group"]==g) for g in ("interior","exterior")]
                ax.bar(x+(i-2.5)*width,values,width,color=color,label=code)
            ax.set(xticks=x,xticklabels=["Interior","Exterior"],ylabel="Mean share per frame line (%)",title=f"{name} / {d} / story 1")
            ax.axhline(0,color="black",lw=.7);ax.grid(axis="y",alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=6,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/"step15_group_mean_shares.png",dpi=180);plt.close(fig)
    # 최대 절대차는 같은 질량배치의 R과 층/선이 같은 값끼리 비교한다.
    names=list(original.mass_cases());matrices=[]
    for d in ("X","Y"):
        matrices.append(np.array([[max(abs(r["diaphragm_effect_percentage_points"]) for r in interactions
                if r["model"]==code and r["mass_case"]==name and r["direction"]==d)
                for name in names] for code in ORDER[1:]]))
    fig,axes=plt.subplots(2,1,figsize=(14,7))
    vmax=max(float(m.max()) for m in matrices)
    for ax,d,matrix in zip(axes,("X","Y"),matrices):
        im=ax.imshow(matrix,cmap="YlOrRd",vmin=0,vmax=vmax,aspect="auto")
        ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(5),yticklabels=ORDER[1:],
               title=f"{d}: maximum absolute line-share change from same-mass R, across all stories")
        for i in range(5):
            for j in range(len(names)):
                ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=8,
                        color="white" if matrix[i,j]>.6*vmax else "black")
        ax.tick_params(axis="x",rotation=25)
    fig.subplots_adjust(left=.13,right=.88,bottom=.12,top=.94,hspace=.5)
    color_axis=fig.add_axes([.905,.19,.016,.62])
    fig.colorbar(im,cax=color_axis,label="Difference (percentage points)")
    fig.savefig(OUT/"step15_share_change_heatmap.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,10),constrained_layout=True)
    for row,d in enumerate(("X","Y")):
        for col,code in enumerate(("R","F","RC150_a30")):
            ax=axes[row,col];c=cases[f"{code}_C99_{d}"]
            for y in (-6,0,6): ax.plot([-9,9],[y,y],color="lightgray",lw=.8)
            for x in (-9,-3,3,9): ax.plot([x,x],[-6,6],color="lightgray",lw=.8)
            coords,values=line_values(c,1)
            for pos,share in zip(coords,values):
                start=np.array([0.,pos]) if d=="X" else np.array([pos,0.])
                delta=np.array([share/100*6.,0.]) if d=="X" else np.array([0.,share/100*6.])
                ax.quiver(*start,*delta,angles="xy",scale_units="xy",scale=1,
                          color="#c44e52" if share<0 else "#3274a1",width=.008)
                textpos=start+delta+np.array([0.,.6] if d=="X" else [.6,0.])
                ax.text(*textpos,f"{share:.1f}%",fontsize=9,ha="center")
            ax.scatter(*c["center_xy_m"],marker="*",s=90,color="#9853a5",label="CM")
            ax.set(title=f"{code} / C99 / {d} / story 1",xlim=(-12,12),ylim=(-9,10),
                   aspect="equal",xlabel="X (m)",ylabel="Y (m)")
    fig.suptitle("Signed line shears: arrows use a common scale; red = opposite to applied direction")
    fig.savefig(OUT/"step15_C99_plan_arrows.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,d in zip(axes,("X","Y")):
        bottom=np.zeros(len(ORDER))
        for field,label,color in (("Fx_lever_share_pct","-y Fx","#377eb8"),
                                  ("Fy_lever_share_pct","x Fy","#e07a37"),
                                  ("column_torsion_share_pct","Column Mz","#9853a5")):
            values=np.array([next(r[field] for r in torques if r["model"]==code and
                r["mass_case"]=="C99" and r["direction"]==d and r["floor"]==1) for code in ORDER])
            ax.bar(ORDER,values,bottom=bottom,label=label,color=color);bottom+=values
        ax.set(ylabel="Contribution to applied torsional moment (%)",title=f"C99 / {d} / story 1")
        ax.tick_params(axis="x",rotation=30);ax.grid(axis="y",alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/"step15_torsional_balance.png",dpi=180);plt.close(fig)


def run():
    data=json.loads((OUT/"step15_results.json").read_text(encoding="utf-8"));cases=data["cases"]
    checks=json.loads((OUT/"verification_checks.json").read_text(encoding="utf-8"))
    with (OUT/"step15_interactions.csv").open(encoding="utf-8-sig") as f:
        interactions=list(csv.DictReader(f))
    for r in interactions:
        for key in ("diaphragm_effect_percentage_points","interaction_percentage_points"):
            r[key]=float(r[key])
    torques=[]
    for c in cases.values():
        for floor in (1,2,3):
            cs=[r for r in c["columns"] if r["floor"]==floor]
            mfx=sum(-r["y_m"]*r["cut_Fx_kN"] for r in cs)
            mfy=sum(r["x_m"]*r["cut_Fy_kN"] for r in cs)
            mz=sum(r["cut_Mz_kNm"] for r in cs)
            expected=c["equilibrium"][floor-1]["expected_torque_kNm"]
            torques.append({"model":c["model"],"mass_case":c["mass_case"],"direction":c["direction"],
                "floor":floor,"Fx_lever_moment_kNm":mfx,"Fy_lever_moment_kNm":mfy,
                "column_torsion_kNm":mz,"external_torque_kNm":expected,
                "Fx_lever_share_pct":100*mfx/expected if abs(expected)>1e-8 else None,
                "Fy_lever_share_pct":100*mfy/expected if abs(expected)>1e-8 else None,
                "column_torsion_share_pct":100*mz/expected if abs(expected)>1e-8 else None})
    save_csv(OUT/"step15_torsional_balance.csv",torques)
    plots(cases,interactions,torques)
    rigid_torque=next(r for r in torques if r["model"]=="R" and r["mass_case"]=="C99"
                     and r["direction"]=="X" and r["floor"]==1)
    tables=[]
    for d in ("X","Y"):
        coords,_=line_values(cases[f"R_C99_{d}"],1)
        rows=[]
        for code in ORDER:
            _,values=line_values(cases[f"{code}_C99_{d}"],1)
            rows.append([code]+[f"{v:.3f}" for v in values])
        tables.append(markdown_table(["모델"]+[f"{'Y' if d=='X' else 'X'}={v:+g} m (%)" for v in coords],rows))
    ranges=[]
    for code in ORDER[1:]:
        rows=[r for r in interactions if r["model"]==code]
        largest=max(rows,key=lambda r:abs(r["diaphragm_effect_percentage_points"]))
        ranges.append([code,f"{abs(largest['diaphragm_effect_percentage_points']):.4f}",
            largest["mass_case"],largest["direction"],largest["floor"],largest["line_coord_m"]])
    largest_f=max(abs(r["diaphragm_effect_percentage_points"]) for r in interactions if r["model"]=="F")
    largest_rc=max(abs(r["diaphragm_effect_percentage_points"]) for r in interactions if r["model"] in SLABS)
    reverse=[(key,r) for key,c in cases.items() for r in c["line_results"] if r["V_kN"] < -1e-8]
    max_reverse=min(reverse,key=lambda pair:pair[1]["signed_share_pct"]) if reverse else None
    extremes="없음" if max_reverse is None else f"{max_reverse[0]}, {max_reverse[1]['floor']}층, 선 좌표 {max_reverse[1]['line_coord_m']:+g} m, {max_reverse[1]['signed_share_pct']:.3f}%"
    # 같은 중심 대조군에서 유한 다이어프램의 국부 질량배치 영향도 정량화한다.
    inertia_rows=[]
    for code in ORDER:
        differences=[]
        for d in ("X","Y"):
            ref=cases[f"{code}_C50_{d}"]["line_results"]
            for name in ("C50_Jlo","C50_Jhi"):
                other=cases[f"{code}_{name}_{d}"]["line_results"]
                differences.extend(abs(a["signed_share_pct"]-b["signed_share_pct"]) for a,b in zip(ref,other))
        inertia_rows.append([code,f"{max(differences):.5f}"])
    text=f'''# STEP15: 다이어프램 조건·질량편심과 중앙/외곽 골조 횡력 분배

## 1. 목적과 해석 범위

같은 총질량과 단위가속도 관성력에서 각 층의 어느 골조선이 횡력을 얼마나 부담하는지 평가했다.
기존 R/S/F와 STEP14의 RC150_a10/a30/a100을 12개 질량배치에 적용했다.
총 72 모델·질량 조합, 방향별 144회 정적해석이다. 대표 U/C99를 먼저 확인하고 전체로 확장했다.
3층 전체에 같은 질량분포를 적용한다. R/S/F는 수치 강성 모델, RC는 가정 물성 기반 후보다.
보·기둥은 탄성 모멘트골조, 기둥 변환은 PDelta, 바닥은 면내 전용/완전 부착이다.
별도의 집합재 상세나 접합부 용량을 설계한 모델은 아니며, 기존 보 격자와 공유 절점으로 힘을 전달한다.
층 총중량 200 kN과 균등 중력 선행해석을 유지한다.
실제 RC 자중과 이 총질량의 불일치는 STEP14와 동일하며, 동일 질량에서의 통제 비교다.
변위·부재력은 질량×1 m/s² 정적 하중에 대한 결과이며 지진 시간이력 최대값이 아니다.

## 2. 골조선과 전단력의 정의

- X 가력: Y=-6, 0, +6 m. 내부 1선, 외곽 2선, 각 선에 기둥 4개.
- Y 가력: X=-9, -3, +3, +9 m. 내부 2선, 외곽 2선, 각 선에 기둥 3개.
- 그룹 합계와 선당 평균을 모두 산출해 선 개수 차이를 구분했다.
- 하나의 기둥은 가력방향에 따른 골조선에 한 번씩만 포함된다.

각 층의 하단 절단면에서 기둥 하단의 전역 요소 저항력을 취했다.
먼저 중력만의 요소력 fg를 저장하고, 수평하중 후 ft에서 빼서 증분을 구했다.
주 입력 방향을 양수로 나타내기 위해 하단 증분 요소력에 음수를 붙였다.

V(l,g) = -sum( f_lower,total - f_lower,gravity )의 가력방향 성분

eta(l,g) = 100 V(l,g) / V_story(l)

eta는 부호 있는 %이며 합이 100%다. 예를 들어 -5%는 전체 층전단력의 5% 크기로
입력과 반대 방향 전단력을 부담한다는 뜻이다. 손상이나 파괴를 나타내지 않는다.
`column_abs_share_pct`는 기둥 전단력 절댓값 합의 분포이며 별도 보조지표다.
부호 있는 eta와 같은 분모를 쓰지 않으므로 혼용하지 않는다.

PDelta에서는 localForce를 단순 회전한 전단력이 전역 평형 전단력과 다를 수 있다.
단일 기둥에 50 kN 중력, X방향 1 kN을 가하면 전역 하단 전단력 크기는 1 kN,
localForce를 축 변환한 크기는 약 1.003344 kN이었다.
따라서 전역 요소력과 실제 기초반력을 기준으로 평형을 확인했다.
공식 설명: https://openseespydoc.readthedocs.io/en/latest/src/eleForce.html
및 https://openseespydoc.readthedocs.io/en/latest/src/pdelta.html

## 3. 검증 결과

- 단일 기둥: Linear/PDelta × X/Y × i/j 순서 = 8조건 통과.
- 대칭 U: 좌우 대칭선 분담률 최대 차이 {checks['max_uniform_mirror_share_error_percentage_points']:.3e} %p.
- 전체 층전단력 평형 최대 오차 {checks['max_force_equilibrium_error_kN']:.3e} kN.
- 전체 수직축 비틀림 모멘트 평형 최대 오차 {checks['max_torque_equilibrium_error_kNm']:.3e} kN·m.
- 1층 기둥 절단력–기초반력 최대 차이 {checks['max_base_cut_reaction_error_kN']:.3e} kN.
- STEP14 공통 지붕절점 변위 재현 최대 차이 {checks['max_roof_reproduction_error_mm']:.3e} mm.
- C99의 8→12분할 골조선 분담률 최대 변화 {checks['mesh12_max_share_change_percentage_points']:.5f} %p.
- 기존 파일 {checks['original_files_checked']}개 보존: 변경 {len(checks['changed'])}개.

기본 검증 허용값은 힘 평형 절대 10^-7 kN, 비틀림 평형 절대 10^-6 kN·m와 상대 10^-8이다.
층전단력은 1/2/3층에서 각각 61.182973/40.788649/20.394324 kN이다.
면내 비틀림 평형은 모든 기둥의 x*Fy-y*Fx+Mz 합으로 검사했다.
주 가력방향 전단력의 편심 모멘트만으로 비틀림 전체를 설명하지 않았다.

X 가력 C99/R의 1층에서 비틀림 분담은 -y*Fx 약 {rigid_torque['Fx_lever_share_pct']:.3f}%,
x*Fy 약 {rigid_torque['Fy_lever_share_pct']:.3f}%, 기둥 Mz 약 {rigid_torque['column_torsion_share_pct']:.3f}%다.
따라서 X방향 골조선만 있는 단순 1방향 예제의 분담률을 현재 3D 골조에 그대로 적용할 수 없다.
직각 방향 전단력은 총합이 0이더라도 힘의 쌍으로 비틀림에 저항할 수 있다.
`step15_torsional_balance.png`에 이 경로를 표시했다.

## 4. C99의 1층 골조선 분담률

X 가력:

{tables[0]}

Y 가력:

{tables[1]}

`step15_C99_plan_arrows.png`는 위 값을 공통 화살표 척도로 나타낸다.
질량중심은 별표, 역방향 전단력은 붉은 화살표다. 화살표는 골조선 합력이며 개별 기둥력이 아니다.

## 5. 다이어프램 효과가 가장 큰 조건

다음 표는 전체 12배치·2방향·3층·골조선에서 같은 질량배치 R 대비 분담률 차이의 최댓값이다.
%p는 두 백분율의 차이다. 예를 들어 20%→25%는 5%p 증가다.

{markdown_table(['모델','최대 절대차 (%p)','질량배치','가력','층','선 좌표 (m)'],ranges)}

기존 F는 이번 물성 RC 범위보다 훨씬 낮은 면내강성을 갖는 수치 대조군이다.
F의 큰 분배 변화를 실제 RC 슬래브의 일반적인 결과로 확대 해석하지 않는다.
RC 후보의 차이가 작은 경우에도 강체가 모든 동특성을 재현한다는 뜻은 아니다.
이 결과는 정적 힘 전달에 관한 결과이며 STEP14의 고차 모드 차이와 구분한다.

## 6. 같은 질량중심이라도 국부 배치가 영향을 주는가?

C50과 C50_Jlo/Jhi는 총질량·중심이 같고 질량분포와 극관성이 다르다.
R에서 균일 가속도 정적 하중은 총합력과 중심으로 결정되므로 세 배치는 같은 힘 분배를 준다.
유한강성 바닥에서는 국부 관성력의 배치가 달라져 분담률이 달라질 수 있다.
정적 비교에서 나타난 차이를 극관성의 동적 효과라고 해석하지 않는다.

{markdown_table(['모델','C50 대비 대조군의 최대 분담률 차이 (%p)'],inertia_rows)}

역방향 전단력은 전체 결과에서 {len(reverse)}개 층·골조선 기록에 나타났다.
가장 작은 부호 있는 분담률: {extremes}.

## 7. 연구 해석

질량이 몰린 쪽의 관성력이 커지는 것은 예상되는 경향이다. 이번 분석은 그 경향의
크기·층별 분포·다이어프램 의존성을 측정한다. 중심 거리만으로 분담률이 결정되는 것은 아니다.
양 방향 골조의 강성, 바닥판 변형, 횡방향 전단력과 기둥 비틀림이 함께 평형을 이룬다.
내부선과 외곽선의 평균만 비교하면 서로 반대 부호인 양쪽 외곽선의 차이가 숨겨질 수 있으므로
선별 결과를 주 결과로, 그룹 합·선당 평균을 보조 결과로 제시했다.

이번 조건에서 F의 동일 배치 R 대비 분담률 차이는 최대 {largest_f:.3f} %p,
RC 후보군의 차이는 최대 {largest_rc:.3f} %p다.
따라서 현재 RC 후보의 정적 횡력 분배는 강체 기준에 가깝고, 낮은 수치 강성 F에서는
가까운 외곽선으로의 힘 집중이 크게 증가했다. 이 근접성을 다른 평면·연결·단면·하중에
대한 보편적인 강체 다이어프램 판정으로 확대하지 않는다.
역방향 전단력은 강체 R에서도 발생하며, 바닥을 유연하게 할수록 모든 반대편 선의
역방향 전단력이 단조롭게 커지는 것은 아니다. 방향·층·선별 결과를 함께 해석해야 한다.

상호작용은 분담률 Q에 대해 Q(D,E)-Q(R,E)-Q(D,U)+Q(R,U)로 계산했다.
단위는 %p이며 `step15_interactions.csv`에 저장했다.
정적 분배의 차분형 상호작용을 재료 비선형성이나 통계적 유의성으로 해석하지 않는다.

## 8. 그림과 파일

- `step15_frame_line_layout.png`: 방향별 내부/외곽 골조선 정의.
- `step15_line_shares_story1.png`, `step15_line_shares_story3.png`: U/C99의 선별 부호 있는 분담률.
- `step15_C99_all_story_shares.png`: C99의 전 층 비교.
- `step15_group_mean_shares.png`: 1층 그룹의 선당 평균.
- `step15_share_change_heatmap.png`: 같은 배치 R 대비 최대 분담률 변화.
- `step15_C99_plan_arrows.png`: 골조선 힘의 방향과 크기.
- `step15_torsional_balance.png`, `step15_torsional_balance.csv`: 양 방향 전단력과 기둥 비틀림의 모멘트 평형.
- `step15_line_shares.csv`: 전체 선별 전단력·분담률.
- `step15_group_shares.csv`: 그룹 합과 선당 평균.
- `step15_equilibrium.csv`, `step15_mesh_check.csv`: 평형·메시 검증.
- `step15_results.json`: 중력 기둥력·증분 기둥력·절점변위 등 원 결과.

## 9. 실행

```powershell
python -m unittest tests.test_step15_frame_shear_distribution -v
python -m verification.step15_frame_shear_distribution
```

그림·보고서만 다시 만들 때: `python -m verification.step15_reporting`.
후속 STEP16에서는 이번 분담률을 기존 주기·모드·면내변형 지표와 함께 해석할 수 있다.
부재 항복·힌지·파괴 또는 실제 지진 최대응답은 이 단계의 평가 대상이 아니다.
'''
    (ROOT/"verification/STEP15_횡력분배_결과분석.md").write_text(text,encoding="utf-8")
    print("STEP15 figures and Korean report written",flush=True)


if __name__ == "__main__": run()
