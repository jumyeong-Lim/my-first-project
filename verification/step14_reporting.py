"""STEP14 저장 결과로 그림과 한국어 분석 보고서를 재생성한다."""
from __future__ import annotations
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verification.step14_material_based_diaphragm import OUT, ROOT, SLABS, ORDER, COLORS, original


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers)] +
                     ["| "+" | ".join(str(v) for v in row)+" |" for row in rows])


def run():
    data=json.loads((OUT/"step14_results.json").read_text(encoding="utf-8"))
    results=data["results"]
    checks=json.loads((OUT/"verification_checks.json").read_text(encoding="utf-8"))
    audit=json.loads((OUT/"mode_audit_checks.json").read_text(encoding="utf-8"))
    with (OUT/"step14_mesh_check.csv").open(encoding="utf-8-sig") as f: mesh=list(csv.DictReader(f))
    with (OUT/"step14_matched_modes.csv").open(encoding="utf-8-sig") as f: modes=list(csv.DictReader(f))
    panels=json.loads((OUT/"panel_constitutive_tests.json").read_text(encoding="utf-8"))
    fig,axes=plt.subplots(1,3,figsize=(13,4.4))
    for ax,key,title in zip(axes,((0,0),(0,1),(2,2)),("A11: constrained axial", "A12: Poisson coupling", "A66: in-plane shear")):
        rows=[r for r in panels if r["mesh_n"]==4]
        x=np.arange(len(rows)); i,j=key
        ax.bar(x-.18,[r["A_theory_kN_m"][i][j] for r in rows],.36,label="Theory")
        ax.bar(x+.18,[r["A_FE_kN_m"][i][j] for r in rows],.36,label="OpenSees")
        ax.set(xticks=x,xticklabels=[r["model"] for r in rows],ylabel="kN/m",title=title)
        ax.tick_params(axis="x",rotation=25);ax.grid(axis="y",alpha=.2)
    axes[0].legend();fig.suptitle("Small-panel verification: 2 m x 1 m, 8 x 4 elements")
    fig.tight_layout();fig.savefig(OUT/"step14_panel_verification.png",dpi=180);plt.close(fig)
    # 강체운동과 면내 왜곡을 포함한 정규화 모드의 에너지 분해를 같은 대응 모드에서 비교한다.
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    energy_colors=["#377eb8","#e07a37","#9853a5","#3c9c70"]
    for ax,i in zip(axes,range(3)):
        values=[]
        for code in ORDER:
            case=results[f"{code}_C99"]
            j=i if code=="R" else case["match_to_rigid"]["orders"][i]-1
            values.append(case["modal"]["character"][j]["energy_X_Y_Rz_distortion"])
        values=np.array(values);bottom=np.zeros(len(ORDER))
        for k,label in enumerate(("X translation","Y translation","Rotation about CM","In-plane distortion")):
            ax.bar(ORDER,values[:,k],bottom=bottom,color=energy_colors[k],label=label)
            bottom+=values[:,k]
        ax.set(title=f"C99 / matched rigid mode {i+1}",ylabel="Mass-weighted modal energy fraction",ylim=(0,1.03))
        ax.tick_params(axis="x",rotation=35)
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc="upper center",ncol=4)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(OUT/"step14_mode_energy.png",dpi=180);plt.close(fig)
    # R에 가까워 원값 그림에서 겹치는 물성 후보를 변화율로 추가 표시한다.
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    names=list(original.mass_cases())
    metrics=("T1_s","X_roof_max_mm","Y_roof_max_mm")
    for ax,metric in zip(axes,metrics):
        x=np.arange(len(names))
        for i,code in enumerate(SLABS):
            values=[]
            for name in names:
                a=original.scalar_results(results[f"{code}_{name}"])[metric]
                b=original.scalar_results(results[f"R_{name}"])[metric]
                values.append(100*(a/b-1))
            ax.bar(x+(i-1)*.24,values,.24,label=code,color=COLORS[i+3])
        ax.set(xticks=x,xticklabels=names,title=metric,ylabel="Change from same-mass R (%)")
        ax.tick_params(axis="x",rotation=70);ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/"step14_physical_vs_rigid.png",dpi=180);plt.close(fig)
    rows=[]
    for code in ORDER:
        case=results[f"{code}_C99"];ref=results["R_C99"]
        tx=case["static_X"][-1]["max_directional_mm"]
        ty=case["static_Y"][-1]["max_directional_mm"]
        rows.append([code,f'{case["modal"]["periods_s"][0]:.6f}',f'{tx:.6f}',f'{ty:.6f}',
            f'{100*(tx/ref["static_X"][-1]["max_directional_mm"]-1):.3f}',
            f'{100*(ty/ref["static_Y"][-1]["max_directional_mm"]-1):.3f}'])
    physical_modes=[r for r in modes if r["model"] in SLABS]
    low=[r for r in physical_modes if float(r["MAC"])<.9]
    mesh_rows=[]
    for code in SLABS:
        selected=[r for r in mesh if r["model"]==code and r["mesh"]=="12"]
        mesh_rows.append([code]+[f'{max(abs(float(r["change_pct"])) for r in selected if r["metric"] in keys):.5f}'
            for keys in (("T1_s",),("X_roof_max_mm","Y_roof_max_mm"),("X_roof_distortion_mm","Y_roof_distortion_mm"))])
    all_rows=[]
    for code in SLABS:
        selected=[r for r in physical_modes if r["model"]==code]
        all_rows.append([code,f'{min(float(r["MAC"]) for r in selected):.6f}',
            f'{max(abs(float(r["period_change_pct"])) for r in selected):.4f}'])
    report=f'''# STEP14 결과분석

계산은 `step14_material_based_diaphragm.py`, 물성·문헌과 해석 범위는
`STEP14_물성근거_및_모델설정.md`에 정리했다. 아래는 저장된 결과에서 자동 생성한 수치다.

## 1. 완료 범위

기존 R/S/F 36조합과 RC150_a10/a30/a100 36조합, 총 72조합을 비교했다.
기존 결과의 대표 6조합은 재해석했다. 물성 후보는 세 층 모두에 적용했다.
각 비교에서 총질량·질량중심·회전관성·균등 중력은 같다.
RC150의 E=30 GPa, nu=0.2, 두께=150 mm는 가정값이다.
alpha=0.1/0.3은 NIST 유효강성 범위를 참고한 고정 탄성계수이고, alpha=1은 비균열 총단면 기준이다.
실건물 또는 시험체의 보정 완료를 의미하지 않는다.

## 2. 작은 패널의 인장·전단강성이 구현되는가?

- X/Y 인장 18시험 최대 강성오차: {checks['max_panel_axial_error_pct']:.3e}%.
- 9개 구성행렬 검증 최대 스케일오차: {checks['max_panel_A_scaled_error']:.3e}.
- 기존 결과 재현 최대 차이: {checks['max_baseline_reproduction_error_pct']:.3e}%.
- 기존 파일 {checks['original_files_checked']}개 해시 비교: 변경 {len(checks['changed'])}개.

Et, Gt뿐 아니라 A12의 포아송 결합도 반력으로 복원하여 일치함을 확인했다.
이 검증은 입력한 등방성 막 구성식의 구현 검증이다. 실제 RC 바닥 실험과의 일치 검증은 아니다.

## 3. 극단적 모서리 편심 C99의 대표 결과

{table(['모델','T1 (s)','지붕 최대 X (mm)','지붕 최대 Y (mm)','X의 R 대비 차이 (%)','Y의 R 대비 차이 (%)'],rows)}

변위는 모든 층에 질량×1 m/s²를 적용한 방향별 정적 관성력의 결과다.
동일 가속도와 질량을 사용하므로 입력 총횡력이 동일하다. X/Y 해석은 별개다.
R/S/F는 기존 모델군이며 RC150은 물성 기반 후보군으로 구분한다.

## 4. 주기·모드와 층 내 변형의 해석

균등 질량 U에서는 X/Y 순수 병진에 대한 바닥판 왜곡 요구가 작아 1차 주기 차이가 매우 작다.
편심이 생기면 회전과 비균일 변위가 함께 나타나므로 바닥판 강성의 차이가 더 드러난다.
이 모델에서 탄성 총단면 RC150_a100, 유효 RC150_a30/a10의 결과가 어디에 놓이는지는
위 표와 전체 12개 질량배치 결과를 함께 보아야 한다.

{table(['물성 후보','전체 12배치×9대응모드 최소 MAC','R 대비 최대 대응주기 차이 (%)'],all_rows)}

고유주기 비교 그림은 단순 모드번호 일치가 아니라 동일 질량가중 MAC로 매칭한 강체 모드 순서다.
MAC 0.9 미만인 물성 후보 대응: {len(low)}개.
낮은 MAC은 형상 대응이 약하다는 경고이며 실제 모드 오류의 자동 판정은 아니다.
원 모드번호, 참여질량 및 에너지 분해는 JSON/CSV에서 함께 확인할 수 있다.
회전 참여율과 모달 운동에너지 비율은 기준과 정의가 다르므로 같은 지표로 섞지 않는다.

추가 확인 결과, 낮은 MAC 4개는 모두 C99의 RC150_a30/a100 4·5차 모드였다.
이 두 모드를 한 부분공간으로 비교하면 최소 주각 코사인 제곱은
{audit['minimum_cluster_cosine_squared']:.6f}이다. 개별 벡터가 혼합되는 반면
두 모드가 나타내는 전체 운동공간은 강체 모델과 매우 가깝다는 해석을 뒷받침한다.
해당 형상은 `step14_near_mode_shapes.png`에 전 층 수평 변형으로 표시했다.
형상 진폭은 각각 1.5 m로 정규화한 시각화이며 실제 응답 변위가 아니다.

**고차 주기는 별도 결론이 필요하다.** RC150_a10/X99의 강체 9차 대응 주기는
0.032721 s로 R의 약 0.021255 s보다 53.95% 길고, 해당 MAC은 약 0.974이다.
그 모드의 X/Y 유효 참여율은 약 0%/0.212%, 회전 참여율은 약 3.085%이다.
주기 차이가 크다는 사실만으로 전체 지진응답이 그 비율만큼 증가한다고 해석할 수 없다.
반대로 1차 주기가 비슷하다고 모든 동특성이 강체와 같다고 해석할 수도 없다.
`step14_physical_vs_rigid.png`는 전체 12질량배치에서 1차 주기·정적 최대변위의 차이를 확대한다.

`step14_all_floor_displacements.png`는 1~3층의 위치별 최대변위와
최소제곱 강체이동을 제거한 면내 잔차 RMS를 보여준다.
`step14_roof_positions.png`는 C99의 지붕 공통 12절점 변위다.
최대변위 한 개가 같아도 위치별 차이와 왜곡 잔차가 같다는 뜻은 아니다.

## 5. 메시 민감도

C99에서 경간당 4/8/12분할을 비교했다. 다음은 8분할 기준 12분할의 변화 절댓값이다.

{table(['모델','T1 변화 (%)','X/Y 최대변위 최대 변화 (%)','X/Y 면내 RMS 최대 변화 (%)'],mesh_rows)}

주기와 전체 변위의 수렴과 작은 면내 잔차의 상대오차를 따로 판단한다.
8→12분할 비교는 국부 연결부 응력 및 모든 고차 모드의 완전 수렴을 뜻하지 않는다.
고차 차이가 크게 나온 X99와 근접 모드가 나타난 C99에 대해서는 9개 대응 모드를
12분할에서 추가 확인했다. 8→12 모드 MAC 최소값은 {audit['minimum_mesh_MAC']:.6f},
대응 주기 차이의 최댓값은 {audit['maximum_higher_mode_mesh_period_change_pct']:.4f}%다.
따라서 확인한 고차 차이를 이번 메시 오차만으로 설명하기는 어렵다.

## 6. 연구 질문에 대한 결론

1. 형식·두께·재료·유효강성 계수로부터 목표 막 강성을 정의하고 OpenSees에 구현하는 연결은 확인했다.
2. 기존 S의 Et는 RC150 총단면의 약 53.33%, F는 약 0.05333%다.
   F의 큰 응답 차이를 이번 RC 후보 바닥의 일반적인 실제 응답으로 해석할 근거는 없다.
3. RC150 후보는 F보다 훨씬 강하다. 강체 근사에 얼마나 가까운지는 동일 편심에서의
   주기·모드·위치별 변위로 판단해야 하며, RC라는 형식명만으로 강체라고 판정하지 않는다.
4. 감소된 E의 선형 고유치 결과는 가정한 할선강성 상태에 대한 결과다. 지진 중 균열 발생을 예측하지 않았다.
5. 동일 총질량 비교는 성공했으나 가정 RC 자중 810 kN/층과 유지한 200 kN/층은 다르다.
   따라서 ‘RC 강성을 근거로 한 통제 비교’가 이번 최종 해석 범위다.

상호작용은 Q(D,E)-Q(R,E)-Q(D,U)+Q(R,U)로 기존 정의를 유지한다.
`step14_interactions.csv`는 이 값과 동일 편심에서 R 대비 차이를 분리한다.
차분이 0이 아니어도 재료 비선형성이나 통계적 유의성을 입증한 것은 아니다.

## 7. 남은 물리적 보정

선택된 건물/시험체의 두께와 재료시험, 배근·균열 상태, 스터드·정착 상세를 확보하면
RC150의 가정값을 교체할 수 있다. 데크 합성바닥은 제품 형상과 방향별 A 및 접합 근거를
확보한 뒤 별도 후보로 추가한다. 실제 질량·중력 및 면외판강성을 반영하는 비교는 별도 분석군이다.
기존 연구의 다음 횡력 분배 분석에는 이번 물성 후보군도 추가할 수 있다.

## 8. 주요 결과 파일

- `material_properties.csv`: 입력과 Et/Gt 비교.
- `panel_axial_tests.csv`, `panel_constitutive_tests.json`: 소형 패널 검증.
- `step14_summary.csv`: 72조합 대표 지표.
- `step14_matched_modes.csv`: 모드 대응, 주기, 참여율.
- `step14_floor_displacements.csv`: 전 층 최대변위·면내 잔차.
- `step14_interactions.csv`: 개별 효과와 차분형 상호작용.
- `step14_mesh_check.csv`, `verification_checks.json`: 메시·재현·보존 검증.
- `near_mode_subspace_check.csv`, `higher_mode_mesh_check.csv`: 근접 모드·고차 모드 추가 검증.
- `step14_results.json`: 모든 원 모드와 위치별 변위장 및 입력 가정.
'''
    (ROOT/"verification/STEP14_결과분석.md").write_text(report,encoding="utf-8")
    print("STEP14 report and additional figures written")


if __name__ == "__main__":
    run()
