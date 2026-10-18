# Two-dimensional verification models

## STEP08: 바닥판 면내강성 민감도

`step08_diaphragm_inplane_sensitivity.py`는 STEP07의 대칭 3D 골조에 강체 구속과 서로 다른 면내강성의 셸 바닥판을 적용한다.
층 내 상대변위와 모드 주기·형상 변화를 비교한다. 상세 가정과 한계는 [STEP08_설명.md](STEP08_설명.md)를 참고한다.
STEP07·08의 결과와 후속 검증 목록은 [STEP07_STEP08_검토.md](STEP07_STEP08_검토.md)에 정리했다.

```powershell
python verification/step08_diaphragm_inplane_sensitivity.py
python -m unittest discover -s tests -p test_step08_diaphragm_inplane_sensitivity.py -v
```

## STEP07: 3D 대칭 기준 골조 확장

`step07_3d_reference_frame.py`에 1층 소형 검증모델과 3층 3×2경간 대칭 3D 골조를 추가했다.
강체 다이어프램·분포 수평질량·중력 선행해석 후 병진/비틀림 모드를 검증한다.
한국어 모델 가정과 결과 설명은 [STEP07_설명.md](STEP07_설명.md)를 참고한다.

```powershell
python verification/step07_3d_reference_frame.py
python -m unittest discover -s tests -p test_step07_3d_reference_frame.py -v
```

This step verifies the static lateral stiffness of a one-story, one-bay,
two-dimensional steel portal frame before modal or three-dimensional models
are developed.

## Model

- Units: kN, m, s
- Bay width: 6 m
- Story height: 4 m
- Assumed beam properties: `A = 0.010 m^2`, `I = 2.0e-4 m^4`
- Assumed column properties: `A = 0.020 m^2`, `I = 4.0e-4 m^4`
- Steel elastic modulus: 200 GPa
- Fixed column bases and rigid beam-column joints
- Total horizontal reference load: 1 kN, divided equally between roof nodes
- Linear elastic `elasticBeamColumn` elements and `Linear` transformation

## Analytical comparison

The textbook limits are

```text
Ib -> 0        k =  6 E Ic / h^3
Ib -> infinity k = 24 E Ic / h^3
```

For finite beam stiffness,

```text
rho = (E Ib / L) / (2 E Ic / h)
k = (24 E Ic / h^3) (12 rho + 1) / (12 rho + 4)
```

The section properties are round verification values, not catalogue sections
and not sections inherited from the previous project. The formula neglects
member axial deformation. The benchmark cases therefore multiply the member
areas by 1,000,000 while leaving the bending inertias unchanged. The additional
assumed-section case uses the reference areas and reports the stiffness
reduction caused by axial deformation.

## Run

```powershell
python verification/step01_static_stiffness.py
python -m unittest discover -s tests -p "test_step01_static_stiffness.py" -v
```

Outputs are written to `results/verification/step01_static_stiffness/`.

## Step 02 - Natural period

Step 02 uses the same assumed section properties and frame geometry. A 200 kN floor
weight is converted to mass using `M = W/g`, divided between the two roof
nodes, and the roof horizontal DOFs are tied to form one lateral floor DOF.

The OpenSees eigenperiod is checked against

```text
T = 2 pi sqrt(M / k)
```

where `k` is the Step 01 static stiffness with the assumed member areas.

```powershell
python verification/step02_natural_period.py
python -m verification.step02_natural_period
python -m unittest discover -s tests -p "test_step02_natural_period.py" -v
```

Outputs are written to `results/verification/step02_natural_period/`.

## Step 03 - Three-story shear-building comparison

Step 03 compares two models with the same three 200 kN floor weights:

1. A three-DOF shear building assembled directly from `K` and `M`.
2. A three-story, one-bay OpenSees elastic moment frame.

The shear-building story stiffness is the Step 01 one-story static stiffness
obtained with the same beam-inertia and axial-area factors. For uniform story
stiffness `k`,

```text
K = [[ 2k, -k,  0],
     [ -k, 2k, -k],
     [  0, -k,  k]]

M = diag(m1, m2, m3)
```

The analytical model is solved as a symmetric generalized eigenproblem. In the
OpenSees model, the two horizontal DOFs on each floor are tied with `equalDOF`;
vertical translations and joint rotations remain in the frame and are
statically condensed during eigenanalysis. The comparison reports all three
periods, roof-normalized mode shapes, and mass-weighted modal assurance
criterion (MAC).

The sensitivity cases use beam-inertia factors `0.25`, `1.0`, and `4.0`. An
additional axial-rigid case uses an area factor of `1e6` to separate axial
deformation from frame/shear-building idealization effects.

### Reference-model values and their status

The following are researcher-assumed inputs, not values obtained from an
existing building or a catalogue section:

- Three stories and one 6 m bay
- Uniform 4 m story height
- Steel elastic modulus `E = 200 GPa`
- Beam properties `A = 0.010 m^2`, `I = 2.0e-4 m^4`
- Column properties `A = 0.020 m^2`, `I = 4.0e-4 m^4`
- Uniform floor weight `W = 200 kN` at each floor
- Gravity `g = 9.80665 m/s^2`
- Fixed bases, rigid beam-column joints, linear-elastic members
- Lumped floor mass; member self-mass is not included
- Only the two horizontal DOFs on each floor are tied

The following values are calculated, not independently assumed:

- Floor mass: `m = W/g = 20.39432 kN s^2/m`
- Mass assigned at each beam-column node: `m/2 = 10.19716 kN s^2/m`
- Reference story stiffness: `k = 14977.84 kN/m`, obtained from the Step 01
  one-story OpenSees static analysis
- Eigenvalues, natural periods, mode shapes, and MAC values

```powershell
python verification/step03_three_story_shear_comparison.py
python -m verification.step03_three_story_shear_comparison
python -m unittest discover -s tests -p "test_step03_three_story_shear_comparison.py" -v
```

Outputs are written to
`results/verification/step03_three_story_shear_comparison/`.

The output directory includes two model-definition figures:

- `shear_building_model.png`
- `opensees_frame_model.png`

It also includes three beam-stiffness sensitivity figures:

- `beam_stiffness_sensitivity_summary.png`: calibrated story stiffness,
  first-mode period difference, and MAC
- `beam_stiffness_period_comparison.png`: matched shear-building and OpenSees
  periods for modes 1-3 at `0.25 Ib`, `1.0 Ib`, and `4.0 Ib`
- `fixed_shear_period_comparison.png`: one fixed reference shear building and
  the OpenSees frame periods at `0.25 Ib`, `1.0 Ib`, and `4.0 Ib`; four bars
  are shown for each mode
- `beam_stiffness_mode_shapes.png`: the reference shear-building mode shapes
  together with the OpenSees frame mode shapes for all three beam stiffnesses

Yellow boxes in these figures identify assumed inputs. Blue or green boxes
identify derived values or OpenSees model definitions.

## OpenSeesPy commands checked

- `mass(nodeTag, *massValues)`: nodal mass for each DOF
- `equalDOF(rNodeTag, cNodeTag, *dofs)`: retained/constrained DOF equality
- `eigen('-fullGenLapack', n)`: requested eigenvalues
- `modalProperties('-return')`: total free mass and modal properties after `eigen`

Official references:

- https://openseespydoc.readthedocs.io/en/latest/src/mass.html
- https://openseespydoc.readthedocs.io/en/latest/src/equalDOF.html
- https://openseespydoc.readthedocs.io/en/latest/src/eigen.html
- https://openseespydoc.readthedocs.io/en/latest/src/modalProperties.html

## Step 04 - Three-story multi-bay 2D frame

Step 04 extends the verified frame to one, two, and three bays. All models use
the same assumed 6 m bay width, 4 m story height, member properties, and total
floor weight of 200 kN. The total floor mass is intentionally fixed as the bay
count changes, so this verification isolates the effect of adding frame lines
and members rather than increasing both stiffness and tributary mass.

For each bay count, three independent static unit-load cases construct the
floor flexibility matrix:

```text
C_ij = displacement of floor i caused by a unit horizontal load at floor j
Kc = inverse(C)
```

The statically condensed `3 x 3` floor stiffness matrix is then used in

```text
[Kc - omega^2 M] phi = 0
```

and its periods and mode shapes are checked against the full OpenSees frame
eigenanalysis. This avoids assuming a shear-building stiffness pattern for the
multi-story moment frame.

Assumed inputs:

- Bay counts: 1, 2, and 3
- Total floor weight: 200 kN at every floor for every bay count
- Floor mass is divided equally among all beam-column nodes on that floor
- Same beam and column properties as Steps 01-03
- Fixed bases, rigid beam-column connectivity, linear-elastic members
- Horizontal floor DOFs tied; vertical translations and rotations remain free
- Member self-mass is not included

```powershell
python verification/step04_three_story_multibay_frame.py
python -m verification.step04_three_story_multibay_frame
python -m unittest discover -s tests -p "test_step04_three_story_multibay_frame.py" -v
```

Outputs are written to
`results/verification/step04_three_story_multibay_frame/`.

## Step 05 - Floor horizontal-constraint and top mass-distribution sensitivity

Step 05 uses the Step 04 three-story, three-bay frame and compares four cases:

1. Tied floor-X DOFs with centered floor mass
2. Untied floor-X DOFs with centered floor mass
3. Tied floor-X DOFs with eccentric top-floor nodal mass distribution
4. Untied floor-X DOFs with eccentric top-floor nodal mass distribution

The same total 200 kN floor weight is retained in every case.  The eccentric
top-floor case moves the nodal mass centroid from `x = 9.0 m` to `x = 12.6 m`,
corresponding to `e/B = 0.20` for the 18 m floor width.  The four top-floor
mass fractions are `[0.07, 0.19, 0.31, 0.43]` from left to right; this is an
assumed sensitivity distribution.

This 2D eccentric case is not a model of 3D torsional eccentricity.  It only
tests whether asymmetric nodal mass distribution affects the planar frame
when floor horizontal DOFs are tied or untied.  The later 3D model is required
to study translation-torsion coupling due to separation of mass and stiffness
centers.

The static comparison applies a total 1 kN roof load equally to the four roof
nodes.  It reports each floor's nodal horizontal displacements, their spread,
and the effective roof stiffness.  The eigenanalysis reports the first three
periods, floor-average mode shapes, and within-floor modal displacement spread.

```powershell
python verification/step05_floor_constraint_mass_sensitivity.py
python -m verification.step05_floor_constraint_mass_sensitivity
python -m unittest discover -s tests -p "test_step05_floor_constraint_mass_sensitivity.py" -v
```

Outputs are written to
`results/verification/step05_floor_constraint_mass_sensitivity/`.

## Step 06 - Linear/PDelta gravity-preload sensitivity

Step 06 uses the Step 04 three-story, three-bay reference frame.  Each floor's
200 kN weight is divided equally among its four beam-column nodes and applied
in ten static increments.  The gravity pattern is then held constant before
the eigenvalue and small roof-load stiffness analyses.

Only the column geometric transformation changes between the two models:

- `Linear`: no second-order geometric-stiffness effect
- `PDelta`: OpenSees P-large-Delta coordinate transformation

Beams retain a `Linear` transformation in both models.  Gravity factors
`0.0, 0.5, 1.0, 1.5, 2.0` demonstrate that the transformations match without
compressive preload and that the PDelta period increase/stiffness reduction
grows with gravity load.  These factors are sensitivity cases; factor `1.0`
is the reference assumed floor weight.

The OpenSees PDelta transformation does not include P-small-delta effects.

```powershell
python verification/step06_linear_pdelta_sensitivity.py
python -m verification.step06_linear_pdelta_sensitivity
python -m unittest discover -s tests -p "test_step06_linear_pdelta_sensitivity.py" -v
```

Outputs are written to
`results/verification/step06_linear_pdelta_sensitivity/`.

## STEP08 - 추가 수치 검증

[한국어 검증 보강 보고서](STEP08_검증보강.md): 전 층 면내변형, 경간당 1/2/4/8분할 및 정적 16분할, 연속 강성 모드 추적, 유효 참여질량 검산. 바닥판 형식은 미정이며 완전 부착을 가정한다.

```powershell
python -m verification.step08_validation
```

결과: `results/verification/step08_diaphragm_inplane_sensitivity/validation/`.

## STEP09 - 질량편심 단독 민감도

[한국어 STEP09 설명](STEP09_설명.md): 강체 다이어프램 3D 기준 골조에서 균등 중력하중과 총질량을 유지하고, 질량을 +X 끝선·+Y 끝선·한 모서리에 단계적으로 집중한다.

```powershell
python -m verification.step09_mass_eccentricity
```

결과: `results/verification/step09_mass_eccentricity/`.

## STEP10 - 다이어프램 강성 × 질량배치 상호작용

[STEP10 설명 및 결과분석](STEP10_설명_및_결과분석.md): 3개 다이어프램 × 12개 질량배치, 총 36조합의 고유주기·층별 위치변위·모드 결합과 개별 효과/상호작용을 비교한다. 변위는 승인된 단위가속도 정적 관성력 기준이다.

```powershell
python -m verification.step10_diaphragm_mass_interaction
python -m verification.step10_validation
```

결과: `results/verification/step10_diaphragm_mass_interaction/`.

## STEP11 - STEP10 결론의 강건성 및 최종 해석 범위

[STEP11 강건성·결론 보고서](STEP11_강건성_및_최종결론.md): STEP10의 36조합 결과를 기준으로 대표 질량배치 U/X99/Y99/C99에 대해 셸 메시 4/8/16분할, 가정 부재 단면특성 공통 배율 0.75/1.00/1.25, 기둥 Linear/PDelta의 독립적 민감도를 비교한다. 52개 추가 조건을 해석한다.

```powershell
python -m verification.step11_robustness_conclusion
```

결과: `results/verification/step11_robustness_conclusion/`.
