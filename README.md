# My First Project

Git과 GitHub 사용법을 연습하면서, OpenSeesPy로 2D 1경간 3층 골조 모델을 만들어보는 프로젝트입니다.

## 프로젝트 소개

이 저장소는 Python 코드 작성, Git 커밋, GitHub 업로드, 폴더 구조 정리, 해석 결과 시각화를 연습하기 위해 만들었습니다.

현재는 OpenSeesPy를 사용해 다음 두 가지 구조 모델을 비교합니다.

- 강체 다이어프램을 적용한 2D 탄성 골조 모델
- 강체 다이어프램을 적용하지 않은 2D 탄성 골조 모델

## 사용한 기술

- Python
- OpenSeesPy
- opsvis
- matplotlib
- Git / GitHub

## 폴더 구조

```text
my-first-project/
├─ models/
│  ├─ frame_with_rigid_diaphragm.py
│  └─ frame_without_rigid_diaphragm.py
└─ results/
   ├─ figures/
   └─ animations/
```

## 실행 방법

강체 다이어프램 적용 모델:

```bash
python models/frame_with_rigid_diaphragm.py
```

강체 다이어프램 미적용 모델:

```bash
python models/frame_without_rigid_diaphragm.py
```

실행하면 `results/figures/` 폴더에 모델 형상과 모드 형상 PNG가 저장되고, `results/animations/` 폴더에 모드 형상 GIF가 저장됩니다.

## 주요 기능

- 2D 1경간 3층 탄성 골조 모델 생성
- 중력하중 정적해석
- 고유치 해석을 통한 모드 주기 계산
- 모드 형상 출력
- opsvis와 matplotlib을 이용한 모델 및 모드 형상 시각화
- 강체 다이어프램 적용/미적용 모델 비교

## 만든 이유

GitHub 저장소 관리 방법을 익히고, 구조해석 모델링 코드를 체계적으로 정리하는 연습을 하기 위해 만들었습니다.

## 앞으로 개선할 점

- `.gitignore`를 추가해 자동 생성 결과 파일 관리하기
- 횡하중 해석 추가하기
- 층간변위 계산 추가하기
- pushover 해석으로 확장하기
- 모델 입력값을 별도 설정 파일로 분리하기
