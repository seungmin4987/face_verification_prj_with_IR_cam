# face_verification_prj_with_IR_cam

# 얼굴 인증 시스템 프로그램

## 📖 개요
이 프로그램은 OpenCV와 IR 카메라를 활용해 얼굴을 등록하고, 등록된 얼굴과 실시간 촬영된 얼굴 데이터를 비교하여 인증을 수행한다. 주요 기술로는 LBP(Local Binary Pattern)를 이용한 얼굴 특징 추출, HOG(Histogram of Oriented Gradients)를 이용한 눈 상태 분석, 코사인 유사도를 기반으로 한 얼굴 인증 판단이 포함된다.

---

## 함수 역할 정리

### 메인 기능
- **`displayMenu`**: 사용자 인터페이스를 출력해 메뉴를 선택할 수 있도록 함.
- **`main`**: 프로그램의 주요 실행 흐름을 제어하며, 메뉴 선택에 따라 작업 수행.

---

### 얼굴 등록 관련
- **`register_face`**: IR 카메라로 얼굴을 캡처하고 안정화된 데이터를 등록.
- **`calculate_histogram`**: 얼굴 랜드마크 위치를 기준으로 LBP 히스토그램을 생성.
- **`make_lbp_img`**: 입력 이미지를 LBP(Local Binary Pattern) 이미지로 변환.

---

### 얼굴 인증 관련
- **`safe_verification_total_task`**: 얼굴 인증 전체 프로세스를 제어, 눈 상태 확인 및 인증 수행.
- **`face_verification`**: 실시간 얼굴 데이터를 등록된 히스토그램과 비교해 코사인 유사도를 계산하고 인증 여부 판단.

---

### 눈 상태 확인 및 전처리
- **`is_both_eye_opened`**: 얼굴 이미지에서 양쪽 눈이 열려 있는지 HOG 데이터를 기반으로 분석.
- **`is_eye_opened`**: 특정 눈 영역의 HOG 데이터를 분석해 눈이 열려 있는지 확인.
- **`crop_eye_region`**: 랜드마크 좌표를 기준으로 눈 영역을 잘라내고 전처리.
- **`increaseContrast`**: 이미지 대비를 조정해 밝고 어두운 영역의 차이를 강화.
- **`rotateImage`**: 얼굴 이미지를 수평 정렬하도록 회전.

---

### 데이터 분석 및 저장
- **`calculate_vector`**: 이미지의 모든 픽셀에 대해 그래디언트 벡터(크기와 방향) 계산.
- **`append_histogram_to_csv`**: HOG 데이터를 CSV 파일로 저장해 디버깅 및 분석에 활용.

---

### 데이터 유효성 확인
- **`is_bright_frame`**: IR 카메라 데이터의 밝기를 확인해 유효한 프레임만 필터링.

