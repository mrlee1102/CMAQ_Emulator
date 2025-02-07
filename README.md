# CMAQ_Emulator
> *Cloning CMAQ Simulator based on **Conditional U-Net** architecture*

## Package List
- **requirements.txt**  
  - 필요한 패키지 목록. 다음 명령어로 설치합니다:
  ```bash
  pip install -r requirements.txt
  ```
## main contents structure
- **main (Critical Directory)**
  ```bash
    .
    └── main (critical directory)
      ├── resources
      │   ├── ctrl
      │   │   └── # Input data used in the learning phase
      │   └── geom
      │       └── # Geometric data required for grid mapping
      ├── datasets
      │   └── # Label data required for learning
      └── src
          └── # Trained model of conditional U-net model
  ```
1. **resources**
    - **ctrl**: 학습 단계에서 사용되는 입력 데이터
    - **geom**: 격자 매핑에 필요한 기하(geometry) 정보
2. **datasets**
    - 모델 학습에 필요한 **레이블(label) 데이터**
3. **src**
    - **Conditional U-Net 모델**이 훈련된 결과물(Trained model)
