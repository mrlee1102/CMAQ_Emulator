import joblib

# 저장된 스케일러 객체 로드 (StandardScaler)
scaler = joblib.load('/home/user/workdir/CMAQ_Emulator/main/o3_training/o3_prediction_origin_v2.pkl')

# StandardScaler는 보통 mean_와 scale_ (표준편차)를 멤버 변수로 갖습니다.
mean = scaler.mean_    # 예: numpy 배열
scale = scaler.scale_  # 예: numpy 배열

# 헤더 파일에 저장 (배열 형태로 정의)
with open("/home/user/workdir/CMAQ_Emulator/o3_model_compile/params/scale_params.h", "w") as f:
    f.write("#ifndef SCALE_PARAMS_H\n")
    f.write("#define SCALE_PARAMS_H\n\n")
    
    # 평균값을 배열로 저장
    f.write("static const float PREDICTION_MEAN[] = {")
    f.write(", ".join("{:.6f}f".format(val) for val in mean))
    f.write("};\n\n")
    
    # 스케일(표준편차)를 배열로 저장
    f.write("static const float PREDICTION_SCALE[] = {")
    f.write(", ".join("{:.6f}f".format(val) for val in scale))
    f.write("};\n\n")
    
    f.write("#endif // SCALE_PARAMS_H\n")
