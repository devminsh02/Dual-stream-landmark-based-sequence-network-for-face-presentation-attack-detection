import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

def create_kalman_filter(initial_value=0.0):
    """
    Kalman 필터 생성 함수.
    상태 벡터 차원은 2, 측정 차원은 1로 설정하여,
    measurementMatrix와 transitionMatrix, 노이즈 공분산 행렬을 초기화.
    """
    kalman = cv2.KalmanFilter(2, 1)  # 2 state variables, 1 measurement
    kalman.measurementMatrix = np.array([[1, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 0.03  # 프로세스 노이즈
    kalman.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.5  # 측정 노이즈
    kalman.statePre = np.array([[initial_value], [0]], np.float32)
    return kalman

# 1. 지정한 디렉토리 내에서 "printed"가 포함된 영상 파일 검색
video_dir = r"path" # 사용자가 지정한 영상 폴더 경로
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if "WIN_20250331_22_15_48_Pro" in f]

if not video_files:
    print("지정한 폴더에서 'printed'가 포함된 영상 파일을 찾지 못함.")
    exit()

# 2. CSV 파일 저장 경로 설정 (savepath) 및 폴더 생성
save_path = r"path"  # 원하는 저장 경로로 변경
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 3. MediaPipe FaceMesh 초기화 (얼굴 랜드마크 검출)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)

# 4. 영상 파일들을 순회하면서 처리
for video_path in video_files:
    print(f"\n영상 처리 시작: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # 5. 각 랜드마크(468개)마다 밝기와 색상 변화량을 위한 Kalman 필터 초기화
    kalman_filters_brightness = [create_kalman_filter(0) for _ in range(468)]
    kalman_filters_color = [create_kalman_filter(0) for _ in range(468)]
    
    # 6. CSV 저장을 위한 DataFrame 컬럼 초기화
    columns = []
    for i in range(468):
        columns.append(f"landmark_{i}_color_diff")
        columns.append(f"landmark_{i}_brightness_diff")
    results_df = pd.DataFrame(columns=columns)
    
    prev_colors = None
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 종료
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, c = frame.shape
        
        # 현재 프레임의 각 랜드마크 색상 값을 저장할 리스트
        current_colors = [None] * 468
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # 각 랜드마크의 좌표 기반 색상 값 추출
            for i, lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                if 0 <= x < w and 0 <= y < h:
                    current_colors[i] = frame[y, x]
                else:
                    current_colors[i] = None
            
            # 이전 프레임 데이터가 있다면 변화량 계산 후 Kalman 필터 적용
            if prev_colors is not None:
                frame_data = {}
                for idx in range(468):
                    if current_colors[idx] is not None and prev_colors[idx] is not None:
                        curr_color = np.array(current_colors[idx], dtype=np.float32)
                        prev_color = np.array(prev_colors[idx], dtype=np.float32)
                        # 색상 차이: 두 벡터 간 유클리드 거리
                        raw_color_diff = np.linalg.norm(curr_color - prev_color)
                        # 밝기 차이: 색상 평균값의 차이
                        raw_brightness_diff = abs(np.mean(curr_color) - np.mean(prev_color))
                        
                        # Kalman 필터 업데이트 및 예측 (밝기)
                        measurement_brightness = np.array([[raw_brightness_diff]], np.float32)
                        kalman_filters_brightness[idx].correct(measurement_brightness)
                        filtered_brightness = kalman_filters_brightness[idx].predict()[0, 0]
                        
                        # Kalman 필터 업데이트 및 예측 (색상)
                        measurement_color = np.array([[raw_color_diff]], np.float32)
                        kalman_filters_color[idx].correct(measurement_color)
                        filtered_color = kalman_filters_color[idx].predict()[0, 0]
                        
                        frame_data[f"landmark_{idx}_color_diff"] = filtered_color
                        frame_data[f"landmark_{idx}_brightness_diff"] = filtered_brightness
                    else:
                        frame_data[f"landmark_{idx}_color_diff"] = np.nan
                        frame_data[f"landmark_{idx}_brightness_diff"] = np.nan
                
                results_df.loc[frame_count] = frame_data
                frame_count += 1
                
                # 진행상황 텍스트 표시 (옵션)
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 도트(랜드마크) 시각화 코드는 제거 (영상에 표시하지 않음)
            
            # 현재 프레임의 색상 데이터를 이전 프레임 데이터로 업데이트
            prev_colors = current_colors.copy()
        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Video Processing", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 누르면 종료
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 7. 영상 처리 완료 후, 영상 파일명 기반 CSV 파일 저장
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_landmark_data_{timestamp}.csv"
    # save_path 내에 CSV 파일 저장
    full_output_path = os.path.join(save_path, output_filename)
    results_df.to_csv(full_output_path, index=False)
    print(f"CSV 파일 저장됨: {full_output_path} (총 {frame_count} 프레임 처리)")
