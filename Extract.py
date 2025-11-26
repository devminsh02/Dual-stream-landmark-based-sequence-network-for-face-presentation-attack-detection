import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, TimeDistributed, Flatten, Masking, Dense, Dropout, Bidirectional, LSTM, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report

############################################
# 1. CSV 파일 경로 설정 (face와 attack 폴더)
############################################
def get_csv_files():
    base_dir = r"path"
    face_dir = os.path.join(base_dir, "face")   # face 폴더
    attack_dir = os.path.join(base_dir, "attack")  # attack 폴더
    face_files = sorted(glob.glob(os.path.join(face_dir, "*.csv")))
    attack_files = sorted(glob.glob(os.path.join(attack_dir, "*.csv")))
    return face_files, attack_files

############################################
# 2. CSV 파일 로딩 및 전처리 (시퀀스 단위)
############################################
def load_valid_sequence(file_path, skiprows=2):
    """
    pandas를 사용해 CSV 파일을 읽은 후, skiprows 이후부터
    완전히 빈 행이 나오기 전까지의 데이터를 사용
    """
    try:
        df = pd.read_csv(file_path, header=None, skiprows=skiprows)
    except Exception as e:
        print(f"파일 로드 실패: {file_path}, 에러: {e}")
        return None
    valid_rows = []
    for idx, row in df.iterrows():
        if row.isnull().all():
            break  # 빈 행 발견 시 중단
        valid_rows.append(row)
    if not valid_rows:
        print(f"빈 파일 또는 유효한 데이터 없음: {file_path}")
        return None
    valid_df = pd.DataFrame(valid_rows)
    data = valid_df.values.astype(np.float32)
    return data

def load_sequences_from_csv(file_paths):
    """
    각 CSV 파일을 읽어, 3번째 행부터 시작하는 모든 유효 행을 하나의 시퀀스로 취급
    각 행의 열 수가 짝수라면, (num_frames, total_columns//2, 2)로 재구성
    """
    sequences = []
    for file_path in file_paths:
        data = load_valid_sequence(file_path, skiprows=2)
        if data is None:
            continue
        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if np.all(data[0] == 0):
            print(f"첫 행이 0인 파일 건너뜀: {file_path}")
            continue
        if data.shape[1] % 2 != 0:
            print(f"파일 {file_path}의 열 개수({data.shape[1]})가 짝수가 아님. 건너뜀.")
            continue
        num_landmarks = data.shape[1] // 2
        num_frames = data.shape[0]
        data = data.reshape(num_frames, num_landmarks, 2)
        sequences.append(data)
    return sequences

############################################
# 3. 데이터 준비: 시퀀스 로드, 분할, 정규화, 패딩 (행 단위 샘플)
############################################
def prepare_data():
    face_files, attack_files = get_csv_files()
    face_sequences = load_sequences_from_csv(face_files)
    attack_sequences = load_sequences_from_csv(attack_files)
    print(f"Face 시퀀스 개수: {len(face_sequences)}, Attack 시퀀스 개수: {len(attack_sequences)}")
    
    X_all = face_sequences + attack_sequences
    y_all = [1] * len(face_sequences) + [0] * len(attack_sequences)
    
    X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # 각 시퀀스의 shape: (num_frames, num_landmarks, 2)
    max_len = max(max(seq.shape[0] for seq in X_train_seq),
                  max(seq.shape[0] for seq in X_test_seq))
    num_landmarks = X_train_seq[0].shape[1]
    feat_per_landmark = X_train_seq[0].shape[2]
    
    # 정규화: 훈련 시퀀스 모든 프레임을 평탄화하여 StandardScaler 학습
    train_frames = [seq.reshape(seq.shape[0], -1) for seq in X_train_seq]
    train_concat = np.vstack(train_frames)
    scaler = StandardScaler()
    scaler.fit(train_concat)
    
    def scale_sequence(seq):
        orig_shape = seq.shape  # (num_frames, num_landmarks, 2)
        seq_flat = seq.reshape(seq.shape[0], -1)
        seq_scaled = scaler.transform(seq_flat)
        return seq_scaled.reshape(orig_shape)
    
    X_train_scaled = [scale_sequence(seq) for seq in X_train_seq]
    X_test_scaled  = [scale_sequence(seq) for seq in X_test_seq]
    print("정규화 후 샘플 shape:", X_train_scaled[0].shape)
    
    # 패딩: 시퀀스 길이가 다를 경우, max_len에 맞춰 패딩 (시간 축)
    def pad_sequence(seq, max_len, pad_value=0.0):
        num_frames = seq.shape[0]
        if num_frames < max_len:
            pad_width = max_len - num_frames
            padding = np.full((pad_width, num_landmarks, feat_per_landmark), pad_value, dtype=np.float32)
            seq_padded = np.concatenate([seq, padding], axis=0)
        else:
            seq_padded = seq[:max_len]
        return seq_padded
    
    X_train_pad = np.array([pad_sequence(seq, max_len, pad_value=0.0) for seq in X_train_scaled])
    X_test_pad  = np.array([pad_sequence(seq, max_len, pad_value=0.0) for seq in X_test_scaled])
    
    print("패딩 후 학습 데이터 shape:", X_train_pad.shape)  # (num_train, max_len, num_landmarks, 2)
    print("패딩 후 테스트 데이터 shape:", X_test_pad.shape)
    
    return X_train_pad, X_test_pad, np.array(y_train), np.array(y_test), scaler, max_len, num_landmarks, feat_per_landmark, 0.0

############################################
# 4. 모델 구성: 두 Branch 병렬 처리 후 병합 (프레임별 vs 시퀀스 기반)
############################################
def build_model(max_len, num_landmarks, feat_per_landmark, pad_value):
    input_shape = (max_len, num_landmarks, feat_per_landmark)
    inputs = Input(shape=input_shape, name="input_sequence")
    
    # Branch 1: 프레임별 처리 (각 행 Dense 처리 후 평균 pooling)
    branch1 = TimeDistributed(Flatten(), name="td_flatten_branch1")(inputs)
    branch1 = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                              name="td_dense_branch1")(branch1)
    branch1 = GlobalAveragePooling1D(name="global_avg_pool_branch1")(branch1)
    
    # Branch 2: 시퀀스 전체를 BiLSTM으로 처리
    branch2 = TimeDistributed(Flatten(), name="td_flatten_branch2")(inputs)
    branch2 = Masking(mask_value=pad_value, name="masking_branch2")(branch2)
    branch2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.5,
                                  kernel_regularizer=regularizers.l2(0.01)),
                             name="bilstm1_branch2")(branch2)
    branch2 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.5, recurrent_dropout=0.5,
                                  kernel_regularizer=regularizers.l2(0.01)),
                             name="bilstm2_branch2")(branch2)
    
    # 병합: 두 branch의 출력 concat
    merged = Concatenate(name="concat_branches")([branch1, branch2])
    merged = Dropout(0.5, name="dropout_merged")(merged)
    hidden = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="dense_hidden")(merged)
    output = Dense(1, activation='sigmoid', name="output")(hidden)
    
    model = Model(inputs=inputs, outputs=output, name="MultiBranch_Model")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

############################################
# 5. 모델 학습 및 체크포인트 저장
############################################
def train_model(model, X_train, y_train, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "best_model.h5")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,         # 에포크 10으로 제한
        batch_size=16,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    return history

############################################
# 6. 평가 및 평가지표 계산
############################################
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 세트 정확도: {accuracy * 100:.2f}%")
    
    y_pred = (model.predict(X_test) >= 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Attack(0)", "Face(1)"], digits=4)
    
    print("Confusion Matrix:")
    print(cm)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(report)
    return cm, f1, report

############################################
# 7. 결과 시각화 및 저장
############################################
def save_results(history, cm, report, save_dir):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()
    
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Attack(0)", "Face(1)"], rotation=45)
    plt.yticks(tick_marks, ["Attack(0)", "Face(1)"])
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normalized_confusion_matrix.png"))
    plt.close()
    
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

############################################
# 8. 새로운 CSV 파일 예측 (단일 행 샘플)
############################################
def predict_new_csv(model, scaler, max_landmarks, feat_per_sample, pad_value, new_csv_path):
    if os.path.exists(new_csv_path):
        # 새로운 CSV 파일의 첫 행만 사용 (skiprows=2)
        sample = load_valid_sequence(new_csv_path, skiprows=2)
        if sample is None:
            print("새로운 CSV 파일에서 유효한 데이터를 찾지 못했습니다.")
            return
        sample = sample[0]  # 첫 행 선택 (1차원 array)
        if sample.size % 2 != 0:
            print(f"새로운 CSV 파일의 첫 행 열 개수({sample.size})가 짝수가 아님.")
            return
        current_landmarks = sample.size // 2
        if current_landmarks != max_landmarks:
            print(f"새로운 CSV 파일의 랜드마크 수({current_landmarks})가 학습 데이터({max_landmarks})와 다름.")
            return
        sample = sample.reshape(max_landmarks, feat_per_sample)
        sample_flat = sample.reshape(-1)
        sample_scaled = scaler.transform([sample_flat])[0].reshape(sample.shape)
        X_new = np.expand_dims(sample_scaled, axis=0)  # (1, max_landmarks, 2)
        pred_prob = model.predict(X_new)[0,0]
        pred_class = 1 if pred_prob >= 0.5 else 0
        label_str = "Face" if pred_class == 1 else "Attack"
        print(f"새로운 샘플 예측 결과: {pred_prob:.3f}의 확률로 '{label_str}'으로 판단")
    else:
        print(f"예측용 CSV 파일이 존재하지 않습니다: {new_csv_path}")

############################################
# 9. 메인 
############################################
def main():
    save_dir = r"C:\Users\User\OneDrive\Desktop\L-TEST\results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 준비 (행 단위 샘플)
    X_train, X_test, y_train, y_test, scaler, max_len, num_landmarks, feat_per_sample, pad_value = prepare_data()
    
    # 모델 구성: 두 branch를 병렬로 학습 후 병합
    model = build_model(max_len, num_landmarks, feat_per_sample, pad_value)
    
    # 모델 학습
    history = train_model(model, X_train, y_train, save_dir)
    
    # 평가
    cm, f1, report = evaluate_model(model, X_test, y_test)
    
    # 결과 저장
    save_results(history, cm, report, save_dir)
    
    # 새로운 CSV 파일 예측 (경로 수정)
    new_csv_path = r"path"
    predict_new_csv(model, scaler, num_landmarks, feat_per_sample, pad_value, new_csv_path)

if __name__ == "__main__":
    main()
