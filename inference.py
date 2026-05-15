import os
import sys
import json
import pickle
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab_clean"
RESULTS_DIR = os.path.join(BASE_DIR, "results_final_model")

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    return y_trimmed, 16000

def extract_features(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    S = np.abs(librosa.stft(y))
    sc = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    sc_mean = np.mean(sc)
    sc_std = np.std(sc)
    
    sban = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    sban_mean = np.mean(sban)
    sban_std = np.std(sban)
    
    sroll = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    sroll_mean = np.mean(sroll)
    sroll_std = np.std(sroll)
    
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    if f0 is not None and len(f0) > 0:
        pitch_mean = np.mean(f0)
        pitch_std = np.std(f0)
        pitch_min = np.min(f0)
        pitch_max = np.max(f0)
    else:
        pitch_mean, pitch_std, pitch_min, pitch_max = 0.0, 0.0, 0.0, 0.0
        
    mfccs13 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs13_mean = np.mean(mfccs13, axis=1)
    
    delta_mfccs13 = librosa.feature.delta(mfccs13)
    delta_mfccs13_mean = np.mean(delta_mfccs13, axis=1)
    
    delta2_mfccs13 = librosa.feature.delta(mfccs13, order=2)
    delta2_mfccs13_mean = np.mean(delta2_mfccs13, axis=1)
    
    mfccs20 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs20_mean = np.mean(mfccs20, axis=1)
    
    features = np.concatenate([
        [duration, rms_mean, rms_std, zcr_mean, zcr_std, sc_mean, sc_std, 
         sban_mean, sban_std, sroll_mean, sroll_std, pitch_mean, pitch_std, pitch_min, pitch_max],
        mfccs13_mean,
        delta_mfccs13_mean,
        delta2_mfccs13_mean,
        mfccs20_mean
    ])
    
    return features.reshape(1, -1)

def load_objects():
    with open(os.path.join(RESULTS_DIR, "final_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, "final_model_extratrees.pkl"), "rb") as f:
        et = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, "final_model_svc.pkl"), "rb") as f:
        svc = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, "final_thresholds.json"), "r") as f:
        thresholds = json.load(f)
    with open(os.path.join(RESULTS_DIR, "final_metrics.json"), "r") as f:
        metrics = json.load(f)
        
    class_names = {
        0: 'belly pain',
        1: 'burping',
        2: 'discomfort',
        3: 'hungry',
        4: 'tired'
    }
    
    return scaler, et, svc, thresholds, metrics, class_names

def predict(file_path):
    print(f"Processing audio: {file_path}")
    y, sr = preprocess_audio(file_path)
    X = extract_features(y, sr)
    
    print("Loading models and thresholds...")
    scaler, et, svc, thresholds_dict, metrics, class_names = load_objects()
    
    num_classes = len(class_names)
    thresholds = np.array([thresholds_dict[str(i)] for i in range(num_classes)])
    
    et_w = metrics.get("best_et_weight", 0.7)
    svc_w = metrics.get("best_svc_weight", 0.3)
    
    print("Running inference...")
    X_scaled = scaler.transform(X)
    
    et_probs = et.predict_proba(X_scaled)[0]
    svc_probs = svc.predict_proba(X_scaled)[0]
    
    final_probs = et_w * et_probs + svc_w * svc_probs
    
    passed = []
    for i in range(num_classes):
        if final_probs[i] >= thresholds[i]:
            passed.append((i, final_probs[i]))
            
    if len(passed) == 0:
        final_class = np.argmax(final_probs)
        decision_type = "Fallback (Max Probability)"
    else:
        final_class = max(passed, key=lambda item: item[1])[0]
        decision_type = "Passed Threshold (Max Prob)"
        
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Predicted Class : {class_names[final_class].upper()} (Class {final_class})")
    print(f"Decision Logic  : {decision_type}")
    print("\n--- Decision Breakdown ---")
    for i in range(num_classes):
        name = class_names[i]
        prob = final_probs[i]
        th = thresholds[i]
        status = "PASSED" if prob >= th else "FAILED"
        
        # Determine base model probabilities
        et_p = et_probs[i]
        svc_p = svc_probs[i]
        
        marker = "--> " if i == final_class else "    "
        print(f"{marker}{name.capitalize():<12} | Prob: {prob:.4f} (ET: {et_p:.2f}, SVC: {svc_p:.2f}) | Thresh: {th:.2f} | {status}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: File not found -> {audio_path}")
        sys.exit(1)
        
    predict(audio_path)
