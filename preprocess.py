# preprocess.py
import numpy as np
from scipy.signal import resample_poly, butter, lfilter
import librosa # Hoặc có thể dùng scipy.io.wavfile để đọc wav

def load_audio_mat(filepath):
    """Tải dữ liệu audio từ file .mat (cần scipy.io.loadmat)"""
    from scipy.io import loadmat
    try:
        mat_data = loadmat(filepath)
        if 'y' in mat_data and 'Fs' in mat_data:
            y = mat_data['y'].astype(np.float32).ravel() # Đảm bảo là vector 1D float
            fs = int(mat_data['Fs'][0][0])
            return y, fs
        else:
            print(f"Warning: 'y' or 'Fs' not found in {filepath}")
            return None, None
    except Exception as e:
        print(f"Error loading mat file {filepath}: {e}")
        return None, None

def preprocess_audio_signal(y_in, fs_in, apply_lpf, target_fs, lpf_cutoff_hz):
    """Tiền xử lý tín hiệu audio."""
    # 1. Chuyển sang Mono (nếu y_in là stereo)
    if y_in.ndim > 1 and y_in.shape[1] > 1:
        y_mono = y_in[:, 0] # Lấy kênh đầu
    else:
        y_mono = y_in.ravel()

    # 2. Resample
    y_current = y_mono
    fs_current = fs_in
    if fs_in != target_fs:
        try:
            # Sử dụng resample_poly cho chất lượng tốt hơn là resample đơn giản
            # Hoặc librosa.resample
            y_current = librosa.resample(y_mono, orig_sr=fs_in, target_sr=target_fs)
            fs_current = target_fs
        except Exception as e:
            print(f"Warning: Resampling failed: {e}. Using original Fs {fs_in}.")
            # fs_current vẫn là fs_in

    # 3. Low-pass Filter
    if apply_lpf:
        if fs_current / 2 <= lpf_cutoff_hz:
            print(f"Warning: LPF cutoff {lpf_cutoff_hz}Hz too high for Fs {fs_current}Hz. Skipping LPF.")
        else:
            try:
                nyquist = 0.5 * fs_current
                normal_cutoff = lpf_cutoff_hz / nyquist
                b, a = butter(5, normal_cutoff, btype='low', analog=False) # Butterworth order 5
                y_current = lfilter(b, a, y_current)
            except Exception as e:
                print(f"Warning: Lowpass filter failed: {e}. Skipping LPF.")
    
    return y_current.astype(np.float32), fs_current