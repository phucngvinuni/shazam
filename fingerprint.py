# fingerprint.py
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import maximum_filter, minimum_filter, binary_erosion 
import config # Giả sử bạn có một file config.py chứa các tham số cấu hình
# binary_erosion có thể dùng để tìm local maxima nếu kết hợp đúng cách,
# nhưng logic so sánh lân cận trực tiếp có thể dễ hiểu hơn ban đầu.
BITS_F = config.BITS_F_QUANT if hasattr(config, 'BITS_F_QUANT') else 10
BITS_DT = config.BITS_DT_QUANT if hasattr(config, 'BITS_DT_QUANT') else 9
def find_peaks_constellation(log_S_mag, T_vec, config):
    """Tìm các peak trên spectrogram theo logic tương tự MATLAB."""
    gs = config.GS_PEAK_FIND
    peaks_per_sec = config.PEAKS_PER_SEC

    if log_S_mag.size == 0 or T_vec.size < 1:
        return np.array([], dtype=bool).reshape(log_S_mag.shape)

    # Cách 1: Dùng maximum_filter (hiệu quả hơn vòng lặp)
    # Tạo một footprint cho bộ lọc cực đại cục bộ
    footprint = np.ones((gs, gs), dtype=bool)
    footprint[gs//2, gs//2] = False # Không so sánh với chính nó
    
    local_max = maximum_filter(log_S_mag, footprint=footprint, mode='constant', cval=-np.inf)
    peak_candidates_mask = (log_S_mag > local_max)

    # Thresholding
    peak_values = log_S_mag[peak_candidates_mask]
    final_peak_mask = np.zeros_like(log_S_mag, dtype=bool)

    if T_vec.size >= 2:
        signal_duration_s = T_vec[-1] - T_vec[0]
        if signal_duration_s <= 0 and len(T_vec) > 1: # Fallback
             signal_duration_s = (len(T_vec) -1) * np.median(np.diff(T_vec))
             if np.isnan(signal_duration_s) or signal_duration_s <=0 : signal_duration_s = 0.001
        elif len(T_vec) == 1: signal_duration_s = 0.032 # Default for single frame
    elif T_vec.size == 1:
        signal_duration_s = 0.032 
    else: # T_vec rỗng
        signal_duration_s = 0


    desired_num_peaks = int(np.ceil(signal_duration_s * peaks_per_sec))
    if desired_num_peaks > 0 and peak_values.size > 0:
        sorted_peak_values = np.sort(peak_values)[::-1] # Sắp xếp giảm dần
        num_peaks_to_take = min(desired_num_peaks, len(sorted_peak_values))
        if num_peaks_to_take > 0:
            threshold_val = sorted_peak_values[num_peaks_to_take - 1]
            final_peak_mask = (log_S_mag >= threshold_val) & peak_candidates_mask
            
    return final_peak_mask


def generate_hashes(peak_mask, F_vector, T_vector, config):
    """
    Tạo các cặp peak, lượng tử hóa đặc trưng (f_Hz, f_Hz, dt_frames), và hash chúng.
    Output: hashed_fingerprints là [hash_value, anchor_time_frame_index]
    """
    deltaF_bins_cfg = config.DELTA_F_BINS # Giả sử đây là số bin để mở rộng tìm kiếm
                                        # không phải là bước lượng tử hóa tần số Hz
    deltaTL_frames_cfg = config.DELTA_TL_FRAMES
    deltaTU_frames_cfg = config.DELTA_TU_FRAMES # Dùng để xác định range_dt_for_scaling
    fanOut_cfg = config.FAN_OUT

    # --- Tham số lượng tử hóa cho các thành phần của fingerprint THÔ ---
    # Bước lượng tử hóa cho tần số (Hz). Ví dụ: mỗi 10Hz là một mức.
    FREQ_HZ_QUANTIZATION_STEP = getattr(config, 'FREQ_HZ_QUANT_STEP', 10) 
    # Bước lượng tử hóa cho delta time (số frame). Ví dụ: mỗi 2 frame là một mức.
    DT_FRAMES_QUANTIZATION_STEP = getattr(config, 'DT_FRAMES_QUANT_STEP', 2) 

    # --- Tham số lượng tử hóa cho các thành phần ĐỂ TẠO HASH ---
    # Số bit này sẽ xác định dải giá trị của các thành phần sau khi lượng tử hóa lần 2 (cho hash)
    BITS_F_FOR_HASH = getattr(config, 'BITS_F_QUANT', 8) 
    BITS_DT_FOR_HASH = getattr(config, 'BITS_DT_QUANT', 7)
    
    MAX_VAL_F_HASH_QUANT = 2**BITS_F_FOR_HASH - 1
    MAX_VAL_DT_HASH_QUANT = 2**BITS_DT_FOR_HASH - 1
    
    hashed_fingerprints_list = []
    if np.sum(peak_mask) == 0: return np.array([])

    anchor_freq_indices, anchor_time_indices = np.where(peak_mask)
    if anchor_time_indices.size == 0: return np.array([])
    
    # Sắp xếp anchors theo thời gian (tùy chọn)
    sorted_indices = np.argsort(anchor_time_indices)
    anchor_time_indices = anchor_time_indices[sorted_indices]
    anchor_freq_indices = anchor_freq_indices[sorted_indices]

    for i in range(len(anchor_time_indices)):
        f1_idx_anchor = anchor_freq_indices[i]
        t1_idx_anchor = anchor_time_indices[i]

        # Lấy tần số Hz thực tế của anchor peak
        f1_hz_anchor = F_vector[f1_idx_anchor]

        # Xác định target zone
        t_min_target_frame = t1_idx_anchor + deltaTL_frames_cfg
        t_max_target_frame = min(len(T_vector) - 1, t1_idx_anchor + deltaTU_frames_cfg)
        if t_min_target_frame > len(T_vector) - 1 or t_min_target_frame > t_max_target_frame:
            continue

        # Target zone tần số vẫn dựa trên BIN indices để tìm kiếm lân cận
        f_min_target_bin = max(0, f1_idx_anchor - deltaF_bins_cfg)
        f_max_target_bin = min(len(F_vector) - 1, f1_idx_anchor + deltaF_bins_cfg)
        
        if f_min_target_bin > f_max_target_bin: continue

        target_zone_peaks_mask = peak_mask[f_min_target_bin : f_max_target_bin + 1, 
                                           t_min_target_frame : t_max_target_frame + 1]
        
        target_f_sub_indices, target_t_sub_indices = np.where(target_zone_peaks_mask)
        
        num_targets_in_zone = len(target_f_sub_indices)
        actual_fanOut_to_use = min(num_targets_in_zone, fanOut_cfg)

        if actual_fanOut_to_use > 0:
            # (Tùy chọn: sort target theo độ mạnh trước khi chọn)
            # Lấy các target được chọn
            for k_pair in range(actual_fanOut_to_use):
                f2_idx_target_in_zone = target_f_sub_indices[k_pair]
                t2_idx_target_in_zone = target_t_sub_indices[k_pair]

                # Chuyển về chỉ số của spectrogram đầy đủ
                f2_idx_target_full = f2_idx_target_in_zone + f_min_target_bin
                t2_idx_target_full = t2_idx_target_in_zone + t_min_target_frame
                
                # Lấy tần số Hz thực tế của target peak
                f2_hz_target = F_vector[f2_idx_target_full]
                dt_frames_diff = t2_idx_target_full - t1_idx_anchor

                # --- Lượng tử hóa các thành phần THÔ (để so sánh UCRF nếu cần) ---
                # Đây là các giá trị sẽ được dùng để so sánh tính duy nhất của raw_fp
                f1_hz_quant_raw = int(round(f1_hz_anchor / FREQ_HZ_QUANTIZATION_STEP))
                f2_hz_quant_raw = int(round(f2_hz_target / FREQ_HZ_QUANTIZATION_STEP))
                dt_frames_quant_raw = int(round(dt_frames_diff / DT_FRAMES_QUANTIZATION_STEP))
                # raw_fingerprint_tuple = (f1_hz_quant_raw, f2_hz_quant_raw, dt_frames_quant_raw) 
                # (Không dùng trực tiếp ở đây, nhưng đây là ý tưởng cho UCRF)

                # --- Lượng tử hóa các thành phần cho HASHING ---
                # Mục tiêu là map về dải [0, MAX_VAL_F_HASH_QUANT] và [0, MAX_VAL_DT_HASH_QUANT]
                # Giả sử F_vector[0] là fmin (có thể là 0Hz) và F_vector[-1] là fmax (Fs/2)
                # Và dt_frames_diff nằm trong [deltaTL_frames_cfg, deltaTU_frames_cfg]

                # Lượng tử hóa tần số Hz cho hash
                # Scale f_hz từ [F_vector[0], F_vector[-1]] về [0, MAX_VAL_F_HASH_QUANT]
                # Cần biết dải tần số tối đa thực sự (Fs_processed / 2)
                fs_proc = config.TARGET_FS # Lấy từ config vì F_vector có thể không bao phủ hết nếu fmin, fmax được dùng trong CQT/Mel
                fmax_hz = fs_proc / 2.0
                
                f1_hash_scaled = np.floor(f1_hz_anchor * (MAX_VAL_F_HASH_QUANT + 1) / fmax_hz) if fmax_hz > 0 else 0
                f1_hash_quant = int(min(max(f1_hash_scaled, 0), MAX_VAL_F_HASH_QUANT))

                f2_hash_scaled = np.floor(f2_hz_target * (MAX_VAL_F_HASH_QUANT + 1) / fmax_hz) if fmax_hz > 0 else 0
                f2_hash_quant = int(min(max(f2_hash_scaled, 0), MAX_VAL_F_HASH_QUANT))
                
                # Lượng tử hóa dt_frames cho hash
                # Scale dt_frames_diff từ [deltaTL_cfg, deltaTU_cfg] về [0, MAX_VAL_DT_HASH_QUANT]
                dt_shifted_for_hash = dt_frames_diff - deltaTL_frames_cfg
                range_dt_for_hash = deltaTU_frames_cfg - deltaTL_frames_cfg
                if range_dt_for_hash < 0: range_dt_for_hash = 0 # Tránh lỗi nếu TL > TU
                
                if range_dt_for_hash == 0: 
                    td_hash_scaled = 0
                else: 
                    td_hash_scaled = np.floor(dt_shifted_for_hash * (MAX_VAL_DT_HASH_QUANT + 1) / (range_dt_for_hash + 1)) # +1 để bao gồm điểm cuối
                
                td_hash_quant = int(min(max(td_hash_scaled, 0), MAX_VAL_DT_HASH_QUANT))   

                # Tạo Hash Value
                hash_value = (f1_hash_quant << (BITS_F + BITS_DT)) + \
                             (f2_hash_quant << BITS_DT) + \
                             td_hash_quant
                
                hashed_fingerprints_list.append((hash_value, t1_idx_anchor)) # Lưu hash và thời gian của anchor
    
    if not hashed_fingerprints_list: 
        return np.array([]) # Trả về mảng rỗng nếu không có gì
    return np.array(hashed_fingerprints_list, dtype=np.int32)


def create_fingerprints(y_processed, fs_processed, config):
    """Tạo fingerprint đã hash cho tín hiệu."""
    window_samples = int(round(config.WINDOW_MS / 1000 * fs_processed))
    if window_samples < 2: return np.array([])
    
    noverlap_samples = int(round(window_samples * config.OVERLAP_RATIO))
    nfft_points = 1 << (window_samples - 1).bit_length() # next_power_of_2(window_samples)

    if window_samples <= 0 or noverlap_samples < 0 or noverlap_samples >= window_samples or len(y_processed) < window_samples:
        print("Warning: Invalid spectrogram params or short signal for fingerprinting.")
        return np.array([])

    try:
        F_vec, T_vec, S_complex = scipy_spectrogram(
            y_processed,
            fs=fs_processed,
            window='hann', # Hoặc hamming(window_samples, sym=False)
            nperseg=window_samples,
            noverlap=noverlap_samples,
            nfft=nfft_points,
            detrend=False, # Thường là False cho audio
            mode='complex'
        )
    except Exception as e:
        print(f"Error in spectrogram calculation: {e}")
        return np.array([])

    if S_complex.size == 0 or T_vec.size == 0 or F_vec.size == 0:
        print("Warning: Spectrogram output is empty.")
        return np.array([])
        
    log_S_mag = np.log10(np.abs(S_complex) + 1e-9) # Thêm epsilon nhỏ
    
    peak_mask = find_peaks_constellation(log_S_mag, T_vec, config)
    if np.sum(peak_mask) == 0: return np.array([])
        
    hashed_fingerprints = generate_hashes(peak_mask, F_vector=F_vec, T_vector=T_vec, config=config)
    return hashed_fingerprints