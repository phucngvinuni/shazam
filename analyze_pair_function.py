# File: analyze_pair_function.py
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram # Đảm bảo import đúng
import os

# Import các hàm và class từ các file khác của bạn
from config import Params 
from preprocess import load_audio_mat, preprocess_audio_signal
from fingerprint import find_peaks_constellation 
# Bạn sẽ cần một hàm để trích xuất raw fingerprints (f1,f2,t1,dt) và một hàm để tính hash
# Dưới đây là các hàm giả định/cục bộ

def extract_raw_fingerprints_local_for_analysis(peak_mask, F_vector, T_vector, config):
    """Hàm con cục bộ để trích xuất raw fingerprints (f1_idx,f2_idx,t1_idx,dt_frames)."""
    # Lấy các tham số từ config mà hàm này cần
    deltaF_bins = getattr(config, 'DELTA_F_BINS_CONFIG', getattr(config, 'DELTA_F_BINS', 15))
    deltaTL_frames = getattr(config, 'DELTA_TL_FRAMES_CONFIG', getattr(config, 'DELTA_TL_FRAMES', 2))
    deltaTU_frames = getattr(config, 'DELTA_TU_FRAMES_CONFIG', getattr(config, 'DELTA_TU_FRAMES', 65))
    fanOut = getattr(config, 'FAN_OUT_CONFIG', getattr(config, 'FAN_OUT', 3))
    
    raw_fp_list = []
    if np.sum(peak_mask) == 0: return np.array([])
    anchor_freq_indices, anchor_time_indices = np.where(peak_mask)
    if anchor_time_indices.size == 0: return np.array([])
    
    sorted_indices = np.argsort(anchor_time_indices)
    anchor_time_indices = anchor_time_indices[sorted_indices]
    anchor_freq_indices = anchor_freq_indices[sorted_indices]

    for i in range(len(anchor_time_indices)):
        f1_idx=anchor_freq_indices[i]; t1_idx=anchor_time_indices[i]
        t_min=t1_idx+deltaTL_frames; t_max=min(len(T_vector)-1, t1_idx+deltaTU_frames)
        if t_min > len(T_vector)-1 or t_min > t_max: continue
        f_min=max(0, f1_idx-deltaF_bins); f_max=min(len(F_vector)-1, f1_idx+deltaF_bins)
        
        # Đảm bảo slice không rỗng
        if f_min > f_max or t_min > t_max: continue

        target_zone_peaks = peak_mask[f_min:f_max+1, t_min:t_max+1]
        target_f_sub_indices, target_t_sub_indices = np.where(target_zone_peaks)
        num_targets_in_zone=len(target_f_sub_indices); actual_fo=min(num_targets_in_zone,fanOut)
        if actual_fo > 0:
            for k in range(actual_fo):
                f2_full=target_f_sub_indices[k]+f_min; t2_full=target_t_sub_indices[k]+t_min
                dt=t2_full-t1_idx
                raw_fp_list.append([f1_idx, f2_full, t1_idx, dt])
    return np.array(raw_fp_list, dtype=np.int32) if raw_fp_list else np.array([])

def calculate_single_hash_value_local_for_analysis(f1_idx, f2_idx, dt_frames, config, N_ACTUAL_FREQ_BINS):
    """Hàm con cục bộ tính hash value."""
    deltaTL_frames = getattr(config, 'DELTA_TL_FRAMES_CONFIG', getattr(config, 'DELTA_TL_FRAMES', 2))
    deltaTU_frames = getattr(config, 'DELTA_TU_FRAMES_CONFIG', getattr(config, 'DELTA_TU_FRAMES', 65))
    BITS_F = getattr(config, 'BITS_F_QUANT', 8)
    BITS_DT = getattr(config, 'BITS_DT_QUANT', 7)
    MAX_VAL_F_QUANT = 2**BITS_F - 1
    MAX_VAL_DT_QUANT = 2**BITS_DT - 1

    f1_scaled = np.floor((f1_idx)*(MAX_VAL_F_QUANT+1)/N_ACTUAL_FREQ_BINS ) if N_ACTUAL_FREQ_BINS >0 else 0 # f_idx đã từ 0
    f1_quant = int(min(max(f1_scaled,0),MAX_VAL_F_QUANT))
    f2_scaled = np.floor((f2_idx)*(MAX_VAL_F_QUANT+1)/N_ACTUAL_FREQ_BINS ) if N_ACTUAL_FREQ_BINS >0 else 0
    f2_quant = int(min(max(f2_scaled,0),MAX_VAL_F_QUANT))
    dt_shifted = dt_frames - deltaTL_frames; range_dt = deltaTU_frames - deltaTL_frames
    if range_dt < 0: range_dt=0
    if range_dt == 0: td_scaled=0
    else: td_scaled=np.floor(dt_shifted*(MAX_VAL_DT_QUANT+1)/(range_dt+1) ) if range_dt >=0 else 0 # Sửa range_dt > 0 thành >=0
    td_quant = int(min(max(td_scaled,0),MAX_VAL_DT_QUANT))   
    hash_value = (f1_quant << (BITS_F + BITS_DT)) + (f2_quant << BITS_DT) + td_quant
    return hash_value

def run_pair_analysis(song_id1, song_id2, config_to_test, db_song_path_for_analysis):
    print(f"  Analyzing Pair (Function): Song {song_id1} vs Song {song_id2} with current config...")
    spectrograms_log_mag = {}; F_vectors = {}; T_vectors = {}; peak_masks = {}; raw_fingerprints_dict = {}

    for sid_loop in [song_id1, song_id2]:
        filepath = os.path.join(db_song_path_for_analysis, f"{sid_loop}.mat")
        y_orig, fs_orig = load_audio_mat(filepath)
        if y_orig is None: print(f"    Could not load song {sid_loop}. Analysis for this pair might fail."); return None, None, 0, 0

        y_proc, fs_proc = preprocess_audio_signal(y_orig, fs_orig,
                                                config_to_test.APPLY_LPF,
                                                config_to_test.TARGET_FS,
                                                config_to_test.LPF_PASSBAND_HZ)
        
        window_samples = int(round(config_to_test.WINDOW_MS / 1000 * fs_proc))
        noverlap_samples = int(round(window_samples * config_to_test.OVERLAP_RATIO))
        nfft_points = 1 << (window_samples - 1).bit_length()

        # --- THÊM DEBUG PRINT Ở ĐÂY ---
        print(f"    Debug Spectrogram Input for Song {sid_loop}:")
        print(f"      Fs_proc: {fs_proc}, Window Samples: {window_samples}, Noverlap: {noverlap_samples}")
        print(f"      NFFT: {nfft_points}, Length of y_proc: {len(y_proc)}")

        if window_samples <= 0 or noverlap_samples < 0 or noverlap_samples >= window_samples or len(y_proc) < window_samples:
            print(f"    ERROR: Skipping song {sid_loop} due to invalid spectrogram parameters or short signal.")
            return None, None, 0, 0 

        try:
            F, T, S_complex = scipy_spectrogram(
                y_proc, fs=fs_proc, window='hann',
                nperseg=window_samples, noverlap=noverlap_samples, nfft=nfft_points,
                mode='complex' )
        except Exception as e_spec:
             print(f"    ERROR in scipy_spectrogram for song {sid_loop}: {e_spec}")
             return None, None, 0, 0 

        if S_complex.size == 0 or T.size == 0 or F.size == 0 : 
            print(f"    Spectrogram S_complex, T or F is empty for song {sid_loop}.")
            return None, None, 0, 0
        
        spectrograms_log_mag[sid_loop] = np.log10(np.abs(S_complex) + 1e-9)
        F_vectors[sid_loop] = F; T_vectors[sid_loop] = T
        peak_masks[sid_loop] = find_peaks_constellation(spectrograms_log_mag[sid_loop], T, config_to_test)
        raw_fingerprints_dict[sid_loop] = extract_raw_fingerprints_local_for_analysis(peak_masks[sid_loop], F_vectors[sid_loop], T_vectors[sid_loop], config_to_test)
        
    if song_id1 not in raw_fingerprints_dict or song_id2 not in raw_fingerprints_dict or \
       raw_fingerprints_dict[song_id1].size == 0 or raw_fingerprints_dict[song_id2].size == 0:
        print(f"    Not enough fingerprint data for pair {song_id1}v{song_id2} to compare.")
        len1 = raw_fingerprints_dict.get(song_id1, np.array([])).shape[0]
        len2 = raw_fingerprints_dict.get(song_id2, np.array([])).shape[0]
        return 0, 0, len1, len2

    raw_fp1_list = raw_fingerprints_dict[song_id1]
    raw_fp2_list = raw_fingerprints_dict[song_id2]
    set_fp1_comparable = {tuple(fp) for fp in raw_fp1_list[:, [0, 1, 3]]} 
    set_fp2_comparable = {tuple(fp) for fp in raw_fp2_list[:, [0, 1, 3]]}
    ucrf_count = len(set_fp1_comparable.intersection(set_fp2_comparable))

    all_hashes_fp1 = set()
    if raw_fp1_list.size > 0:
        N_ACTUAL_FREQ_BINS_1 = len(F_vectors[song_id1])
        for fp_raw in raw_fp1_list:
            all_hashes_fp1.add(calculate_single_hash_value_local_for_analysis(fp_raw[0], fp_raw[1], fp_raw[3], config_to_test, N_ACTUAL_FREQ_BINS_1))
    all_hashes_fp2 = set()
    if raw_fp2_list.size > 0:
        N_ACTUAL_FREQ_BINS_2 = len(F_vectors[song_id2])
        for fp_raw in raw_fp2_list:
            all_hashes_fp2.add(calculate_single_hash_value_local_for_analysis(fp_raw[0], fp_raw[1], fp_raw[3], config_to_test, N_ACTUAL_FREQ_BINS_2))
    common_hash_count = len(all_hashes_fp1.intersection(all_hashes_fp2))
    
    if common_hash_count > ucrf_count:
        print(f"    Pair {song_id1}v{song_id2}: WARNING! Quantization collision ({common_hash_count}H vs {ucrf_count}UCRF)")
    elif common_hash_count < ucrf_count:
        print(f"    Pair {song_id1}v{song_id2}: INFO. Fewer common hashes {common_hash_count} than UCRF {ucrf_count}.")
    # else: print(f"    Pair {song_id1}v{song_id2}: Consistent hashing for UCRF.")

    return ucrf_count, common_hash_count, raw_fp1_list.shape[0], raw_fp2_list.shape[0]