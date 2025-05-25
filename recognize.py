# recognize.py
import numpy as np
from collections import Counter
from preprocess import preprocess_audio_signal # Giả sử load_audio_mat được gọi bên ngoài
from fingerprint import create_fingerprints
# from config import Params # Sẽ được truyền vào

def match_and_score_signal(clip_hashed_fingerprints, db_hash_table, config_params, clip_duration_s):
    """So khớp và tính điểm."""
    # Lấy ngưỡng từ config_params
    # (Logic chọn ngưỡng linh hoạt dựa trên clip_duration_s)
    current_min_match_threshold = config_params.DEFAULT_MIN_MATCH_THRESHOLD
    thresholds_cfg = config_params.THRESHOLDS_FOR_CLIP_LENGTH
    if clip_duration_s <= thresholds_cfg["short_max_len"]:
        current_min_match_threshold = thresholds_cfg["short_thresh"]
    elif clip_duration_s <= thresholds_cfg["medium_max_len"]:
        current_min_match_threshold = thresholds_cfg["medium_thresh"]
    else:
        current_min_match_threshold = thresholds_cfg["long_thresh"]
    
    # Ngưỡng để yêu cầu thêm dữ liệu
    REQUEST_MORE_DATA_FACTOR = 0.6 
    threshold_for_more_data = int(np.ceil(current_min_match_threshold * REQUEST_MORE_DATA_FACTOR))
    threshold_for_more_data = max(2, threshold_for_more_data)
    if threshold_for_more_data >= current_min_match_threshold:
        threshold_for_more_data = current_min_match_threshold - 1
    if threshold_for_more_data < 1: threshold_for_more_data = 1

    # --- Matching ---
    ID_NOT_FOUND_OUTPUT = 0
    if clip_hashed_fingerprints.size == 0 or db_hash_table.size == 0:
        return ID_NOT_FOUND_OUTPUT, 0 # songID, actual_match_count

    match_list_offsets_ids = [] # list of (song_id_db, offset_frames)

    # Để tìm kiếm nhanh, DB_HASH_TABLE phải được sort theo hash_value (cột 0)
    # và chúng ta có thể dùng np.searchsorted hoặc một vòng lặp thông minh
    db_hashes = db_hash_table[:, 0]

    for i in range(clip_hashed_fingerprints.shape[0]):
        clip_hash_val = clip_hashed_fingerprints[i, 0]
        clip_t1_frame = clip_hashed_fingerprints[i, 1]
        
        # Tìm kiếm hiệu quả (nếu db_hashes đã sort)
        # left = np.searchsorted(db_hashes, clip_hash_val, side='left')
        # right = np.searchsorted(db_hashes, clip_hash_val, side='right')
        # matching_indices_in_db = np.arange(left, right)
        
        # Cách đơn giản hơn, nhưng chậm hơn cho DB lớn
        matching_indices_in_db = np.where(db_hashes == clip_hash_val)[0]

        if matching_indices_in_db.size > 0:
            for idx in matching_indices_in_db:
                db_t1_frame = db_hash_table[idx, 1]
                db_song_id = db_hash_table[idx, 2]
                offset = db_t1_frame - clip_t1_frame
                match_list_offsets_ids.append((db_song_id, offset))
    
    if not match_list_offsets_ids:
        return ID_NOT_FOUND_OUTPUT, 0

    # --- Scoring ---
    # Chuyển list of tuples thành mảng NumPy để xử lý dễ hơn
    match_array = np.array(match_list_offsets_ids, dtype=np.int32)
    
    unique_sids = np.unique(match_array[:, 0])
    best_song_id_candidate = ID_NOT_FOUND_OUTPUT
    max_overall_offset_count = 0

    for sid in unique_sids:
        offsets_for_this_sid = match_array[match_array[:, 0] == sid, 1]
        if len(offsets_for_this_sid) >= threshold_for_more_data: # Chỉ xét nếu có tiềm năng
            if len(offsets_for_this_sid) == 0: continue # Nên được xử lý bởi if trên
            
            # Tìm mode và count của mode
            offset_counts = Counter(offsets_for_this_sid)
            if not offset_counts: continue # Nếu rỗng
            
            most_common_tuple = offset_counts.most_common(1) # List of (value, count)
            if not most_common_tuple: continue

            current_max_count_for_song = most_common_tuple[0][1] # Lấy count

            if current_max_count_for_song > max_overall_offset_count:
                max_overall_offset_count = current_max_count_for_song
                best_song_id_candidate = sid
            # (Có thể thêm tie-breaking nếu muốn)
            
    # Quyết định cuối cùng
    songID_final = ID_NOT_FOUND_OUTPUT
    if max_overall_offset_count >= current_min_match_threshold:
        songID_final = best_song_id_candidate
    elif max_overall_offset_count >= threshold_for_more_data: # Nằm giữa ngưỡng more_data và ngưỡng chính
        songID_final = -1 # Yêu cầu thêm dữ liệu
        print(f"  (recognize.py) Requesting more data. Match count: {max_overall_offset_count} (Thresh_more: {threshold_for_more_data}, Thresh_main: {current_min_match_threshold}) for SID {best_song_id_candidate if best_song_id_candidate !=0 else 'N/A'}")


    # print(f"  (recognize.py debug) Clip: {clip_duration_s:.2f}s, ActualMatchCount: {max_overall_offset_count}, MainThresh: {current_min_match_threshold}, MoreDataThresh: {threshold_for_more_data}, Identified: {songID_final}")
    return songID_final, max_overall_offset_count


def main_shazam_recognizer(y_clip_input, fs_clip_original, db_hash_table, db_params_config):
    """Hàm chính để nhận dạng, tương tự mainabc123.m."""
    # 1. Tiền xử lý clip
    y_processed_clip, fs_processed_clip = preprocess_audio_signal(
        y_clip_input, fs_clip_original,
        db_params_config.APPLY_LPF, 
        db_params_config.TARGET_FS, 
        db_params_config.LPF_PASSBAND_HZ
    )
    if y_processed_clip.size == 0: return 0 # Không tìm thấy

    # 2. Tạo fingerprint cho clip
    # config cho create_fingerprints sẽ là db_params_config
    clip_hashed_fp = create_fingerprints(y_processed_clip, fs_processed_clip, db_params_config)

    # 3. So khớp và Scoring
    clip_duration_s = len(y_clip_input) / fs_clip_original
    song_id, _ = match_and_score_signal(clip_hashed_fp, db_hash_table, db_params_config, clip_duration_s)
    
    return song_id