# File: main_test_recognition.py
import numpy as np
import os
import random # Để tạo initialTime ngẫu nhiên
import time # Để đo thời gian (tùy chọn)
from config import Params # Import class Params từ config.py
from database import load_shazam_db # Hàm tải database
from preprocess import load_audio_mat # Hàm tải audio từ .mat
from recognize import main_shazam_recognizer # Hàm nhận dạng chính

def run_official_lab_test_simulation(songs_folder_path, num_songs_to_eval, db_file_to_load, max_clip_len_s_allowed):
    """
    Mô phỏng script test của MATLAB với logic tăng dần độ dài clip.
    """
    print(f"--- Running Official Lab Test Simulation (Python) ---")
    print(f"Loading database from: {db_file_to_load}")
    db_hashtable, db_params_config = load_shazam_db(db_file_to_load)

    if db_hashtable is None or db_params_config is None:
        print("Failed to load database. Exiting test.")
        return

    print(f"Testing on {num_songs_to_eval} songs from '{songs_folder_path}', max clip length: {max_clip_len_s_allowed}s.")

    # --- Setting initial times (Tương tự logic MATLAB) ---
    # Chúng ta sẽ không lưu y, Fs vào cache để mô phỏng việc load lại file mỗi lần trong vòng lặp while
    # Điều này sẽ chậm hơn nhưng giống với script mẫu của bạn hơn.
    # Nếu muốn nhanh hơn, bạn có thể cache như phiên bản trước.
    clip_actual_lengths_s = np.zeros(num_songs_to_eval)
    clip_initial_start_times_s = np.zeros(num_songs_to_eval) # Thời gian bắt đầu (giây)
    
    valid_song_indices = []
    print("Determining random initial start times for clips...")
    for i in range(num_songs_to_eval):
        song_idx_matlab_style = i + 1 # MATLAB index từ 1
        path_to_song_file = os.path.join(songs_folder_path, f"{song_idx_matlab_style}.mat")

        if not os.path.exists(path_to_song_file):
            print(f"  Warning: Song file {path_to_song_file} not found. Song {song_idx_matlab_style} will be skipped.")
            continue
        
        y_full, fs_current_song = load_audio_mat(path_to_song_file)
        if y_full is None:
            print(f"  Warning: Could not load song {song_idx_matlab_style}. Skipping.")
            continue
            
        clip_actual_lengths_s[i] = len(y_full) / fs_current_song
        
        # Logic chọn initialTime giống MATLAB: randi(round(clipLength(i)-10))
        # randi(N) trong MATLAB trả về số nguyên từ 1 đến N.
        # random.randint(a,b) trong Python trả về số nguyên từ a đến b (bao gồm cả hai).
        # max_val_for_randi = int(round(clip_actual_lengths_s[i] - max_clip_len_s_allowed))
        # Để giống hệt, nếu clipLength(i)-10 < 1, randi(1) sẽ trả về 1.
        
        # Điều chỉnh logic chọn initialTime để nó có ý nghĩa hơn một chút
        # và đảm bảo có thể cắt clip
        min_required_clip_for_start_calc = 1.0 # Cần ít nhất 1s ở cuối
        if clip_actual_lengths_s[i] < min_required_clip_for_start_calc + max_clip_len_s_allowed:
            # Nếu bài hát quá ngắn để có thể có một điểm bắt đầu ngẫu nhiên mà vẫn đảm bảo clip 10s
            # thì điểm bắt đầu sẽ là 0 (để lấy từ đầu bài hát)
            # Hoặc nếu bài hát ngắn hơn cả max_clip_len_s_allowed, điểm bắt đầu là 0
            start_range_max = 0
        else:
            start_range_max = int(round(clip_actual_lengths_s[i] - max_clip_len_s_allowed))

        if start_range_max < 0 : start_range_max = 0 # randi(0) sẽ lỗi, randi(1) trả về 1
                                                   # random.randint(0,0) trả về 0
        
        # initialTime trong MATLAB là giây, nhưng nó được dùng để nhân với Fs
        # Ở đây chúng ta cũng tính theo giây.
        if start_range_max == 0:
            clip_initial_start_times_s[i] = 0.0
        else:
            # random.uniform(0, start_range_max) cho số float
            clip_initial_start_times_s[i] = random.uniform(0, start_range_max)

        valid_song_indices.append(i) # Lưu chỉ số Python (0 đến N-1)
    
    if not valid_song_indices:
        print("No valid songs found to test.")
        return

    num_songs_to_actually_test = len(valid_song_indices)
    print(f"Finished pre-processing. Will test {num_songs_to_actually_test} songs.\n")

    # --- Testing algorithms with varying times ---
    identified_song_ids_final = np.zeros(num_songs_to_eval, dtype=int)
    time_taken_per_song_s = np.zeros(num_songs_to_eval, dtype=int) # Số giây clip đã dùng
    misidentified_log_output = []

    for k_test_loop_idx in range(num_songs_to_actually_test):
        python_song_idx = valid_song_indices[k_test_loop_idx] # Chỉ số Python (0 đến N-1)
        true_song_id_matlab_style = python_song_idx + 1 # ID thật của bài hát (1 đến 50)

        print(f"--- Testing Song (True ID: {true_song_id_matlab_style}) ---")
        
        time_taken_for_this_song_s = 0
        id_returned_by_shazam = 0 # Giả định mainabc123 trả về 0 nếu không tìm thấy ban đầu

        while id_returned_by_shazam == 0: # Vòng lặp cho đến khi mainabc123 trả về ID khác 0
            time_taken_for_this_song_s += 1 # Tăng độ dài clip lên 1 giây
            
            # Tải lại file audio (giống logic MATLAB mẫu)
            # Điều này không hiệu quả, nhưng để mô phỏng đúng
            path_to_current_song_for_load = os.path.join(songs_folder_path, f"{true_song_id_matlab_style}.mat")
            y_full_current_song, fs_current_song = load_audio_mat(path_to_current_song_for_load)
            if y_full_current_song is None:
                print(f"    Could not reload song {true_song_id_matlab_style} in while loop. Breaking.")
                id_returned_by_shazam = -999 # Đánh dấu lỗi
                break 

            if time_taken_for_this_song_s > max_clip_len_s_allowed:
                time_taken_for_this_song_s -= 1 # Ghi lại độ dài cuối cùng đã thử
                print(f"  Max clip length {max_clip_len_s_allowed}s reached for song {true_song_id_matlab_style}. Final ID: {id_returned_by_shazam}")
                break 
            
            # Cắt yInput
            # initialTime(i)*Fs : initialTime(i)*Fs + timeTaken(i)*Fs
            start_s_for_clip = clip_initial_start_times_s[python_song_idx]
            start_sample_idx = int(round(start_s_for_clip * fs_current_song)) # Python index từ 0
            
            # Trong MATLAB, y(a:b) lấy từ index a đến b.
            # Trong Python, y[a:b] lấy từ a đến b-1.
            # Độ dài clip MATLAB: N = b - a + 1
            # Độ dài clip Python: N = b - a
            # (initialTime(i)*Fs + timeTaken(i)*Fs) là index cuối cùng (MATLAB)
            # (initialTime(i)*Fs + timeTaken(i)*Fs -1) là index cuối cùng (Python) nếu start là 0
            
            # Để đơn giản, chúng ta sẽ tính số mẫu cần lấy
            num_samples_for_clip = int(round(time_taken_for_this_song_s * fs_current_song))
            end_sample_idx = start_sample_idx + num_samples_for_clip # Python slicing sẽ lấy đến end_sample_idx - 1

            # Đảm bảo không vượt quá giới hạn và clip hợp lệ
            start_sample_idx = max(0, start_sample_idx) # Đảm bảo không âm
            end_sample_idx = min(end_sample_idx, len(y_full_current_song))

            if start_sample_idx >= end_sample_idx or (end_sample_idx - start_sample_idx) < fs_current_song / 4: # Ít nhất 0.25s
                print(f"  Clip for song {true_song_id_matlab_style} (len {time_taken_for_this_song_s}s) became invalid. Using previous result.")
                if time_taken_for_this_song_s > 1: time_taken_for_this_song_s -=1
                else: time_taken_for_this_song_s = 0
                break 
            
            yInput_segment = y_full_current_song[start_sample_idx : end_sample_idx]
            
            print(f"  Attempting ID for song {true_song_id_matlab_style} with clip length: {time_taken_for_this_song_s}s, starting at {start_s_for_clip:.2f}s")
            
            try:
                # Gọi hàm nhận dạng Python của bạn
                id_returned_by_shazam = main_shazam_recognizer(yInput_segment, fs_current_song, db_hashtable, db_params_config)
                print(f"      main_shazam_recognizer returned: {id_returned_by_shazam}")
                # Script MATLAB không xử lý -1, nó sẽ dừng vòng while nếu id_returned_by_shazam > 0
                if id_returned_by_shazam == -1: # Nếu hàm Python trả về -1
                    id_returned_by_shazam = 0 # Coi như chưa tìm thấy để tiếp tục lặp
                    if time_taken_for_this_song_s == max_clip_len_s_allowed:
                        break # Dừng nếu đã ở max length
            except Exception as e_main:
                print(f"    ERROR calling main_shazam_recognizer for song {true_song_id_matlab_style}: {e}. Treating as Not Found.")
                id_returned_by_shazam = 0
                break
        
        identified_song_ids_final[python_song_idx] = id_returned_by_shazam
        time_taken_per_song_s[python_song_idx] = time_taken_for_this_song_s

        if id_returned_by_shazam != true_song_id_matlab_style and id_returned_by_shazam != 0:
             misidentified_log_output.append({
                 "file": f"{true_song_id_matlab_style}.mat", "true_id": true_song_id_matlab_style, 
                 "identified_id": id_returned_by_shazam, 
                 "clip_start_s": start_s_for_clip, 
                 "clip_len_s": time_taken_for_this_song_s
             })

    # --- Results (Mô phỏng cách tính của MATLAB script) ---
    # Chỉ xét các bài hát thực sự đã được thử (valid_song_indices)
    true_ids_for_stats = np.array([idx + 1 for idx in valid_song_indices])
    identified_ids_for_stats = identified_song_ids_final[valid_song_indices]
    time_taken_for_stats = time_taken_per_song_s[valid_song_indices]

    num_actually_tested = len(valid_song_indices)

    if num_actually_tested == 0:
        print("No songs were valid for testing.")
        return

    correct_identifications = np.sum(identified_ids_for_stats == true_ids_for_stats)
    
    accuracy_out = correct_identifications / num_actually_tested
    avg_time_taken_out = np.sum(time_taken_for_stats) / num_actually_tested
    
    # Tính điểm theo công thức trong script MATLAB mẫu của bạn:
    # points = 3*sum(songID == [1:n]) - nnz(songID);
    # nnz(songID) trong MATLAB là số phần tử khác 0.
    # sum(songID == [1:n]) là số lần đúng.
    points_out = 3 * correct_identifications - np.count_nonzero(identified_ids_for_stats) 
    # Hoặc nếu nnz(songID) là tổng số lần trả về ID > 0 (cả đúng và sai)
    # num_positive_ids_returned = np.count_nonzero(identified_ids_for_stats > 0)
    # points_out = 3 * correct_identifications - num_positive_ids_returned

    score_out = points_out / avg_time_taken_out if avg_time_taken_out > 0 else 0

    print(f"\n--- FINAL RESULTS (Python Test Script) ---")
    print(f"Accuracy: {accuracy_out:.4f} ({accuracy_out*100:.2f}%)")
    print(f"Average Time Taken (clip length in s): {avg_time_taken_out:.4f}")
    print(f"Points (3*Correct - NonZero IDs): {points_out}") # Điều chỉnh nếu cần
    print(f"Score (Points/AvgTimeTaken): {score_out:.4f}")

    if misidentified_log_output:
        print("\n--- MISIDENTIFICATION DETAILS ---")
        print(f"{'Original File':<15} | {'True ID':>7} | {'Identified ID':>13} | {'Clip Start (s)':>14} | {'Clip Length (s)':>15}")
        print("-" * 80)
        for item in misidentified_log_output:
            print(f"{item['file']:<15} | {item['true_id']:7d} | {str(item['identified_id']):>13} | {item['clip_start_s']:14.2f} | {item['clip_len_s']:15d}")
    print(f"=============================================================\n")


if __name__ == "__main__":
    config = Params() # Sử dụng config mặc định
    
    # Đây là các đường dẫn và số lượng cho bộ test đầy đủ 50 bài
    # Nếu bạn muốn test bộ rút gọn, hãy thay đổi các giá trị này
    # và đảm bảo main_shazam_recognizer tải đúng database đã được tạo cho bộ rút gọn đó.
    songs_to_test_path = config.DB_SONG_PATH # 'songDataBase/'
    num_songs_overall = 50
    db_to_use = config.DB_MAT_FILE # 'my_shazam_database_python.pkl'
    max_clip_duration = 10

    if not os.path.exists(db_to_use):
        print(f"Database file {db_to_use} not found. Please run main_create_db.py first.")
    else:
        run_official_lab_test_simulation(songs_to_test_path, num_songs_overall, db_to_use, max_clip_duration)