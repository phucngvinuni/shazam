# File: auto_tune_fingerprint_params.py
import numpy as np
import os
import itertools 
import pickle   
from copy import deepcopy 
import time
from scipy.signal import spectrogram as scipy_spectrogram # Đảm bảo đã cài scipy
from config import Params 
from database import create_shazam_db 
from analyze_pair_function import run_pair_analysis 
def ifelse(condition, true_val, false_val):
    if condition: return true_val
    else: return false_val
overall_tic_start_time = time.time() # Đảm bảo time được import
print('--- Starting Shazam Parameter Optimization Script ---'); # Dùng print trong Python

# --- Configuration ---
DB_PATH_TUNING = 'songDatabaseReduced_problematic/' # Đảm bảo thư mục này có file 1.mat, 2.mat ...
NUM_SONGS_TUNING_DB = 8 # Ví dụ: sẽ dùng 1.mat, 2.mat, 3.mat, 4.mat
TEST_CLIPS_PATH_FOR_OPTIMIZATION = 'songDataBaseShortReduced/' # Không dùng trong script này
MAX_CLIP_DURATION_SECONDS_FOR_TEST_CONFIG = 10 # Không dùng trong script này

output_tuning_log_filename = 'shazam_fp_tuning_log.pkl'
print(f"Fingerprint parameter tuning. Log will be saved to: {output_tuning_log_filename}")

# --- Parameter Space ---
param_grid = {
    'PEAKS_PER_SEC': [15, 20],            
    'DELTA_F_BINS': [10, 15],           
    'DELTA_TU_FRAMES': [50, 60],        
    'FAN_OUT': [2, 3],                  
    'BITS_F_QUANT': [8, 9],             
    'BITS_DT_QUANT': [7, 8]             
}
# Các tham số cố định khác sẽ lấy từ Params() mặc định

keys, values = zip(*param_grid.items())
parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
total_iterations = len(parameter_combinations)
print(f"Total parameter combinations for fingerprinting to test: {total_iterations}")
if total_iterations == 0:
    print('No parameter combinations to test. Exiting.')
    exit()

tuning_results_log = [] 
# log_idx không cần thiết nếu dùng list.append

# Các cặp bài hát muốn phân tích UCRF (ID phải nằm trong 1 đến NUM_SONGS_TUNING_DB)
PAIRS_TO_ANALYZE = []
if NUM_SONGS_TUNING_DB >= 2:
    PAIRS_TO_ANALYZE.append((1, 2)) 
if NUM_SONGS_TUNING_DB >= 4:
    PAIRS_TO_ANALYZE.append((3, 4))
if NUM_SONGS_TUNING_DB >= 6:
    PAIRS_TO_ANALYZE.append((5, 6))
if NUM_SONGS_TUNING_DB >= 8:
    PAIRS_TO_ANALYZE.append((7, 8))
# Thêm các cặp khác nếu NUM_SONGS_TUNING_DB lớn hơn

print(f"Will analyze UCRF for pairs: {PAIRS_TO_ANALYZE} (using songs 1 to {NUM_SONGS_TUNING_DB} from {DB_PATH_TUNING})")


for iter_idx, param_combination in enumerate(parameter_combinations):
    print(f"\n\n================================================================")
    print(f"TUNING ITERATION {iter_idx + 1} of {total_iterations}")

    current_config_obj = deepcopy(Params()) 
    for key, value in param_combination.items():
        if hasattr(current_config_obj, key): # Kiểm tra xem thuộc tính có tồn tại không
            setattr(current_config_obj, key, value)
        else:
            print(f"Warning: Parameter '{key}' not found in Params class. Check config.py and param_grid.")
            # Để an toàn, có thể đặt tên thuộc tính trong config.py giống hệt key trong param_grid
            # Hoặc sửa lại tên key trong param_grid cho khớp
            # Ví dụ: Nếu trong Params là DELTA_F_BINS_CONFIG, thì key trong param_grid cũng nên là DELTA_F_BINS_CONFIG

    # Gán các tên biến mà các hàm con có thể đang mong đợi (nếu có)
    # Tốt hơn là sửa các hàm con để chúng nhận trực tiếp các trường từ current_config_obj
    current_config_obj.DELTA_F_BINS_CONFIG = current_config_obj.DELTA_F_BINS 
    current_config_obj.DELTA_TL_FRAMES_CONFIG = current_config_obj.DELTA_TL_FRAMES
    current_config_obj.DELTA_TU_FRAMES_CONFIG = current_config_obj.DELTA_TU_FRAMES
    current_config_obj.FAN_OUT_CONFIG = current_config_obj.FAN_OUT

    print("Current Fingerprinting Configuration (Varied Part):")
    for key, value in param_combination.items(): print(f"  {key}: {value}")
    print(f"  (Other params like TARGET_FS={current_config_obj.TARGET_FS}, WINDOW_MS={current_config_obj.WINDOW_MS}, etc., from Params default or fixed_params)")

    db_created_successfully = False # <<<<<< KHỞI TẠO Ở ĐÂY
    db_creation_error_message = ''
    created_db_for_analysis = None
    temp_db_file = f'temp_tuning_db_iter{iter_idx+1}.pkl' # Tên file DB tạm thời duy nhất

    print(f"Creating temporary database ({temp_db_file}) for this configuration...")
    try:
        created_db_for_analysis = create_shazam_db(
            DB_PATH_TUNING, 
            NUM_SONGS_TUNING_DB, 
            current_config_obj, 
            temp_db_file
        )
        if created_db_for_analysis is not None:
            db_created_successfully = True
            print(f"  Temporary DB created successfully.")
        else:
            db_creation_status = "DB Creation Failed (create_shazam_db returned None)"
            print(f"  {db_creation_status}")
            db_creation_error_message = db_creation_status # Lưu lại
    except Exception as e_db:
        db_creation_status = f"DB Creation Error: {e_db}"
        print(f"  {db_creation_status}")
        db_creation_error_message = db_creation_status # Lưu lại
    
    ucrf_results_for_this_config = {}
    if db_created_successfully: 
        print("  Analyzing UCRF for specified pairs...")
        for id1, id2 in PAIRS_TO_ANALYZE:
            if id1 <= NUM_SONGS_TUNING_DB and id2 <= NUM_SONGS_TUNING_DB and id1 != id2:
                ucrf, ch, nfp1, nfp2 = run_pair_analysis(id1, id2, current_config_obj, DB_PATH_TUNING)
                ucrf_results_for_this_config[f"{id1}v{id2}"] = {'ucrf': ucrf, 'common_hashes': ch, 'fp1_count':nfp1, 'fp2_count':nfp2}
            else:
                print(f"  Skipping pair ({id1},{id2}) as IDs might be out of range for the tuning DB or identical.")
        if os.path.exists(temp_db_file): # Dọn dẹp file DB tạm
            try: os.remove(temp_db_file); print(f"  Removed temporary DB: {temp_db_file}");
            except Exception as e_del: print(f"  Warning: Could not remove temp DB {temp_db_file}: {e_del}");
    else: 
        for id1, id2 in PAIRS_TO_ANALYZE:
             ucrf_results_for_this_config[f"{id1}v{id2}"] = {'ucrf': 'N/A due to DB error'}

    tuning_results_log.append({
        'config_tested': param_combination,
        'full_config_obj': current_config_obj, # Lưu cả object config đầy đủ để tham khảo
        'ucrf_analysis': ucrf_results_for_this_config,
        'db_creation_status': ifelse(db_created_successfully, "OK", db_creation_error_message)
    })
    
    try:
        with open(output_tuning_log_filename, 'wb') as f_log:
            pickle.dump(tuning_results_log, f_log)
        print(f"Tuning log updated: {output_tuning_log_filename}")
    except Exception as e_log_save:
        print(f"Error saving tuning log: {e_log_save}")

print("\n--- Fingerprint Parameter Tuning Script Finished ---")
total_script_time_minutes = (time.time() - overall_tic_start_time) / 60 
print(f"Total tuning time: {total_script_time_minutes:.2f} minutes")
print(f"Results saved in {output_tuning_log_filename}")

# --- Phân tích tuning_results_log ---
if os.path.exists(output_tuning_log_filename):
    with open(output_tuning_log_filename, 'rb') as f_log:
        final_log = pickle.load(f_log)
    if final_log: # Kiểm tra xem log có rỗng không
        print("\n--- Tuning Log Summary (Focus on UCRF) ---")
        # Tìm cấu hình cho tổng UCRF thấp nhất trên các cặp đã phân tích
        best_config_info = None
        min_total_ucrf = float('inf')

        for i, record in enumerate(final_log):
            print(f"\nIteration {i+1}: DB Status: {record['db_creation_status']}")
            print("  Config Tested:")
            current_total_ucrf_for_config = 0
            valid_ucrf_found = False
            for param, val in record['config_tested'].items(): print(f"    {param}: {val}")
            print("  UCRF Analysis:")
            if isinstance(record['ucrf_analysis'], dict):
                for pair_key, pair_data in record['ucrf_analysis'].items():
                    ucrf_val = pair_data.get('ucrf', 'N/A')
                    print(f"    Pair {pair_key}: UCRF={ucrf_val}, "
                          f"FP1={pair_data.get('fp1_count','N/A')}, FP2={pair_data.get('fp2_count','N/A')}")
                    if isinstance(ucrf_val, (int, float)):
                        current_total_ucrf_for_config += ucrf_val
                        valid_ucrf_found = True
            else: print(f"    {record['ucrf_analysis']}")
            
            if valid_ucrf_found and record['db_creation_status'] == "OK":
                print(f"  Total UCRF for this config (for analyzed pairs): {current_total_ucrf_for_config}")
                if current_total_ucrf_for_config < min_total_ucrf:
                    min_total_ucrf = current_total_ucrf_for_config
                    best_config_info = record
        
        if best_config_info:
            print("\n--- Best Configuration Found (Lowest Total UCRF for analyzed pairs) ---")
            print(f"Achieved Min Total UCRF: {min_total_ucrf}")
            print("Config Parameters:")
            for param, val in best_config_info['config_tested'].items(): print(f"  {param}: {val}")
            print("Detailed UCRF for best config:")
            if isinstance(best_config_info['ucrf_analysis'], dict):
                for pair_key, pair_data in best_config_info['ucrf_analysis'].items():
                    print(f"  Pair {pair_key}: UCRF={pair_data.get('ucrf', 'N/A')}")
        else:
            print("Could not determine a best configuration from the log (no valid UCRF data or all DB creations failed).")
    else: print(f"Log file {output_tuning_log_filename} is empty or invalid.")
else: print(f"Log file {output_tuning_log_filename} not found.");

def ifelse(condition, true_val, false_val): # Di chuyển hàm này ra ngoài hoặc import
    if condition: return true_val
    else: return false_val