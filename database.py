# database.py
import numpy as np
import os
import pickle # Để lưu/tải object Python (như list of tuples hoặc dict)
# Hoặc dùng numpy.savez_compressed nếu chỉ lưu mảng NumPy
from preprocess import load_audio_mat, preprocess_audio_signal
from fingerprint import create_fingerprints
from config import Params

def create_shazam_db(db_song_path, num_songs, config_params, output_db_file):
    """Tạo database fingerprint và lưu vào file."""
    print(f"--- Creating Shazam Database (Python) ---")
    print(f"Using song path: {db_song_path}, Number of songs: {num_songs}")
    print(f"Config: TargetFS={config_params.TARGET_FS}, LPF={config_params.APPLY_LPF}@{config_params.LPF_PASSBAND_HZ}Hz, "
          f"Window={config_params.WINDOW_MS}ms, Overlap={config_params.OVERLAP_RATIO*100}%, "
          f"Peaks/s={config_params.PEAKS_PER_SEC}, FanOut={config_params.FAN_OUT}")

    # HashTable sẽ là một dictionary: hash_value -> list of (t1_anchor_idx, song_id)
    # Hoặc một list of tuples: (hash_value, t1_anchor_idx, song_id) rồi sort
    # Cách 2 dễ hơn để bắt đầu và tương tự MATLAB
    
    all_hashed_fingerprints_list = []
    
    for song_id in range(1, num_songs + 1):
        filepath = os.path.join(db_song_path, f"{song_id}.mat")
        print(f"\nProcessing DB song {song_id}/{num_songs}: {filepath}")
        
        y_orig, fs_orig = load_audio_mat(filepath)
        if y_orig is None:
            print(f"  Skipping song {song_id}: Could not load.")
            continue
            
        y_proc, fs_proc = preprocess_audio_signal(y_orig, fs_orig, 
                                                config_params.APPLY_LPF, 
                                                config_params.TARGET_FS, 
                                                config_params.LPF_PASSBAND_HZ)
        if y_proc.size == 0:
            print(f"  Skipping song {song_id}: Empty after preprocessing.")
            continue

        hashed_fp_one_song = create_fingerprints(y_proc, fs_proc, config_params) # Trả về [hash_val, t1_idx]
        
        if hashed_fp_one_song.size > 0:
            num_fp_this_song = hashed_fp_one_song.shape[0]
            # Thêm song_id vào: [hash_val, t1_idx, song_id]
            song_id_col = np.full((num_fp_this_song, 1), song_id, dtype=np.int32)
            db_entries_one_song = np.hstack((hashed_fp_one_song, song_id_col))
            all_hashed_fingerprints_list.append(db_entries_one_song)
            print(f"  Song {song_id}: Generated {num_fp_this_song} hashed fingerprints.")
        else:
            print(f"  Song {song_id}: No hashed fingerprints generated.")

    if not all_hashed_fingerprints_list:
        print("ERROR: No fingerprints generated for the entire database.")
        return None

    # Ghép tất cả lại thành một mảng lớn
    hash_table_final = np.vstack(all_hashed_fingerprints_list)
    
    print(f"\nSorting final DB hashTable ({hash_table_final.shape[0]} entries)...")
    # Sắp xếp theo hash_value (cột 0)
    sort_indices = np.argsort(hash_table_final[:, 0])
    hash_table_final_sorted = hash_table_final[sort_indices]
    print("  Sorting Done.")

    # Lưu database (bao gồm cả config)
    db_to_save = {
        'hashTable': hash_table_final_sorted,
        'PARAMS_CONFIG': config_params # Lưu lại config đã dùng
    }
    try:
        with open(output_db_file, 'wb') as f:
            pickle.dump(db_to_save, f)
        print(f"New DB ('{output_db_file}') created: {hash_table_final_sorted.shape[0]} entries. PARAMS_CONFIG also saved.")
    except Exception as e:
        print(f"Error saving database to {output_db_file}: {e}")
        return None
        
    print("--- Finished creating database (Python) ---")
    return hash_table_final_sorted # Trả về để có thể dùng ngay nếu muốn

def load_shazam_db(db_filepath):
    """Tải database từ file pickle."""
    if not os.path.exists(db_filepath):
        print(f"Database file {db_filepath} not found!")
        return None, None
    try:
        with open(db_filepath, 'rb') as f:
            loaded_db = pickle.load(f)
        if 'hashTable' in loaded_db and 'PARAMS_CONFIG' in loaded_db:
            print(f"Database '{db_filepath}' loaded successfully.")
            return loaded_db['hashTable'], loaded_db['PARAMS_CONFIG']
        else:
            print(f"Error: 'hashTable' or 'PARAMS_CONFIG' not found in {db_filepath}")
            return None, None
    except Exception as e:
        print(f"Error loading database from {db_filepath}: {e}")
        return None, None