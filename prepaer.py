# File: prepare_tuning_dataset.py
import os
import shutil # Thư viện để copy file

# --- Configuration ---
ORIGINAL_DB_PATH = 'songDataBase/' # Thư mục chứa 50 bài hát gốc
TUNING_DB_PATH = 'songDatabaseReduced_problematic/' # Thư mục sẽ được tạo và chứa các file chọn lọc

# Danh sách ID các bài hát bạn muốn đưa vào bộ tuning
# Đây là các ID gốc (ví dụ, nếu file là 13.mat, thì ID là 13)
# Đảm bảo các ID này tương ứng với các file có thật trong ORIGINAL_DB_PATH
SONG_IDS_FOR_TUNING = [13, 26, 28, 44, 38, 46, 43, 41] # Ví dụ các cặp hay sai

# QUAN TRỌNG: Cách đặt tên file trong thư mục tuning
# Hàm create_shazam_db hiện tại mong đợi các file được đặt tên là 1.mat, 2.mat, ...
# Do đó, chúng ta sẽ copy và đổi tên chúng nếu cần.
# Hoặc, bạn có thể sửa create_shazam_db để nó đọc file dựa trên danh sách ID.
# Cách đơn giản hơn là copy và đổi tên.

RENAME_FILES_SEQUENTIALLY = True # Đặt là True để đổi tên thành 1.mat, 2.mat,...
                                # Đặt là False để giữ nguyên tên gốc (ví dụ 13.mat, 26.mat)
                                # Nếu False, bạn cần sửa create_shazam_db để đọc đúng tên.

print(f"--- Preparing Tuning Dataset ---")
print(f"Original DB: {ORIGINAL_DB_PATH}")
print(f"Tuning DB to create/populate: {TUNING_DB_PATH}")
print(f"Song IDs to include: {SONG_IDS_FOR_TUNING}")

if not os.path.isdir(ORIGINAL_DB_PATH):
    print(f"ERROR: Original database path '{ORIGINAL_DB_PATH}' not found.")
    exit()

# Tạo thư mục tuning nếu chưa có
if not os.path.isdir(TUNING_DB_PATH):
    try:
        os.makedirs(TUNING_DB_PATH)
        print(f"Created directory: {TUNING_DB_PATH}")
    except Exception as e:
        print(f"ERROR: Could not create directory {TUNING_DB_PATH}: {e}")
        exit()
else:
    print(f"Directory {TUNING_DB_PATH} already exists. Files might be overwritten.")

copied_count = 0
for idx, original_song_id in enumerate(SONG_IDS_FOR_TUNING):
    source_filename = f"{original_song_id}.mat"
    source_filepath = os.path.join(ORIGINAL_DB_PATH, source_filename)

    if RENAME_FILES_SEQUENTIALLY:
        destination_filename = f"{idx + 1}.mat" # Đổi tên thành 1.mat, 2.mat, ...
    else:
        destination_filename = source_filename # Giữ nguyên tên gốc

    destination_filepath = os.path.join(TUNING_DB_PATH, destination_filename)

    if os.path.exists(source_filepath):
        try:
            shutil.copy2(source_filepath, destination_filepath) # copy2 giữ lại metadata
            print(f"  Copied: {source_filepath} -> {destination_filepath}")
            copied_count += 1
        except Exception as e:
            print(f"  ERROR copying {source_filename}: {e}")
    else:
        print(f"  Source file not found: {source_filepath}")

print(f"\nFinished preparing tuning dataset. Total files copied: {copied_count} into {TUNING_DB_PATH}")
if copied_count != len(SONG_IDS_FOR_TUNING):
    print("WARNING: Not all requested songs were copied. Check source files.")