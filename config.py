# config.py
class Params:
    TARGET_FS = 8000  # Hz
    APPLY_LPF = True
    LPF_PASSBAND_HZ = 3000  # Hz

    WINDOW_MS = 64  # ms
    OVERLAP_RATIO = 0.5
    FREQ_HZ_QUANT_STEP = 20 # Lượng tử hóa tần số Hz thành các bước 20Hz
    DT_FRAMES_QUANT_STEP = 3 # Lượng tử hóa dt_frames thành các bước 3 frame
    GS_PEAK_FIND = 9  # Kích thước grid tìm peak
    PEAKS_PER_SEC = 20  # Số lượng peak tìm mỗi giây
    DELTA_F_BINS = 15 # Khoảng +- bins tần số cho target zone
    DELTA_TL_FRAMES = 2  # Min frame offset
    DELTA_TU_FRAMES = 50 # Max frame offset
    FAN_OUT = 3

    # Tham số lượng tử hóa cho hash
    BITS_F_QUANT = 8
    BITS_DT_QUANT = 7
    FREQ_HZ_QUANT_STEP = 20 # Lượng tử hóa tần số Hz thành các bước 20Hz
    DT_FRAMES_QUANT_STEP = 3 # Lượng tử hóa dt_frames thành các bước 3 frame
    # Ngưỡng khớp
    DEFAULT_MIN_MATCH_THRESHOLD = 10
    THRESHOLDS_FOR_CLIP_LENGTH = {
        "short_max_len": 1.5,  # s
        "short_thresh": 6,
        "medium_max_len": 3.5, # s
        "medium_thresh": 10,
        "long_thresh": 12,
    }

    # Đường dẫn (điều chỉnh nếu cần)
    DB_SONG_PATH = 'songDataBase/'
    DB_MAT_FILE = 'my_shazam_database_python.pkl' # Hoặc .npz

    # Cho việc test (nếu dùng bộ rút gọn)
    DB_REDUCED_PATH = 'songDatabaseReduced/'
    NUM_SONGS_REDUCED = 10
    TEST_CLIPS_PATH = 'songDataBaseShortReduced/' # Hoặc songDataBaseShort/