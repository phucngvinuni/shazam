# main_create_db.py
import os
from config import Params
from database import create_shazam_db

if __name__ == "__main__":
    # Sử dụng các tham số từ config.py
    # Bạn có thể tạo nhiều đối tượng Params khác nhau nếu muốn thử nghiệm config khác nhau
    current_config = Params() 
    
    # Đảm bảo thư mục database gốc tồn tại
    if not os.path.isdir(current_config.DB_SONG_PATH):
        print(f"ERROR: Database song path '{current_config.DB_SONG_PATH}' not found.")
        exit()
        
    # Tạo tên file DB output
    output_db_filepath = current_config.DB_MAT_FILE # Ví dụ: 'my_shazam_database_python.pkl'
    
    # Tạo database
    # Giả sử bạn có 50 bài hát trong DB_SONG_PATH
    create_shazam_db(current_config.DB_SONG_PATH, 50, current_config, output_db_filepath)