import numpy as np
import os

# --- 1. HÀM ĐỌC VÀ XỬ LÝ DỮ LIỆU ---
def load_raw_data(filepath):
    """
    Hàm chỉ đọc dữ liệu thô để kiểm tra
    """
    print(f"Đang đọc dữ liệu thô từ {filepath}...")

    if not os.path.exists(filepath):
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return None, None

    # Đọc header
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        header = f.readline().strip().split(',')
        header = [h.strip('"').strip() for h in header]

    # Đọc dữ liệu thô
    try:
        raw_data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=None, encoding='utf-8-sig')
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None, None
        
    return np.array(header), raw_data

def load_and_process_data(filepath):
    """
    Đọc dữ liệu và mã hóa (Encoding) sang dạng số.
    """
    print(f"Đang đọc dữ liệu từ {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    # Đọc header
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        header = [h.strip().strip('"').strip("'") for h in f.readline().strip().split(',')]
    
    # Load dữ liệu thô
    try:
        raw_data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=None, encoding='utf-8-sig')
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None, None, None

    # Xác định Target
    target_col = 'Attrition_Flag'
    if target_col not in header:
        candidates = [c for c in header if 'Attrition' in c]
        if candidates:
            target_col = candidates[0]
        else:
            raise KeyError(f"Thiếu cột target 'Attrition_Flag'")
    
    target_idx = header.index(target_col)
    y_raw = np.array([row[target_idx] for row in raw_data])
    # Mã hóa Target: 'Attrited Customer' -> 1, 'Existing Customer' -> 0
    y = np.array([1 if str(val).strip('"').strip() == 'Attrited Customer' else 0 for val in y_raw])

    # Xác định Features (Bỏ cột rác)
    ignore_cols = ['CLIENTNUM', target_col, 'Naive_Bayes_Classifier'] 
    features = []
    feature_names = []

    for i, name in enumerate(header):
        if any(ign in name for ign in ignore_cols): continue
        
        feature_names.append(name)
        col_data = np.array([row[i] for row in raw_data])
        
        # Xử lý kiểu dữ liệu
        try:
            # Nếu là số -> Giữ nguyên
            features.append(col_data.astype(float))
        except ValueError:
            # Nếu là chữ -> Label Encoding thủ công
            _, encoded = np.unique(col_data, return_inverse=True)
            features.append(encoded.astype(float))
            
    X = np.column_stack(features)
    feature_names = np.array(feature_names)
    print(f"-> Load thành công. Shape: {X.shape}")
    return X, y, feature_names

# --- 2. HÀM TIỀN XỬ LÝ DỮ LIỆU ---
def load_data_for_preprocessing(filepath):
    """
    Đọc dữ liệu dạng chuỗi để tạo ma trận 2D.
    Dùng chuyên biệt cho việc cắt lớp và xử lý ma trận.
    """
    print(f"--- Đang tải dữ liệu 2D từ {filepath} ---")
    if not os.path.exists(filepath):
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return None, None

    # Đọc header
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        header = f.readline().strip().replace('"', '').split(',')
        header = np.array([h.strip() for h in header])

    # Đọc dữ liệu
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str, encoding='utf-8-sig')
        # Xóa dấu ngoặc kép bao quanh dữ liệu
        data = np.char.strip(data, '"')
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None, None
        
    print(f"Kích thước dữ liệu (Rows, Cols): {data.shape}")
    return header, data

def clean_data_numpy(header, data):
    """
    Loại bỏ các cột không cần thiết: CLIENTNUM, Naive_Bayes..., Avg_Open_To_Buy
    """
    # Từ khóa các cột cần bỏ
    cols_to_drop = ['CLIENTNUM', 'Naive_Bayes', 'Avg_Open_To_Buy']
    
    # Tạo mask giữ lại cột hợp lệ
    keep_mask = np.array([not any(x in col for x in cols_to_drop) for col in header])
    
    header_clean = header[keep_mask]
    data_clean = data[:, keep_mask]
    
    print(f"Đã loại bỏ cột nhiễu. Kích thước mới: {data_clean.shape}")
    return header_clean, data_clean

def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    """
    Chia tập dữ liệu Train/Test thủ công bằng NumPy.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Tạo danh sách chỉ số và xáo trộn
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Tính điểm cắt
    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    print(f"-> Chia dữ liệu: Train ({len(train_idx)}), Test ({len(test_idx)})")
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class NumpyPreprocessor:
    """
    Class xử lý dữ liệu sử dụng NumPy.
    """
    def __init__(self):
        self.stats = {} # Lưu tham số học được từ Train
        self.header = None
        
    def fit(self, header, X_train):
        """Học tham số (Mean, Median, Categories) từ tập Train."""
        print("--- Đang 'Fit' trên tập Train ---")
        self.header = header
        self.col_types = {} 
        
        # Định nghĩa thứ tự cho biến Ordinal
        self.ord_orders = {
            'Education_Level': ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
            'Income_Category': ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
            'Card_Category': ['Blue', 'Silver', 'Gold', 'Platinum']
        }

        
        for i, col_name in enumerate(header):
            col_data = X_train[:, i]
            
            # Kiểm tra Biến số
            try:
                # Thử ép kiểu sang float, nếu thành công là biến số
                vals = col_data.astype(float)
                self.col_types[i] = 'num'
                # Lưu Median và IQR (Robust Scaler)
                self.stats[col_name] = {
                    'median': np.median(vals),
                    'q75': np.percentile(vals, 75),
                    'q25': np.percentile(vals, 25)
                }
            except:
                # Biến Ordinal
                if col_name in self.ord_orders:
                    self.col_types[i] = 'ord'
                # Biến Categorical (Nominal)
                else:
                    self.col_types[i] = 'cat'
                    # Lưu các giá trị unique để One-Hot
                    self.stats[col_name] = {'categories': np.unique(col_data)}

    def transform(self, X):
        """Áp dụng tham số đã học để biến đổi dữ liệu."""
        processed_cols = []
        new_header = []
        
        for i, col_name in enumerate(self.header):
            col_data = X[:, i]
            col_type = self.col_types.get(i, 'ignore')
            
            # Xử lý biến số
            if col_type == 'num':
                vals = col_data.astype(float)
                
                # Log Transform cho các cột tiền tệ lớn (như trong EDA)
                if any(x in col_name for x in ['Limit', 'Amt', 'Bal']):
                    vals = np.log1p(vals)
                    prefix = "Log_"
                else:
                    prefix = ""

                # Robust Scaling: (x - median) / IQR
                stats = self.stats[col_name]
                iqr = stats['q75'] - stats['q25']
                if iqr == 0: iqr = 1 # Tránh chia cho 0
                
                scaled = (vals - stats['median']) / iqr
                processed_cols.append(scaled.reshape(-1, 1))
                new_header.append(f"{prefix}{col_name}")
                
            # Xử lý biến Ordinal
            elif col_type == 'ord':
                order = self.ord_orders[col_name]
                # Map giá trị sang index (0, 1, 2...)
                mapped = np.array([order.index(v) if v in order else 0 for v in col_data])
                processed_cols.append(mapped.reshape(-1, 1))
                new_header.append(col_name)
                
            # Xử lý biến Categorical
            elif col_type == 'cat':
                cats = self.stats[col_name]['categories']
                for cat in cats:
                    # Tạo cột nhị phân: 1 nếu khớp, 0 nếu không
                    ohe_col = (col_data == cat).astype(int)
                    processed_cols.append(ohe_col.reshape(-1, 1))
                    new_header.append(f"{col_name}_{cat}")
        
        # Gộp các cột lại thành ma trận
        return np.hstack(processed_cols), np.array(new_header)

def save_numpy(path, X, y, header):
    """Lưu dữ liệu dạng .npy (nhị phân) để đọc nhanh và giữ nguyên kiểu số."""
    np.save(path.replace('.npy', '_X.npy'), X)
    np.save(path.replace('.npy', '_y.npy'), y)
    np.save(path.replace('.npy', '_header.npy'), header)
    print(f"Đã lưu file: {path}")

# --- 3. TẠO THÊM ĐẶC TRƯNG MỚI ---
def feature_engineering_numpy(X, feature_names):
    """
    Tạo thêm đặc trưng mới:
    Avg_Trans_Amt = Total_Trans_Amt / Total_Trans_Ct
    """
    print("--- Feature Engineering: Tạo cột Avg_Trans_Amt ---")
    
    try:
        # Tìm index cột
        idx_amt = np.where(feature_names == 'Total_Trans_Amt')[0][0]
        idx_ct = np.where(feature_names == 'Total_Trans_Ct')[0][0]
        
        # Lấy dữ liệu
        amt = X[:, idx_amt].astype(float)
        ct = X[:, idx_ct].astype(float)
        
        # Tính toán (tránh chia cho 0)
        ct[ct == 0] = 1 
        avg_amt = amt / ct
        
        # Thêm vào ma trận X
        X_new = np.hstack((X, avg_amt.reshape(-1, 1)))
        
        # Cập nhật tên cột
        names_new = np.append(feature_names, 'Avg_Trans_Amt')
        
        print(f"-> Đã thêm 1 đặc trưng mới. Shape: {X_new.shape}")
        return X_new, names_new
        
    except IndexError:
        print("Lỗi: Không tìm thấy cột Total_Trans_Amt hoặc Total_Trans_Ct để tạo feature mới.")
        return X, feature_names