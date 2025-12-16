import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. PHÂN TÍCH BIẾN SỐ (NUMERICAL) ---
def plot_numerical_analysis(X, feature_names, numerical_cols):
    """
    Vẽ Histogram và Boxplot cho các biến số.
    """
    sns.set_theme(style="whitegrid")
    print(f"=== PHÂN TÍCH CHI TIẾT BIẾN SỐ ===")

    for name in numerical_cols:
        if name not in feature_names:
            continue

        col_idx = np.where(feature_names == name)[0][0]
        data = X[:, col_idx]

        # Thống kê
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        # Tính Skewness
        n = len(data)
        skewness = 0 if std_val == 0 else np.sum((data - mean_val)**3) / (n * std_val**3)
        
        if abs(skewness) < 0.5: shape = "Đối xứng"
        elif skewness > 0.5: shape = "Lệch phải"
        else: shape = "Lệch trái"

        print(f"\nĐẶC TRƯNG: {name}")
        print(f"   - Mean: {mean_val:.2f} | Median: {median_val:.2f} | Std: {std_val:.2f}")
        print(f"   - Phân phối: {shape} (Skew: {skewness:.2f})")

        # Vẽ hình
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        sns.histplot(data, kde=True, ax=axes[0], color='skyblue')
        axes[0].axvline(mean_val, color='red', linestyle='--', label='Mean')
        axes[0].axvline(median_val, color='green', linestyle='-', label='Median')
        axes[0].set_title(f'Distribution: {name}')
        axes[0].legend()

        sns.boxplot(x=data, ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Box Plot: {name}')
        
        plt.tight_layout()
        plt.show()
        print("-" * 80)

# --- 2. PHÂN TÍCH BIẾN PHÂN LOẠI (CATEGORICAL) 
def analyze_and_plot_categorical(data, column_names, categorical_cols):
    """
    Phân tích và vẽ biểu đồ biến phân loại.
    """
    print("=== PHÂN TÍCH CHI TIẾT BIẾN PHÂN LOẠI ===")
    
    # Chuyển header về dạng list string
    if isinstance(column_names, np.ndarray):
        column_names = column_names.tolist()
    
    clean_header = []
    for col in column_names:
        s = str(col)
        # Xử lý nếu header bị dính byte string
        if s.startswith("b'") or s.startswith('b"'):
            s = s[2:-1]
        clean_header.append(s.replace("'", "").replace('"', "").strip())

    for col_name in categorical_cols:
        target_col = col_name.strip()
        
        # Kiểm tra cột có tồn tại trong header không
        if target_col not in clean_header:
            print(f"Không tìm thấy cột '{target_col}' trong dữ liệu.")
            continue

        print(f"\n{'='*60}")
        print(f"CỘT: {target_col}")
        print(f"{'='*60}")

        col_idx = clean_header.index(target_col)
        col_data = None

        try:
            col_data = np.array([row[col_idx] for row in data])
        except IndexError:
            print(f"Lỗi: Dữ liệu dòng không đủ cột tại index {col_idx}.")
            continue
        except Exception as e:
            try:
                col_data = data[:, col_idx]
            except Exception:
                print(f"Lỗi trích xuất dữ liệu: {e}")
                continue

        # Phân tích dữ liệu phân loại
        if col_data is not None:
            # Tính toán tần suất
            total_rows = len(col_data)
            unique_vals, counts = np.unique(col_data, return_counts=True)
            
            # Sắp xếp giảm dần theo số lượng
            sorted_indices = np.argsort(-counts)
            unique_vals = unique_vals[sorted_indices]
            counts = counts[sorted_indices]

            # Decode nếu dữ liệu là dạng Bytes
            if unique_vals.size > 0 and isinstance(unique_vals[0], (bytes, np.bytes_)):
                unique_vals = [x.decode('utf-8') for x in unique_vals]

            # In số liệu thống kê
            print(f"1. Số lượng giá trị (Unique): {len(unique_vals)}")
            print("2. Top giá trị phổ biến:")
            for i in range(min(5, len(unique_vals))):
                pct = (counts[i] / total_rows) * 100
                print(f"   - {unique_vals[i]}: {counts[i]} ({pct:.2f}%)")
            
            top_pct = (counts[0] / total_rows) * 100
            balance_msg = "Mất cân bằng (Imbalanced)" if top_pct > 70 else "Tương đối cân bằng"
            print(f"3. Đánh giá: {balance_msg}")

            # Vẽ biểu đồ
            plt.figure(figsize=(10, 5))
            bars = plt.bar(unique_vals, counts, color='skyblue', edgecolor='black', alpha=0.7)
            
            plt.title(f'Phân phối: {target_col}')
            plt.xlabel(target_col)
            plt.ylabel('Số lượng')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Hiển thị con số trên đầu cột
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                         ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.show()

# --- 5. PHÂN TÍCH BẢNG CHÉO ---
def plot_cross_tab(data, column_names, row_col, col_col):
    """
    Tạo bảng tần suất và Heatmap cho 2 biến phân loại.
    """
    print(f"\n=== BẢNG CHÉO: {row_col} x {col_col} ===")
    
    # Chuẩn hóa Header
    if isinstance(column_names, np.ndarray): column_names = column_names.tolist()
    clean_header = [str(c).replace("b'", "").replace("'", "").strip() for c in column_names]
    
    # Lấy dữ liệu 2 cột
    try:
        row_idx = clean_header.index(row_col)
        col_idx = clean_header.index(col_col)
        
        # Trích xuất và làm sạch dữ liệu
        row_data = [str(row[row_idx]).replace("b'", "").replace("'", "").strip() for row in data]
        col_data = [str(row[col_idx]).replace("b'", "").replace("'", "").strip() for row in data]
    except ValueError:
        print(f"Không tìm thấy cột")
        return

    # Tính toán tần suất
    unique_rows = sorted(list(set(row_data)))
    unique_cols = sorted(list(set(col_data)))
    
    matrix = np.zeros((len(unique_rows), len(unique_cols)), dtype=int)
    
    for r, c in zip(row_data, col_data):
        i = unique_rows.index(r)
        j = unique_cols.index(c)
        matrix[i, j] += 1
        
    # In bảng tần suất
    print(f"{row_col[:15]:<20} | " + " | ".join([f"{c:<15}" for c in unique_cols]))
    print("-" * (20 + 18 * len(unique_cols)))
    
    # Dòng nội dung
    for i, r_label in enumerate(unique_rows):
        row_str = f"{r_label:<20} | " + " | ".join([f"{val:<15}" for val in matrix[i]])
        print(row_str)

    # Vẽ Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=unique_cols, yticklabels=unique_rows)
    plt.title(f'Cross-tab: {row_col} vs {col_col}')
    plt.xlabel(col_col)
    plt.ylabel(row_col)
    plt.tight_layout()
    plt.show()

def plot_grouped_stats(data, column_names, num_col, cat_col):
    """
    Thống kê biến Số theo biến Phân loại.
    """
    print(f"\n=== THỐNG KÊ NHÓM: {num_col} theo {cat_col} ===")
    
    # Chuẩn hóa Header
    if isinstance(column_names, np.ndarray): column_names = column_names.tolist()
    clean_header = [str(c).replace("b'", "").replace("'", "").strip() for c in column_names]
    
    try:
        num_idx = clean_header.index(num_col)
        cat_idx = clean_header.index(cat_col)
        
        # Lấy dữ liệu
        num_data = []
        cat_data = []
        
        for row in data:
            try:
                # Ép kiểu số
                val = float(row[num_idx])
                # Làm sạch nhãn phân loại
                cat = str(row[cat_idx]).replace("b'", "").replace("'", "").strip()
                
                num_data.append(val)
                cat_data.append(cat)
            except ValueError:
                continue # Bỏ qua dòng lỗi
                
    except ValueError:
        print(f"Không tìm thấy cột")
        return

    # Gom nhóm dữ liệu
    groups = {}
    for n, c in zip(num_data, cat_data):
        if c not in groups: groups[c] = []
        groups[c].append(n)
        
    # In bảng thống kê
    print(f"{'Nhóm (Category)':<25} | {'Mean':<10} | {'Median':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 90)
    
    sorted_keys = sorted(groups.keys())
    plot_data = []
    
    for key in sorted_keys:
        vals = groups[key]
        print(f"{key:<25} | {np.mean(vals):<10.2f} | {np.median(vals):<10.2f} | {np.std(vals):<10.2f} | {np.min(vals):<10.2f} | {np.max(vals):<10.2f}")
        plot_data.append(vals)

    # Vẽ Boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(plot_data, labels=sorted_keys, patch_artist=True, 
                boxprops=dict(facecolor='#6BAED6', color='black'),
                medianprops=dict(color='red', linewidth=2))
    
    plt.title(f'Phân phối {num_col} theo {cat_col}', fontsize=14)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- 3. PHÂN TÍCH KHÁM PHÁ DỮ LIỆU ---
def plot_churn_pie_chart(y):
    """
    Vẽ biểu đồ tròn thể hiện tỷ lệ Churn.
    """
    print("=== BIỂU ĐỒ TỶ LỆ RỜI BỎ (CHURN RATE) ===")
    
    unique_vals, counts = np.unique(y, return_counts=True)
    labels = []
    colors = []
    explode = []
    
    for val in unique_vals:
        s = str(val)
        
        if s.startswith("b'") or s.startswith('b"'):
            s = s[2:-1]
        
        s = s.replace("'", "").replace('"', "").strip()
        
        if s in ['1', '1.0', 'Attrited Customer', 'Attrited']:
            label_display = 'Attrited Customer (Rời bỏ)'
            color = '#FF6F61' # Màu Đỏ Cam
            exp = 0.1        
        
        elif s in ['0', '0.0', 'Existing Customer', 'Existing']:
            label_display = 'Existing Customer (Hiện tại)'
            color = '#6BAED6' # Màu Xanh Dương
            exp = 0
            
        else:
            label_display = s
            color = 'lightgray'
            exp = 0
            
        labels.append(label_display)
        colors.append(color)
        explode.append(exp)

    # Vẽ biểu đồ tròn
    plt.figure(figsize=(8, 8))
    
    if all(c == 'lightgray' for c in colors):
        print("Cảnh báo: Dữ liệu không khớp quy chuẩn 0/1 hoặc Attrited/Existing.")
        colors = ['#ff9999', '#66b3ff']
        if len(labels) > 2: colors = None 

    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=colors, explode=explode, shadow=True, 
            textprops={'fontsize': 12, 'weight': 'bold'})
    
    plt.title('Tỷ lệ Khách hàng Rời bỏ vs Hiện tại', fontsize=15, fontweight='bold')
    plt.axis('equal') 
    plt.show()

def plot_correlation_bar(feature_names, correlations, sorted_indices):
    """
    Vẽ biểu đồ cột ngang thể hiện hệ số tương quan.
    """
    print("=== BIỂU ĐỒ TƯƠNG QUAN VỚI BIẾN MỤC TIÊU ===")

    names_sorted = np.array(feature_names)[sorted_indices]
    corrs_sorted = np.array(correlations)[sorted_indices]
    colors = ['#FF6F61' if x < 0 else '#6BAED6' for x in corrs_sorted]

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8)) 
    bars = plt.barh(names_sorted, corrs_sorted, color=colors, edgecolor='black', alpha=0.8)

    # Trang trí
    plt.title('Feature Correlation with Target Variable', fontsize=15, fontweight='bold')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Features')
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Hiển thị giá trị hệ số trên thanh
    for bar, value in zip(bars, corrs_sorted):
        width = bar.get_width()
        offset = 0.01 if value >= 0 else -0.06 
        
        plt.text(width + offset, 
                 bar.get_y() + bar.get_height()/2, 
                 f'{value:.3f}', 
                 va='center', 
                 ha='left' if value >= 0 else 'left', 
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_comparison_histogram(data1, label1, data2, label2, feature_name):
    """
    Vẽ Histogram so sánh phân phối của 2 nhóm.
    """
    print(f"=== SO SÁNH PHÂN PHỐI: {feature_name} ===")
    
    plt.figure(figsize=(10, 6))
    
    # Vẽ Histogram nhóm 1 (Màu xanh)
    sns.histplot(data1, color='#6BAED6', label=label1, kde=True, stat="density", linewidth=0, alpha=0.6)
    
    # Vẽ Histogram nhóm 2 (Màu đỏ cam)
    sns.histplot(data2, color='#FF6F61', label=label2, kde=True, stat="density", linewidth=0, alpha=0.6)
    
    # Trang trí
    plt.title(f'So sánh Phân phối: {feature_name}', fontsize=14, fontweight='bold')
    plt.xlabel(feature_name)
    plt.ylabel('Mật độ (Density)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_scatter_risk_zone(X, y, feature_names):
    """
    Vẽ biểu đồ Scatter so sánh Total_Trans_Ct và Total_Trans_Amt
    để xác định 'Vùng nguy hiểm'.
    """
    print("=== PHÂN TÍCH VÙNG RỦI RO GIAO DỊCH ===")
    
    # Tìm index cột
    try:
        idx_ct = np.where(feature_names == 'Total_Trans_Ct')[0][0]
        idx_amt = np.where(feature_names == 'Total_Trans_Amt')[0][0]
    except IndexError:
        print("Lỗi: Không tìm thấy cột Total_Trans_Ct hoặc Total_Trans_Amt")
        return

    # Lấy dữ liệu
    trans_ct = X[:, idx_ct].astype(float)
    trans_amt = X[:, idx_amt].astype(float)

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=trans_ct, y=trans_amt, hue=y, 
                    palette={0: '#6BAED6', 1: '#FF6F61'}, 
                    alpha=0.6, edgecolor=None)
    
    plt.axvline(x=40, color='red', linestyle='--', label='Ngưỡng giao dịch (40)')
    plt.axhline(y=2500, color='darkred', linestyle='--', label='Ngưỡng chi tiêu (2500$)')
    plt.title('Phân tích Vùng Rủi Ro: Số lượng vs. Giá trị Giao dịch', fontsize=14, fontweight='bold')
    plt.xlabel('Tổng số giao dịch (Total_Trans_Ct)')
    plt.ylabel('Tổng giá trị giao dịch (Total_Trans_Amt)')
    plt.legend(title='Trạng thái (0: Existing, 1: Attrited)')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_churn_by_relationship_depth(X, y, feature_names):
    """
    Vẽ biểu đồ tỷ lệ rời bỏ theo số lượng sản phẩm
    trong nhóm khách hàng có giao dịch giảm sút.
    """
    print("=== PHÂN TÍCH ĐỘ SÂU MỐI QUAN HỆ TRONG NHÓM GIẢM GIAO DỊCH ===")
    
    try:
        idx_rel = np.where(feature_names == 'Total_Relationship_Count')[0][0]
        idx_chng = np.where(feature_names == 'Total_Ct_Chng_Q4_Q1')[0][0]
    except IndexError:
        print("Lỗi: Không tìm thấy cột cần thiết.")
        return

    rel_counts = X[:, idx_rel].astype(float)
    change_rate = X[:, idx_chng].astype(float)

    # Lọc nhóm rủi ro (Change < 0.6)
    low_change_mask = change_rate < 0.6
    
    group_y = y[low_change_mask]
    group_rel = rel_counts[low_change_mask]
    
    if len(group_y) == 0:
        print("Không có dữ liệu thỏa mãn điều kiện lọc.")
        return

    # Tính tỷ lệ churn theo từng mức relationship
    unique_rels = np.unique(group_rel)
    churn_rates = []
    
    for r in unique_rels:
        sub_y = group_y[group_rel == r]
        rate = np.mean(sub_y) if len(sub_y) > 0 else 0
        churn_rates.append(rate)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    sns.barplot(x=unique_rels, y=churn_rates, palette='Reds')
    avg_churn = np.mean(y)
    plt.axhline(y=avg_churn, color='blue', linestyle='--', label=f'Trung bình hệ thống ({avg_churn:.1%})')
    
    plt.title('Tỷ lệ Rời bỏ của nhóm "Giảm Giao Dịch" (<0.6) theo Số SP', fontsize=14, fontweight='bold')
    plt.xlabel('Số lượng sản phẩm (Total_Relationship_Count)')
    plt.ylabel('Tỷ lệ Rời bỏ (Churn Rate)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# --- 4. MA TRẬN TƯƠNG QUAN ---
def plot_correlation_matrix(X, feature_names):
    """
    Vẽ Heatmap tương quan.
    """
    print("=== MA TRẬN TƯƠNG QUAN ===")
    corr = np.corrcoef(X, rowvar=False)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm',
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# --- 5. BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH ---
def plot_confusion_matrix_custom(cm):
    """
    Vẽ Confusion Matrix.
    """
    print("=== CONFUSION MATRIX ===")
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
    fig.colorbar(cax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Dự báo (Predicted)', fontsize=12)
    plt.ylabel('Thực tế (Actual)', fontsize=12)
    plt.title('Ma trận nhầm lẫn', fontsize=14)

def plot_feature_importance_numpy(weights, feature_names, top_n=15):
    """
    Vẽ tầm quan trọng của biến dựa trên trọng số W (Logistic Regression).
    """
    print("=== FEATURE IMPORTANCE ===")
    
    # Sắp xếp trọng số theo giá trị tuyệt đối giảm dần
    indices = np.argsort(np.abs(weights))[::-1][:top_n]
    
    top_features = feature_names[indices]
    top_coeffs = weights[indices]
    
    # Đỏ (Dương - Tăng Churn), Xanh (Âm - Giảm Churn)
    colors = ['#FF6F61' if c > 0 else '#6BAED6' for c in top_coeffs]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_coeffs, color=colors, align='center')
    plt.yticks(range(top_n), top_features)
    plt.gca().invert_yaxis() 
    
    plt.title(f'Top {top_n} Yếu tố ảnh hưởng mạnh nhất (NumPy Model)', fontsize=14)
    plt.xlabel('Trọng số (Weight Magnitude)')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    print("Giải thích:")
    print("- Cột ĐỎ (>0): Tăng nguy cơ rời bỏ (Risk Factors).")
    print("- Cột XANH (<0): Giảm nguy cơ rời bỏ (Protective Factors).")

# === 6. HÀM LƯU ẢNH ===
def save_plot_image(filename, output_dir='images', dpi=300):
    """
    Hàm lưu biểu đồ hiện tại thành file ảnh.
    """
    # 1. Kiểm tra và tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục mới: {output_dir}")

    # 2. Tạo đường dẫn đầy đủ
    full_path = os.path.join(output_dir, filename)

    # 3. Lưu ảnh
    try:
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Đã lưu ảnh thành công: {full_path}")
    except Exception as e:
        print(f"Lỗi khi lưu ảnh: {e}")

# --- 7. KIỂM ĐỊNH GIẢ THUYẾT T-TEST (NUMPY) ---
def run_ttest_transactions(X, y, feature_names):
    """
    Thực hiện kiểm định T-test.
    """
    col_name = 'Total_Trans_Ct'
    print(f"\n{'='*60}")
    print(f"KIỂM ĐỊNH GIẢ THUYẾT: {col_name}")
    print(f"{'='*60}")

    # Tìm vị trí cột
    try:
        col_idx = np.where(feature_names == col_name)[0][0]
    except IndexError:
        print(f"Lỗi: Không tìm thấy cột {col_name}")
        return

    # Phát biểu giả thuyết
    print("Câu hỏi: Số lượng giao dịch trung bình của nhóm Rời bỏ có khác nhóm Hiện tại không?")
    print(" - H0: Mean_Churn == Mean_Existing")
    print(" - H1: Mean_Churn != Mean_Existing")

    # Tách dữ liệu
    churn_data = X[y == 1, col_idx].astype(float)
    existing_data = X[y == 0, col_idx].astype(float)

    # Tính toán các chỉ số thống kê
    n1 = len(existing_data)
    n2 = len(churn_data)
    
    mean1 = np.mean(existing_data)
    mean2 = np.mean(churn_data)
    
    # Tính phương sai mẫu
    var1 = np.var(existing_data, ddof=1)
    var2 = np.var(churn_data, ddof=1)

    print(f"\nTHỐNG KÊ MÔ TẢ:")
    print(f" - Nhóm Existing (n={n1}): Mean = {mean1:.2f}, Var = {var1:.2f}")
    print(f" - Nhóm Churn    (n={n2}): Mean = {mean2:.2f}, Var = {var2:.2f}")

    # Tính T-statistic
    numerator = mean1 - mean2
    denominator = np.sqrt((var1 / n1) + (var2 / n2))
    
    # Tránh chia cho 0
    if denominator == 0:
        t_stat = 0
    else:
        t_stat = numerator / denominator

    # So sánh với miền bác bỏ 
    critical_value = 1.96
    
    print(f"\nKẾT QUẢ KIỂM ĐỊNH:")
    print(f" - T-statistic tính toán: {t_stat:.4f}")
    print(f" - Ngưỡng (Critical Value): {critical_value} (mức tin cậy 95%)")

    print(f"\nKẾT LUẬN:")
    # So sánh trị tuyệt đối của t_stat với ngưỡng
    if abs(t_stat) > critical_value:
        print(" => |T-stat| > 1.96: BÁC BỎ H0.")
        print(" => Có sự khác biệt CÓ Ý NGHĨA THỐNG KÊ về số lượng giao dịch.")
    else:
        print(" => |T-stat| <= 1.96: CHẤP NHẬN H0.")
        print(" => Không có sự khác biệt đáng kể.")
    
    print("-" * 60)