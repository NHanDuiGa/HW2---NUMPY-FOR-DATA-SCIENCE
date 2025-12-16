import numpy as np

# --- 1. CLASS LOGISTIC REGRESSION ---
class LogisticRegressionNumPy:
    def __init__(self, learning_rate=0.01, n_iterations=1000, threshold=0.5):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        # Clip z để tránh tràn số
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for _ in range(self.n_iterations):
            # Tính toán tuyến tính: z = X.w + b
            linear_model = np.einsum('ij,j->i', X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Tính sai số
            error = y_predicted - y

            # Tính Gradient
            dw = (1 / n_samples) * np.einsum('ij,i->j', X, error)
            db = (1 / n_samples) * np.sum(error)

            # 3. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Lưu loss (Thêm epsilon để tránh log(0))
            y_pred_clipped = np.clip(y_predicted, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.losses.append(loss)

    def predict_proba(self, X):
        linear_model = np.einsum('ij,j->i', X, self.weights) + self.bias 
        return self._sigmoid(linear_model)

    def predict(self, X):
        y_predicted = self.predict_proba(X)
        return (y_predicted > self.threshold).astype(int)

# --- 2. CÁC HÀM HỖ TRỢ TÍNH TOÁN ---
def load_processed_data_numpy(data_dir):
    """Load dữ liệu .npy"""
    try:
        X_train = np.load(f'{data_dir}/train_X.npy')
        y_train = np.load(f'{data_dir}/train_y.npy')
        X_test = np.load(f'{data_dir}/test_X.npy')
        y_test = np.load(f'{data_dir}/test_y.npy')
        feature_names = np.load(f'{data_dir}/train_header.npy')
        return X_train, y_train, X_test, y_test, feature_names
    except Exception as e:
        print(f"Lỗi load data: {e}")
        return None, None, None, None, None

def calculate_metrics(y_true, y_pred):
    """Tính toán Accuracy, Precision, Recall, F1"""
    # Ép kiểu int để tính toán chính xác
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    cm = np.array([[TN, FP], [FN, TP]])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }