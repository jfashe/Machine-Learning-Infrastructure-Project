import time
import numpy as np
from monitor import TrainingMonitor

np.random.seed(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy_score(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    return np.mean(y_pred == y_true)

def make_synthetic_data(n_samples=500):
    # Two features, linearly separable-ish
    X = np.random.randn(n_samples, 2)
    true_w = np.array([2.0, -1.5])
    true_b = 0.3
    logits = X @ true_w + true_b
    probs = sigmoid(logits)
    y = (probs > 0.5).astype(int)
    return X, y

def train():
    X, y = make_synthetic_data()

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    lr = 0.1
    epochs = 30

    monitor = TrainingMonitor(log_dir="logs", run_name="baseline_run")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Forward pass
        logits = X @ w + b
        preds = sigmoid(logits)

        loss = binary_cross_entropy(y, preds)
        acc = accuracy_score(y, preds)

        # Gradients (logistic regression)
        dw = (X.T @ (preds - y)) / n_samples
        db = np.mean(preds - y)

        # Update
        w -= lr * dw
        b -= lr * db

        epoch_time = time.time() - t0
        monitor.log_epoch(epoch=epoch, loss=loss, accuracy=acc, lr=lr, duration_sec=epoch_time)

        print(f"Epoch {epoch:02d} | loss={loss:.4f} | acc={acc:.4f} | dt={epoch_time:.4f}s")

    summary = monitor.finalize()

    print("\n=== TRAINING SUMMARY ===")
    print(f"Epochs run: {summary['epochs_ran']}")
    print(f"Best loss: {summary['best_loss']:.4f}")
    print(f"Best accuracy: {summary['best_accuracy']:.4f}")
    print(f"Alerts triggered: {summary['num_alerts']}")
    print(f"Log file: {summary['log_file']}")

if __name__ == "__main__":
    train()