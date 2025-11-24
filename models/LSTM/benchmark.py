import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional, Tuple, Dict
import seaborn as sns

class ModelBenchmark:
    def __init__(self, model, preprocessor, device: str = "cuda"):
        self.model = model
        self.preprocessor = preprocessor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Tuple:
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,  # Mean Absolute Percentage Error
            'Directional_Accuracy': self._directional_accuracy(y_true, y_pred)
        }
        return metrics
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return 0.0
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        return (correct / total) * 100 if total > 0 else 0.0
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                 train_ratio: float = 0.7, 
                 val_ratio: float = 0.15) -> Dict:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(
            X, y, train_ratio, val_ratio
        )
        
        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)
        test_pred = self.predict(X_test)
        
        self.results = {
            'train': {
                'metrics': self.calculate_metrics(y_train, train_pred),
                'true': y_train,
                'pred': train_pred
            },
            'validation': {
                'metrics': self.calculate_metrics(y_val, val_pred),
                'true': y_val,
                'pred': val_pred
            },
            'test': {
                'metrics': self.calculate_metrics(y_test, test_pred),
                'true': y_test,
                'pred': test_pred
            }
        }
        
        return self.results
    
    def print_metrics(self):
        if not self.results:
            print("No results available.")
            return
        print("Model Evaluation Results")
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in self.results:
                print(f"\n{split_name.upper()} SET METRICS:")
                print("-" * 70)
                metrics = self.results[split_name]['metrics']
                
                for metric_name, value in metrics.items():
                    if metric_name == 'MAPE':
                        print(f"  {metric_name:25s}: {value:>10.2f}%")
                    elif metric_name == 'Directional_Accuracy':
                        print(f"  {metric_name:25s}: {value:>10.2f}%")
                    else:
                        print(f"  {metric_name:25s}: {value:>10.6f}")
        print("\n")
    
    def plot_predictions(self, max_points: Optional[int] = 500):
        if not self.results:
            print("No results available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Prediction Analysis', fontsize=16, fontweight='bold')
        ax = axes[0, 0]
        y_test = self.results['test']['true']
        y_test_pred = self.results['test']['pred']
        
        points_to_plot = min(max_points, len(y_test))
        indices = np.arange(points_to_plot)
        
        ax.plot(indices, y_test[:points_to_plot], label='Actual', alpha=0.7, linewidth=2)
        ax.plot(indices, y_test_pred[:points_to_plot], label='Predicted', alpha=0.7, linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Price')
        ax.set_title('Test Set: Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.scatter(y_test, y_test_pred, alpha=0.5, s=20)
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f"Test Set: Scatter Plot (R² = {self.results['test']['metrics']['R2']:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        errors = y_test - y_test_pred
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Test Set: Error Distribution (MAE = {self.results['test']['metrics']['MAE']:.6f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        metrics_to_compare = ['RMSE', 'MAE', 'R2']
        splits = ['train', 'validation', 'test']
        
        x = np.arange(len(metrics_to_compare))
        width = 0.25
        
        for i, split in enumerate(splits):
            values = [self.results[split]['metrics'][m] for m in metrics_to_compare]
            ax.bar(x + i*width, values, width, label=split.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Metrics Comparison Across Splits')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_to_compare)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self):
        if not self.results:
            print("No results available.")
            return
        
        y_test = self.results['test']['true']
        y_test_pred = self.results['test']['pred']
        residuals = y_test - y_test_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Residual Analysis (Test Set)', fontsize=14, fontweight='bold')
        
        ax = axes[0]
        ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals Over Time')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Check)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()