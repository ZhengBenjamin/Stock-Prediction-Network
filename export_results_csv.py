import json
import csv
import os

def export_gpu_training_performance():
    gpu_data = [
        {
            'Model': 'LSTM',
            '1_GPU_Time_s': 76.0,
            '2_GPUs_Time_s': None,
            'Speedup': None,
            'Parallel_Efficiency_%': None,
            'Inference_Throughput_samples_s': None
        },
        {
            'Model': 'CNN_1D',
            '1_GPU_Time_s': 41.0,
            '2_GPUs_Time_s': 23.6,
            'Speedup': 1.74,
            'Parallel_Efficiency_%': 86.8,
            'Inference_Throughput_samples_s': 41336
        },
        {
            'Model': 'CNN_2D',
            '1_GPU_Time_s': 153.0,
            '2_GPUs_Time_s': 75.1,
            'Speedup': 2.04,
            'Parallel_Efficiency_%': 101.9,
            'Inference_Throughput_samples_s': 9866
        }
    ]
    with open('results/gpu_training_performance.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=gpu_data[0].keys())
        writer.writeheader()
        writer.writerows(gpu_data)
    
    print("GPU training performance saved to: results/gpu_training_performance.csv")


def export_prediction_accuracy():
    with open('results/prediction_accuracy.json', 'r') as f:
        data = json.load(f)
    csv_data = []
    for result in data:
        csv_data.append({
            'Model': result['model'].upper().replace('_', ' '),
            'GPUs': result['gpu_count'],
            'R2_Score': round(result['r2_score'], 4),
            'RMSE': round(result['rmse'], 6),
            'MAE': round(result['mae'], 6),
            'Correlation': round(result['correlation'], 4),
            'Direction_Accuracy_%': round(result['direction_accuracy'], 1),
            'Test_Samples': result['test_samples']
        })
    with open('results/prediction_accuracy.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
    
    print("Prediction accuracy saved to: results/prediction_accuracy.csv")


def export_combined_summary():
    with open('results/prediction_accuracy.json', 'r') as f:
        accuracy_data = json.load(f)
    training_times = {
        'LSTM_1': 76.0,
        'CNN 1D_1': 41.0,
        'CNN 1D_2': 23.6,
        'CNN 2D_1': 153.0,
        'CNN 2D_2': 75.1
    }
    
    combined = []
    for result in accuracy_data:
        model_name = result['model'].upper().replace('_', ' ')
        gpu_count = result['gpu_count']
        key = f"{model_name}_{gpu_count}"
        
        combined.append({
            'Model': model_name,
            'GPU_Count': gpu_count,
            'Training_Time_s': training_times.get(key, None),
            'R2_Score': round(result['r2_score'], 4),
            'Correlation': round(result['correlation'], 4),
            'Direction_Accuracy_%': round(result['direction_accuracy'], 1),
            'RMSE': round(result['rmse'], 6),
            'MAE': round(result['mae'], 6)
        })
    
    with open('results/Results_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=combined[0].keys())
        writer.writeheader()
        writer.writerows(combined)
    
    print("Complete summary saved to: results/Results_summary.csv")


def main():
    print("EXPORTING RESULTS TO CSV")
    os.makedirs('results', exist_ok=True)
    
    export_gpu_training_performance()
    export_prediction_accuracy()
    export_combined_summary()

    print("CSV FILES CREATED:")
    print("1. results/gpu_training_performance.csv")
    print("2. results/prediction_accuracy.csv")
    print("3. results/Results_summary.csv")


if __name__ == "__main__":
    main()
