import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import time
import json
import numpy as np

from models.LSTM.model import Model as LSTMModel
from models.CNN.cnn_model import CNNModel
from models.CNN.cnn_model_2d import CNN2DModel
from utils.stock_preprocessor import StockPreprocessor


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def get_gpu_memory_usage(device):
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / 1024**3
    return 0.0


def prepare_data(sequence_length, rank):
    preprocessor = StockPreprocessor(sequence_length=sequence_length)
    data = preprocessor.get_normalized_data()
    X = data[:, :-1, :]
    y = data[:, 1:, 4:5]
    if rank == 0:
        print(f"Data prepared - X: {X.shape}, y: {y.shape}")
    
    return X, y


class DistributedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) 
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def benchmark_inference(model, X, device, num_runs=100, warmup=10):
    model.eval()
    
    sample = torch.tensor(X[0:1], dtype=torch.float32).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
    torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(sample)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
    
    latencies = np.array(latencies) * 1000 # milliseconds
    batch_size = 32
    batch = torch.tensor(X[:batch_size], dtype=torch.float32).to(device)
    
    torch.cuda.synchronize()
    batch_start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch)
    
    torch.cuda.synchronize()
    batch_time = (time.time() - batch_start) / 10
    throughput = batch_size / batch_time
    metrics = {
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'throughput_samples_per_sec': float(throughput),
        'batch_size': batch_size,
        'num_runs': num_runs
    }
    
    return metrics

def train_distributed(rank, world_size, args):
    setup_distributed(rank, world_size)
    gpu_name = torch.cuda.get_device_name(rank)
    
    if rank == 0:
        print(f"DISTRIBUTED TRAINING - {world_size} GPUs")
        print(f"Model: {args.model.upper()}")
        if args.model == 'cnn' and args.cnn_type:
            print(f"CNN Type: {args.cnn_type.upper()}")
        
        print(f"\nGPU Configuration:")
        for i in range(world_size):
            gpu_name_i = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  Rank {i}: {gpu_name_i} ({gpu_mem:.1f} GB)")
        print()
    X, y = prepare_data(args.sequence_length, rank)
    dataset = DistributedDataset(X, y)
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    input_size = X.shape[2]
    
    if args.model == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(rank)
    elif args.model == 'cnn':
        if args.cnn_type == '2d':
            model = CNN2DModel(
                input_size=input_size,
                hidden_channels=args.hidden_channels,
                kernel_time=args.kernel_size,
                kernel_feat=args.kernel_size,
                dropout=args.dropout
            ).to(rank)
        else:
            model = CNNModel(
                input_size=input_size,
                hidden_channels=args.hidden_channels,
                kernel_size=args.kernel_size,
                dropout=args.dropout
            ).to(rank)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {total_params:,} parameters")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Total samples: {len(dataset)}")
        print(f"Batches per GPU: {len(dataloader)}\n")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if rank == 0:
        print("STARTING TRAINING")
    
    training_start = time.time()
    epoch_times = []
    
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_start = time.time()
        if epoch == 0:
            mem_used = get_gpu_memory_usage(rank)
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(rank)
            y_batch = y_batch.to(rank)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / len(dataloader)
        if rank == 0 and epoch % 10 == 0:
            print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")
    
    training_time = time.time() - training_start
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}\n")
    
    inference_metrics = None
    if rank == 0:
        print("RUNNING INFERENCE BENCHMARK")
        print(f"{'='*70}\n")
        
        inference_start = time.time()
        inference_metrics = benchmark_inference(
            model.module,
            X, 
            rank, 
            num_runs=100,
            warmup=10
        )
        inference_time = time.time() - inference_start
        
        print(f"Inference Benchmark Results:")
        print(f"  Mean Latency:    {inference_metrics['mean_latency_ms']:.3f} ms")
        print(f"  Std Latency:     {inference_metrics['std_latency_ms']:.3f} ms")
        print(f"  P50 Latency:     {inference_metrics['p50_latency_ms']:.3f} ms")
        print(f"  P95 Latency:     {inference_metrics['p95_latency_ms']:.3f} ms")
        print(f"  P99 Latency:     {inference_metrics['p99_latency_ms']:.3f} ms")
        print(f"  Throughput:      {inference_metrics['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Benchmark Time:  {inference_time:.2f}s")
        print()

    if rank == 0:
        model_name = args.model
        if args.model == 'cnn':
            model_name = f"cnn_{args.cnn_type}"

        baseline_file = f"results/training_{model_name}_1gpu_bs64.json"
        speedup = None
        efficiency = None
        
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
                baseline_time = baseline['total_training_time']
                speedup = baseline_time / training_time
                efficiency = (speedup / world_size) * 100
        
        results = {
            'model': model_name,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'gpu_count': world_size,
            'gpu_model': gpu_name,
            'gpu_memory_used_gb': mem_used,
            'batch_size_per_gpu': args.batch_size,
            'total_batch_size': args.batch_size * world_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'total_training_time': training_time,
            'avg_time_per_epoch': training_time / args.epochs,
            'min_epoch_time': min(epoch_times),
            'max_epoch_time': max(epoch_times),
            'final_loss': avg_loss,
            'training_throughput_samples_per_sec': (len(dataset) * args.epochs) / training_time,
            'speedup_vs_1gpu': speedup,
            'efficiency_percent': efficiency,
            'inference': inference_metrics,
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/training_{model_name}_{world_size}gpu_bs{args.batch_size}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("PERFORMANCE SUMMARY")
        print(f"Model:{model_name.upper()}")
        print(f"GPUs Used:{world_size}x {gpu_name}")
        print(f"Total Time:{training_time:.2f}s")
        print(f"Time per Epoch:{training_time/args.epochs:.3f}s")
        print(f"Training Throughput:{results['training_throughput_samples_per_sec']:.1f} samples/sec")
        print(f"Final Loss:{avg_loss:.6f}")
        
        if speedup:
            print(f"\nScaling Performance:")
            print(f"  Speedup vs 1 GPU:{speedup:.2f}x")
            print(f"  Efficiency:{efficiency:.1f}%")
        
        print(f"\nInference Performance:")
        print(f"  Latency (mean):{inference_metrics['mean_latency_ms']:.2f} ms")
        print(f"  Latency (P95):{inference_metrics['p95_latency_ms']:.2f} ms")
        print(f"  Throughput:{inference_metrics['throughput_samples_per_sec']:.0f} samples/sec")
        
        print(f"\nResults saved to: {filename}")
        
        checkpoint_dir = f"checkpoints/{model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/best_model_{world_size}gpu.pt"
        torch.save({
            'model_state': model.module.state_dict(),
            'loss': avg_loss,
            'gpu_count': world_size,
            'input_size': input_size,
            'results': results
        }, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}\n")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Distributed Stock Prediction Training with Inference Benchmarking')
    parser.add_argument('--model', choices=['lstm', 'cnn'], required=True)
    parser.add_argument('--cnn_type', choices=['1d', '2d'], default='1d')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--sequence_length', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--world_size', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
        if args.world_size == 0:
            print("ERROR: No GPUs detected!")
            return
    
    print(f"Initializing distributed training with {args.world_size} GPUs")
    
    mp.spawn(
        train_distributed,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()
