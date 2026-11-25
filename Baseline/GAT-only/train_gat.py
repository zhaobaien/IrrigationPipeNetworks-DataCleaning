from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
import os
import sys
import numpy as np
import time
from typing import Any, cast

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent

# Add the project root to the module search path.
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / "PI-GAT"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from graph_utils import build_graph_from_csv  # type: ignore
from timeseries_dataset import IrrigationTimeWindowDataset  # type: ignore
from model_gat import GAT  # type: ignore
import platform
import functools

os.environ["TQDM_DISABLE"] = "False"
os.environ["TQDM_COLOR"] = "True"


def _worker_init_fn_impl(worker_id: int, seed: int) -> None:
    import random as _r
    ws = seed + worker_id
    _r.seed(ws)
    np.random.seed(ws)
    torch.manual_seed(ws)


class LossGAT(nn.Module):
    """Compute the data-only training loss."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def __call__(self, out, batch, model=None, epoch=0):
        # Extract the predictions.
        node_hat = out["node_hat"]
        edge_hat = out["edge_hat"]
        device = node_hat.device

        # Check numerical stability.
        has_nan = (torch.isnan(node_hat).any() or 
                  torch.isnan(edge_hat).any() or 
                  torch.isinf(node_hat).any() or
                  torch.isinf(edge_hat).any())
        
        if has_nan:
            return {
                "L_total": torch.tensor(0.0, device=device, requires_grad=True),
                "L_data": torch.tensor(0.0, device=device, requires_grad=True),
                "L_rec_main": torch.tensor(0.0, device=device, requires_grad=True),
                "L_edge": torch.tensor(0.0, device=device, requires_grad=True),
            }

        # Data loss.
        mask_obs = batch.node_label_mask & (~batch.rest_last)
        mask_inp = batch.node_inpaint_mask & (~batch.rest_last)
        
        # Main node data loss.
        if mask_obs.any():
            L_rec_main = self.mse(node_hat[mask_obs], batch.y_node[mask_obs])
        else:
            L_rec_main = torch.tensor(0.0, device=device, requires_grad=True)

        # Edge data loss.
        mask_edge_valid = batch.edge_label_mask
        if mask_edge_valid.any():
            L_edge = self.mse(edge_hat[mask_edge_valid], batch.y_edge[mask_edge_valid])
        else:
            L_edge = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Total data loss.
        L_data = L_rec_main + 0.5 * L_edge
        L_total = L_data

        return {
            "L_total": L_total,
        }


def calculate_metrics(pred, target, mask):
    """Calculate mean absolute error and root-mean-square error."""
    if not mask.any():
        return {
            "mae": 0.0,
            "rmse": 0.0
        }
    
    pred_masked = pred[mask]
    target_masked = target[mask]
    
    # Regression metrics.
    mae = torch.mean(torch.abs(pred_masked - target_masked)).item()
    rmse = torch.sqrt(torch.mean((pred_masked - target_masked) ** 2)).item()
    
    return {
        "mae": mae,
        "rmse": rmse
    }


def set_seed(seed: int = 42, full_deterministic: bool = False):
    """Seed the random-number generators."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if full_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"✓ Random seed set to {seed}")


def create_optimizer_with_param_groups(model, lr=1e-3, weight_decay=1e-4):
    """Create the optimiser and its parameter groups."""
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


def train_one_epoch(
    model, 
    loader, 
    loss_fn, 
    optimizer, 
    device="cuda", 
    epoch=0,
    clip_grad=1.0,
    accumulation_steps=1
):
    """Train one epoch."""
    model.train()
    
    cum = {
        "L_total": 0.0,
    }
    
    metrics_cum = {
        "mae": 0.0,
        "rmse": 0.0
    }
    
    n = 0
    accum_step = 0
    nan_count = 0
    total_batches = len(loader)
    
    print_interval = max(1, total_batches // 10)

    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        
        out = model(batch)
        loss_dict = loss_fn(out, batch, model, epoch=epoch)
        
        # Backpropagate the loss.
        L_total = loss_dict['L_total']
        
        # Check the gradients for invalid values.
        if torch.isnan(L_total) or torch.isinf(L_total):
            nan_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        L_total.backward()
        accum_step += 1

        if accum_step >= accumulation_steps or idx == total_batches - 1:
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0

        # Accumulate the loss.
        cum["L_total"] += float(loss_dict["L_total"].item())
        
        # Compute and accumulate the metrics.
        metrics = calculate_metrics(
            pred=out['node_hat'],
            target=batch.y_node,
            mask=batch.node_label_mask
        )
        metrics_cum["mae"] += metrics["mae"]
        metrics_cum["rmse"] += metrics["rmse"]
        
        n += 1
        
        # Update the progress display.
        if (idx + 1) % print_interval == 0:
            progress = (idx + 1) / total_batches * 100
            print(f"\r  [{progress:5.1f}%] Loss={cum['L_total']/n:.4f}, "
                  f"MAE={metrics_cum['mae']/n:.4f}, "
                  f"RMSE={metrics_cum['rmse']/n:.4f}", end="")

    print()  # Terminate the progress line.
    
    if nan_count > 0:
        print(f"  警告: {nan_count} 个 NaN batch 被跳过")
    
    # Compute the epoch averages.
    cum["L_total"] /= max(n, 1)
    metrics_cum["mae"] /= max(n, 1)
    metrics_cum["rmse"] /= max(n, 1)
    
    cum.update(metrics_cum)
    return cum


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Evaluate the model on the validation set."""
    model.eval()
    
    cum = {
        "L_total": 0.0,
    }
    
    metrics_cum = {
        "mae": 0.0,
        "rmse": 0.0
    }
    
    n = 0

    for batch in loader:
        batch = batch.to(device)
        
        out = model(batch)
        losses = loss_fn(out, batch, model)
        
        cum["L_total"] += float(losses["L_total"].item())
        
        # Calculate the metrics.
        metrics = calculate_metrics(
            pred=out['node_hat'],
            target=batch.y_node,
            mask=batch.node_label_mask
        )
        metrics_cum["mae"] += metrics["mae"]
        metrics_cum["rmse"] += metrics["rmse"]
        
        n += 1

    # Compute the validation averages.
    cum["L_total"] /= max(n, 1)
    metrics_cum["mae"] /= max(n, 1)
    metrics_cum["rmse"] /= max(n, 1)
    
    cum.update(metrics_cum)
    return cum


def _unwrap_compiled_model(model: Any) -> nn.Module:
    """Expose the underlying torch module for static type checking."""
    return cast(nn.Module, getattr(model, "_orig_mod", model))


def save_checkpoint(model, optimizer, scheduler, epoch, best_val, bad, ckpt_path):
    """Save the training checkpoint."""
    orig_model = _unwrap_compiled_model(model)
    
    checkpoint = {
        "model_state_dict": orig_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val": best_val,
        "bad": bad,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"\n✓ Checkpoint saved at epoch {epoch}")


def main():
    """Train the GAT-only baseline."""
    # Training configuration.
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_EPOCHS = 200
    BATCH_SIZE = 64
    LR = 1e-3
    PATIENCE = 30
    WINDOW = 12
    
    # Path configuration.
    BASE_DIR = Path(__file__).parent
    PREPROCESS_DIR = project_dir / "Preprocess_Timeseries" / "outputs" / "preprocessed"
    
    TOPO_CSV = project_dir / "network_topology.csv"
    ZSCORE_CSV = PREPROCESS_DIR / "flow_5min_zscore.csv"
    RESTMASK_CSV = PREPROCESS_DIR / "mask_rest.csv"
    ZSCORE_STATS = PREPROCESS_DIR / "zscore_stats.json"
    OPEN_CSV = PREPROCESS_DIR / "flow_5min_imputed.csv"
    CREDIBILITY_CSV = project_dir / "PI-GAT" / "preprocessing" / "credibility_windows.csv"
    
    CKPT_PATH = BASE_DIR / "gat_ckpt.pth"
    FINAL_PATH = BASE_DIR / "gat_final.pth"
    LOG_PATH = BASE_DIR / "training_log.csv"
    
    # Initialise the random generators and device.
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    
    # Build the graph.
    print("\n=== 构建图结构 ===")
    data_graph, node2idx, pipe2idx, sensor2edge_idx = build_graph_from_csv(
        TOPO_CSV, weight_mode="conductance", add_self_loops=True
    )
    sensor2node_idx = data_graph.sensor2node_idx
    
    # Build the dataset.
    print("\n=== 构建数据集 ===")
    dataset = IrrigationTimeWindowDataset(
        data_graph=data_graph,
        zscore_path=ZSCORE_CSV,
        restmask_path=RESTMASK_CSV,
        sensor2node_idx=sensor2node_idx,
        sensor2edge_idx=sensor2edge_idx,
        window=WINDOW,
        random_mask_range=(0.1, 0.2),
        open_paths=OPEN_CSV,
        zscore_stats_path=ZSCORE_STATS,
        credibility_path=(CREDIBILITY_CSV if CREDIBILITY_CSV.exists() else None),
    )
    
    # Split the high-credibility observations used for single-stage training.
    def _cred_level_for_dataset_idx(ds: Any, di: int) -> int:
        if getattr(ds, 'has_credibility', False):
            try:
                t = int(ds.valid_t[int(di)])
                return int(ds.cred_level_by_t.get(t, 0))
            except Exception:
                return 0
        return 2  # default high

    idx_all = list(range(len(dataset)))
    idx_high = [i for i in idx_all if _cred_level_for_dataset_idx(dataset, i) >= 2]
    
    def _shuffle_inplace(xs: list[int], seed: int) -> None:
        import random
        rng = random.Random(seed)
        rng.shuffle(xs)

    def _split_70_15_15(xs: list[int]) -> tuple[list[int], list[int], list[int]]:
        n = len(xs)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        return xs[:n_tr], xs[n_tr : n_tr + n_va], xs[n_tr + n_va :]

    idx_high_s = idx_high.copy()
    _shuffle_inplace(idx_high_s, SEED + 11)
    
    train_idx_high, val_idx_high, test_idx_high = _split_70_15_15(idx_high_s)
    
    train_ds = torch.utils.data.Subset(dataset, train_idx_high)
    val_ds = torch.utils.data.Subset(dataset, val_idx_high)
    test_ds = torch.utils.data.Subset(dataset, test_idx_high)

    _train_g = torch.Generator()
    _train_g.manual_seed(SEED)
    _worker_init_fn = functools.partial(_worker_init_fn_impl, seed=SEED)
    train_loader = DataLoader(cast(Any, train_ds), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, generator=_train_g, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(cast(Any, val_ds), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, worker_init_fn=_worker_init_fn)
    
    print(f"数据集(high-only): 训练 {len(train_idx_high)}, 验证 {len(val_idx_high)}, 测试 {len(test_idx_high)}")
    
    # Build the model.
    print("\n=== 构建模型 ===")
    # Use local variables, assertions and casts to expose the graph dimensions to static type checkers.
    num_nodes_raw = getattr(data_graph, 'num_nodes', None)
    if num_nodes_raw is None:
        raise ValueError("data_graph.num_nodes is None — 请检查构建的图")
    # Tell the type checker that num_nodes is available.
    assert num_nodes_raw is not None
    N_nodes = int(num_nodes_raw)

    edge_index_raw = getattr(data_graph, 'edge_index', None)
    if edge_index_raw is None:
        raise ValueError("data_graph.edge_index is None — 请检查构建的图")
    # Treat edge_index explicitly as a tensor so its shape can be read.
    edge_index = cast(torch.Tensor, edge_index_raw)
    N_edges = int(edge_index.shape[1])  # Total edges, including self-loops.
    
    model = GAT(
        win=WINDOW,
        num_nodes=N_nodes,
        num_edges=N_edges,
        hidden=64,
        heads=8,
        dropout=0.1,
        in_feat=2
    ).to(DEVICE)
    
    # Optionally compile the model.
    USE_COMPILE = False  # Set to False to disable compilation.
    
    if USE_COMPILE and platform.system() == "Linux" and hasattr(torch, 'compile') and DEVICE == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ 模型已编译 (torch.compile)")
        except Exception as e:
            print(f"⚠ 模型编译失败，使用 eager 模式: {e}")
    else:
        if platform.system() == "Windows":
            print("✓ Windows 平台，使用 eager 模式（torch.compile 需要 Linux + Triton）")
        else:
            print("✓ 使用 eager 模式")
    
    loss_fn = LossGAT()
    
    # Optimiser and learning-rate scheduler.
    optimizer = create_optimizer_with_param_groups(
        model, lr=LR
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Training log.
    # Overwrite the training log on each run.
    with open(LOG_PATH, 'w') as f:
        f.write("Epoch,Train_Loss,Train_MAE,Train_RMSE,Val_Loss,Val_MAE,Val_RMSE,LR,Time\n")
    
    # Training loop.
    print("\n=== 开始训练 ===")
    best_val = float("inf")
    bad_epochs = 0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        
        print(f"\n[Epoch {epoch}/{MAX_EPOCHS}]")
        
        # Train on one epoch.
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE, epoch,
            clip_grad=1.0, accumulation_steps=1
        )
        
        # Validate the model.
        val_metrics = validate(
            model, val_loader, loss_fn, DEVICE
        )
        
        scheduler.step()
        
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print the epoch results.
        print(f"  Train: Loss={train_metrics['L_total']:.4f}, MAE={train_metrics['mae']:.4f}, RMSE={train_metrics['rmse']:.4f}")
        print(f"  Val:   Loss={val_metrics['L_total']:.4f}, MAE={val_metrics['mae']:.4f}, RMSE={val_metrics['rmse']:.4f}")
        print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
        
        # Record the results in the log.
        with open(LOG_PATH, 'a') as f:
            f.write(f"{epoch},{train_metrics['L_total']:.4f},{train_metrics['mae']:.4f},{train_metrics['rmse']:.4f},"                   f"{val_metrics['L_total']:.4f},{val_metrics['mae']:.4f},{val_metrics['rmse']:.4f},"                   f"{current_lr:.6f},{epoch_time:.1f}s\n")
        
        # Check for early stopping.
        if val_metrics['L_total'] < best_val:
            best_val = val_metrics['L_total']
            bad_epochs = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val, bad_epochs, CKPT_PATH)
            # Save the best model directly to the final checkpoint path.
            torch.save(_unwrap_compiled_model(model).state_dict(), FINAL_PATH)
            print(f"  ✓ 全局最佳模型 (Val Loss={best_val:.4f}) - 已保存到 gat_final.pth")
        else:
            bad_epochs += 1
            print(f"  Bad epochs: {bad_epochs}/{PATIENCE}")
        
        if bad_epochs >= PATIENCE:
            print(f"\n早停触发，在 epoch {epoch} 停止训练")
            break
    
    print("\n=== 训练完成 ===")
    print(f"最佳验证损失: {best_val:.4f}")


if __name__ == "__main__":
    main()
