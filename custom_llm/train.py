import os
import sys
import time
import json
import torch
import argparse
from torch.utils.data import DataLoader

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPT, TransformerConfig
from tokenizer import RobustTokenizer
from dataset import TextDataset

def get_resolved_checkpoint_path(default_path="c:/Rarey Temp/Ai/With Ai/custom_llm/checkpoint.pt"):
    g_drive_dir = "G:/My Drive/CustomLLM"
    if os.path.exists(g_drive_dir):
        return os.path.join(g_drive_dir, "checkpoint.pt")
    return default_path

def get_resolved_log_path(default_path="c:/Rarey Temp/Ai/With Ai/custom_llm/train_log.json"):
    g_drive_dir = "G:/My Drive/CustomLLM"
    if os.path.exists(g_drive_dir):
        return os.path.join(g_drive_dir, "train_log.json")
    return default_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom GPT model from scratch with checkpointing")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=-1, help="Max steps to train (-1 for unlimited)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context window size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N steps")
    
    default_checkpoint = get_resolved_checkpoint_path()
    default_log = get_resolved_log_path()
    
    parser.add_argument("--checkpoint_path", type=str, default=default_checkpoint, help="Path to checkpoint file")
    parser.add_argument("--log_path", type=str, default=default_log, help="Path to write training JSON log")
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, step, loss, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': model.config
    }
    torch.save(checkpoint, path)
    print(f"\n[CHECKPOINT] Saved checkpoint to {path} at Epoch {epoch}, Step {step}, Loss {loss:.4f}")

def load_checkpoint(path, device):
    if os.path.exists(path):
        print(f"Found checkpoint at {path}. Loading...")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        return checkpoint
    return None

def update_json_log(log_path, epoch, step, loss, vram_gb, speed):
    history = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    history.append({
        "epoch": epoch,
        "step": step,
        "loss": float(loss),
        "vram_gb": float(vram_gb),
        "speed": float(speed),
        "timestamp": time.time()
    })
    
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

def train():
    args = parse_args()
    
    # Create directory for checkpoints/logs if they don't exist
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    # 1. Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        
    # Check if checkpoint exists to align dataset block size early
    checkpoint = load_checkpoint(args.checkpoint_path, device)
    if checkpoint:
        checkpoint_block_size = checkpoint['config'].block_size
        if checkpoint_block_size != args.block_size:
            print(f"[ALIGNMENT] Adjusting dataset block size from {args.block_size} to {checkpoint_block_size} to match loaded checkpoint config.")
            args.block_size = checkpoint_block_size

    # Determine if we should force character tokenizer based on checkpoint vocab size
    force_type = None
    if checkpoint:
        if checkpoint['config'].vocab_size < 1000:
            print("[ALIGNMENT] Forcing Character-Level Tokenizer to match loaded checkpoint vocab size.")
            force_type = "char"

    # 2. Tokenizer and Dataset
    tokenizer = RobustTokenizer(force_type=force_type)
    dataset = TextDataset(tokenizer, block_size=args.block_size, split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device == "cuda"))
    
    # 3. Model setup
    if checkpoint:
        # Load from saved configuration to prevent architectural conflicts
        config = checkpoint['config']
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        last_loss = checkpoint['loss']
        print(f"Successfully resumed from Epoch {start_epoch}, Step {start_step} with Loss {last_loss:.4f}")
    else:
        # Create new model
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=0.1
        )
        model = GPT(config)
        start_epoch = 0
        start_step = 0
        last_loss = 0.0
        
    model.to(device)
    
    # 4. Optimizer and GradScaler for Mixed Precision (RTX 3050 benefits from FP16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    # 5. Training loop variables
    total_steps = start_step
    stop_signal_file = "c:/Rarey Temp/Ai/With Ai/custom_llm/stop_signal.txt"
    if os.path.exists(stop_signal_file):
        os.remove(stop_signal_file) # Clean start
        
    print("Starting training loop...")
    model.train()
    
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            
            for step, (x, y) in enumerate(dataloader):
                if total_steps < start_step:
                    # Skip steps if resuming in the middle of an epoch
                    # For simplicity, we just count steps, but this ensures we match total_steps
                    continue
                    
                t0 = time.time()
                x, y = x.to(device), y.to(device)
                
                # Forward and backward passes with Mixed Precision (AMP)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits, loss = model(x, y)
                    
                scaler.scale(loss).backward()
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Stats calculation
                t1 = time.time()
                dt = t1 - t0
                tokens_per_sec = (args.batch_size * args.block_size) / dt
                
                vram_gb = 0.0
                if device == "cuda":
                    vram_gb = torch.cuda.max_memory_allocated(device=device) / 1e9
                    
                total_steps += 1
                current_loss = loss.item()
                
                # Log step progress
                print(f"Step {total_steps} | Loss: {current_loss:.4f} | Speed: {tokens_per_sec:.0f} tok/s | VRAM: {vram_gb:.2f}GB", end="\r")
                
                # Write to logs
                update_json_log(args.log_path, epoch, total_steps, current_loss, vram_gb, tokens_per_sec)
                
                # Check for stop signal file (written by backend API to halt gracefully)
                if os.path.exists(stop_signal_file):
                    print("\n[STOP SIGNAL DETECTED] Pausing training process...")
                    save_checkpoint(model, optimizer, epoch, total_steps, current_loss, args.checkpoint_path)
                    os.remove(stop_signal_file)
                    print("Training stopped and checkpoint saved. Safe to close.")
                    return
                
                # Save checkpoint at regular intervals
                if total_steps % args.save_interval == 0:
                    save_checkpoint(model, optimizer, epoch, total_steps, current_loss, args.checkpoint_path)
                    
                # Break if reached target steps
                if args.steps > 0 and total_steps >= args.steps:
                    print(f"\nReached target step limit of {args.steps}.")
                    save_checkpoint(model, optimizer, epoch, total_steps, current_loss, args.checkpoint_path)
                    return
                    
    except KeyboardInterrupt:
        print("\n[KEYBOARD INTERRUPT] Saving checkpoint before exiting...")
        save_checkpoint(model, optimizer, epoch, total_steps, current_loss, args.checkpoint_path)
        sys.exit(0)
        
    # End of all epochs
    save_checkpoint(model, optimizer, args.epochs, total_steps, current_loss, args.checkpoint_path)
    print("Training completed successfully!")

if __name__ == "__main__":
    train()
