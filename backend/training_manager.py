import os
import subprocess
import signal
import sys
import json
import psutil

class TrainingManager:
    def __init__(
        self, 
        script_path="c:/Rarey Temp/Ai/With Ai/custom_llm/train.py",
        log_path="c:/Rarey Temp/Ai/With Ai/custom_llm/train_log.json",
        stop_signal_path="c:/Rarey Temp/Ai/With Ai/custom_llm/stop_signal.txt",
        checkpoint_path="c:/Rarey Temp/Ai/With Ai/custom_llm/checkpoint.pt"
    ):
        self.script_path = script_path
        self.log_path = log_path
        self.stop_signal_path = stop_signal_path
        self.checkpoint_path = checkpoint_path
        self.process = None

    def get_status(self):
        """Check if training is currently running and read the latest stats."""
        is_running = False
        
        # Check if process is running
        if self.process is not None:
            # Check if subprocess ended
            if self.process.poll() is None:
                is_running = True
            else:
                self.process = None

        # Double check via process name if we started outside this instance
        if not is_running:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmd = proc.info['cmdline']
                    if cmd and any('train.py' in arg for arg in cmd) and any('custom_llm' in arg for arg in cmd):
                        is_running = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

        # Read latest log metrics
        latest_metrics = None
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    history = json.load(f)
                if history:
                    latest_metrics = history[-1]
            except Exception:
                pass

        # Check if checkpoint exists
        has_checkpoint = os.path.exists(self.checkpoint_path)
        checkpoint_size_mb = 0
        if has_checkpoint:
            checkpoint_size_mb = round(os.path.getsize(self.checkpoint_path) / (1024 * 1024), 2)

        return {
            "status": "training" if is_running else "paused",
            "has_checkpoint": has_checkpoint,
            "checkpoint_size_mb": checkpoint_size_mb,
            "metrics": latest_metrics
        }

    def start_training(self, batch_size=16, epochs=10, block_size=128, n_layer=4, n_head=4, n_embd=256, lr=6e-4):
        """Launch the training script in a background process."""
        status = self.get_status()
        if status["status"] == "training":
            return {"success": False, "message": "Training is already running."}

        # Clear any old stop signals
        if os.path.exists(self.stop_signal_path):
            os.remove(self.stop_signal_path)

        # Set up shell command
        python_executable = sys.executable  # Use current running python environment
        cmd = [
            python_executable, self.script_path,
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--block_size", str(block_size),
            "--n_layer", str(n_layer),
            "--n_head", str(n_head),
            "--n_embd", str(n_embd),
            "--lr", str(lr),
            "--checkpoint_path", self.checkpoint_path,
            "--log_path", self.log_path
        ]

        print(f"Launching training process: {' '.join(cmd)}")
        
        # Start background subprocess
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL, # Suppress stdout/stderr to avoid pipe blocking (logs write to JSON anyway)
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return {"success": True, "message": "Training process started in background."}
        except Exception as e:
            return {"success": False, "message": f"Failed to start training process: {str(e)}"}

    def stop_training(self):
        """Instruct the training loop to write a checkpoint and stop gracefully."""
        status = self.get_status()
        if status["status"] != "training":
            return {"success": False, "message": "Training is not currently running."}

        # Write stop signal file
        try:
            with open(self.stop_signal_path, "w") as f:
                f.write("stop")
            print("Stop signal file written. Waiting for process to exit...")
            return {"success": True, "message": "Pause signal sent. The model will save the checkpoint and stop shortly."}
        except Exception as e:
            return {"success": False, "message": f"Failed to write stop signal: {str(e)}"}

    def kill_training(self):
        """Forcefully kill training subprocess if it hangs (not recommended)."""
        if self.process is not None:
            try:
                self.process.kill()
                self.process = None
                return {"success": True, "message": "Training process forcefully terminated."}
            except Exception as e:
                return {"success": False, "message": f"Failed to terminate process: {str(e)}"}
        return {"success": False, "message": "No active subprocess tracking."}

    def get_log_history(self):
        """Return the entire training history for graph visualization."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

if __name__ == "__main__":
    # Test stub
    manager = TrainingManager()
    print("TrainingManager initialized. Status:", manager.get_status())
