# src/training/training_callback.py

from transformers import TrainerCallback
import logging
from typing import Dict, Any
import time
from flask_socketio import SocketIO

class TrainingCallback(TrainerCallback):
    """Custom callback for training monitoring and metrics tracking"""
    def __init__(self, logger, socketio_instance=None):
        self.logger = logger
        self.metrics_history = []
        self.start_time = None
        self.training_started = False
        # Initialize SocketIO directly if no instance provided
        if socketio_instance is None:
            self.socketio = SocketIO(message_queue='redis://localhost:6379')
        else:
            self.socketio = socketio_instance

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.start_time = time.time()
        self.training_started = True
        self.logger.info("Training started")
        self.emit_log("Training started")
        self.emit_metrics({
            'status': 'started',
            'progress': 0,
            'step': 0
        })

    def on_log(self, args, state, control, logs: Dict[str, Any] = None, **kwargs):
        """Called when training metrics are logged"""
        if logs:
            # Add timestamp and step to metrics
            metrics = {
                'step': state.global_step,
                'timestamp': time.time(),
                'progress': (state.global_step / state.max_steps * 100) if state.max_steps else 0,
                **logs
            }

            self.metrics_history.append(metrics)
            
            # Log metrics
            metrics_str = " ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in metrics.items())
            self.logger.info(f"Training metrics - {metrics_str}")
            
            # Emit both log and metrics
            self.emit_log(f"Training metrics - {metrics_str}")
            self.emit_metrics(metrics)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics:
            eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
            self.emit_metrics(eval_metrics)
            self.emit_log(f"Evaluation metrics: {eval_metrics}")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        duration = time.time() - self.start_time
        self.logger.info(f"Training completed in {duration:.2f} seconds")
        self.emit_log(f"Training completed in {duration:.2f} seconds")
        self.emit_metrics({
            'status': 'completed',
            'duration': duration,
            'progress': 100
        })

    def emit_log(self, message: str):
        """Emit log message to the dashboard"""
        try:
            self.socketio.emit('log_update', {'message': message})
        except Exception as e:
            self.logger.error(f"Error emitting log: {e}")

    def emit_metrics(self, metrics: Dict[str, Any]):
        """Emit metrics to the dashboard"""
        try:
            self.socketio.emit('metrics_update', metrics)
        except Exception as e:
            self.logger.error(f"Error emitting metrics: {e}")