import logging
import psutil
import os
import time
import torch
import gc
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tracemalloc
from pathlib import Path
from typing import Dict, Any, Optional, List
import networkx as nx
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class LoggerConfig:
    log_dir: str = "logs"
    metrics_dir: str = "metrics"
    plots_dir: str = "plots"
    log_level: str = "INFO"
    save_frequency: int = 1
    plot_metrics: bool = True
    monitor_network: bool = True
    custom_metrics: List[str] = None
    save_format: str = "csv"

class MLTrainingLogger:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize logger with optional config file"""
        self.config = self._load_config(config_path) if config_path else LoggerConfig()
        self._setup_directories()
        self._setup_logger()
        self.metrics_history = {}
        self.start_time = None
        tracemalloc.start()
        
    def _load_config(self, config_path: str) -> LoggerConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return LoggerConfig(**config_dict)
        
    def _setup_directories(self):
        """Create necessary directories for logs, metrics, and plots"""
        self.log_dir = Path(self.config.log_dir)
        self.metrics_dir = Path(self.config.metrics_dir)
        self.plots_dir = Path(self.config.plots_dir)
        
        for directory in [self.log_dir, self.metrics_dir, self.plots_dir]:
            directory.mkdir(exist_ok=True, parents=True)
            
    def _setup_logger(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('MLTrainingLogger')
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'training_log_{timestamp}.txt'
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def log_custom_metric(self, metric_name: str, value: Any, epoch: int):
        """Log a custom metric"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append({
            'epoch': epoch,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.info(f"Custom metric - {metric_name}: {value}")
        
    def _save_metrics(self):
        """Save metrics to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.config.save_format == 'csv':
            for metric, values in self.metrics_history.items():
                df = pd.DataFrame(values)
                df.to_csv(self.metrics_dir / f'{metric}_{timestamp}.csv', index=False)
        else:
            with open(self.metrics_dir / f'metrics_{timestamp}.json', 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
                
    def plot_metrics(self):
        """Generate visualization plots for metrics"""
        if not self.config.plot_metrics:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot each metric
        for metric_name, values in self.metrics_history.items():
            df = pd.DataFrame(values)
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x='epoch', y='value')
            plt.title(f'{metric_name} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.savefig(self.plots_dir / f'{metric_name}_plot_{timestamp}.png')
            plt.close()
            
        # Create correlation heatmap for all metrics
        if len(self.metrics_history) > 1:
            metric_dfs = []
            for metric_name, values in self.metrics_history.items():
                df = pd.DataFrame(values)
                df = df.rename(columns={'value': metric_name})
                metric_dfs.append(df[['epoch', metric_name]])
            
            combined_df = metric_dfs[0]
            for df in metric_dfs[1:]:
                combined_df = combined_df.merge(df, on='epoch')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(combined_df.corr(), annot=True, cmap='coolwarm')
            plt.title('Metric Correlations')
            plt.savefig(self.plots_dir / f'metric_correlations_{timestamp}.png')
            plt.close()
            
    def log_network_usage(self):
        """Log network usage statistics"""
        if not self.config.monitor_network:
            return
            
        net_io = psutil.net_io_counters()
        self.logger.info("=== Network Usage ===")
        self.logger.info(f"Bytes sent: {net_io.bytes_sent / (1024*1024):.2f} MB")
        self.logger.info(f"Bytes received: {net_io.bytes_recv / (1024*1024):.2f} MB")
        self.logger.info(f"Packets sent: {net_io.packets_sent}")
        self.logger.info(f"Packets received: {net_io.packets_recv}")
        
    def log_error(self, error: Exception, include_trace: bool = True):
        """Enhanced error logging with dependency analysis"""
        self.logger.error("=== Error Details ===")
        self.logger.error(f"Error type: {type(error).__name__}")
        self.logger.error(f"Error message: {str(error)}")
        
        if include_trace:
            self.logger.error("Traceback:")
            import traceback
            trace_str = traceback.format_exc()
            self.logger.error(trace_str)
            
            # Parse traceback for dependency issues
            if "ModuleNotFoundError" in trace_str or "ImportError" in trace_str:
                self._analyze_dependencies()
                
    def _analyze_dependencies(self):
        """Analyze and log package dependencies"""
        self.logger.info("=== Analyzing Dependencies ===")
        import pkg_resources
        
        deps = {}
        for dist in pkg_resources.working_set:
            deps[dist.key] = dist.version
            
        self.logger.info("Installed packages:")
        for package, version in sorted(deps.items()):
            self.logger.info(f"{package}: {version}")
            
    def log_training_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log training metrics with automatic visualization"""
        self.logger.info(f"=== Epoch {epoch} Metrics ===")
        
        for metric_name, value in metrics.items():
            self.log_custom_metric(metric_name, value, epoch)
            
        if epoch % self.config.save_frequency == 0:
            self._save_metrics()
            if self.config.plot_metrics:
                self.plot_metrics()
                
        self.log_memory_usage()
        self.log_gpu_utilization()
        if self.config.monitor_network:
            self.log_network_usage()
            
    def export_config(self):
        """Export current configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_path = self.log_dir / f'config_{timestamp}.yaml'
        
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
