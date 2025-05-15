import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import numpy as np

class MetricsAnalyzer:
    """
    Utility for analyzing and visualizing profiling results.
    """
    
    def __init__(self, results_dir: str = None):
        """
        Initialize metrics analyzer.
        
        Args:
            results_dir: Directory containing profiling results JSON files
        """
        self.results_dir = results_dir
        self.results = []
        
        if results_dir and os.path.exists(results_dir):
            self.load_results(results_dir)
    
    def load_results(self, results_dir: str) -> None:
        """
        Load profiling results from JSON files.
        
        Args:
            results_dir: Directory containing profiling results JSON files
        """
        self.results = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(results_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        result = json.load(f)
                        self.results.append(result)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    
    def add_result(self, result: Dict) -> None:
        """
        Add a single profiling result.
        
        Args:
            result: Dictionary containing profiling results
        """
        if result not in self.results:
            self.results.append(result)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with profiling metrics
        """
        data = []
        
        for result in self.results:
            framework = result.get("framework", "Unknown")
            device = result.get("device", "Unknown")
            model_path = result.get("model_path", "Unknown")
            load_time = result.get("load_time_seconds", 0)
            
            batch_results = result.get("batch_results", {})
            for batch_size, metrics in batch_results.items():
                batch_size = int(batch_size)
                
                latency = metrics.get("latency", {})
                memory = metrics.get("memory", {})
                
                row = {
                    "framework": framework,
                    "device": device,
                    "model_path": model_path,
                    "batch_size": batch_size,
                    "load_time_seconds": load_time,
                    "mean_latency_ms": latency.get("mean", 0),
                    "p50_latency_ms": latency.get("median", 0),
                    "p90_latency_ms": latency.get("p90", 0),
                    "p99_latency_ms": latency.get("p99", 0),
                    "throughput": latency.get("throughput", 0),
                    "memory_used_mb": memory.get("used_mb", 0),
                }
                
                # Add GPU metrics if available
                if "gpu_allocated_mb" in memory:
                    row["gpu_memory_mb"] = memory.get("gpu_allocated_mb", 0)
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_latency_comparison(self, output_path: Optional[str] = None) -> None:
        """
        Plot latency comparison across frameworks and batch sizes.
        
        Args:
            output_path: Path to save the plot
        """
        df = self.to_dataframe()
        
        if df.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        ax = sns.barplot(x="batch_size", y="mean_latency_ms", hue="framework", data=df)
        
        plt.title("Mean Latency Comparison Across Frameworks", fontsize=16)
        plt.xlabel("Batch Size", fontsize=14)
        plt.ylabel("Mean Latency (ms)", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Framework", fontsize=12)
        
        # Annotate bars with values
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        plt.show()
    
    def plot_throughput_comparison(self, output_path: Optional[str] = None) -> None:
        """
        Plot throughput comparison across frameworks and batch sizes.
        
        Args:
            output_path: Path to save the plot
        """
        df = self.to_dataframe()
        
        if df.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        ax = sns.barplot(x="batch_size", y="throughput", hue="framework", data=df)
        
        plt.title("Throughput Comparison Across Frameworks", fontsize=16)
        plt.xlabel("Batch Size", fontsize=14)
        plt.ylabel("Throughput (items/second)", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Framework", fontsize=12)
        
        # Annotate bars with values
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        plt.show()
    
    def plot_memory_comparison(self, output_path: Optional[str] = None) -> None:
        """
        Plot memory usage comparison across frameworks and batch sizes.
        
        Args:
            output_path: Path to save the plot
        """
        df = self.to_dataframe()
        
        if df.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        ax = sns.barplot(x="batch_size", y="memory_used_mb", hue="framework", data=df)
        
        plt.title("Memory Usage Comparison Across Frameworks", fontsize=16)
        plt.xlabel("Batch Size", fontsize=14)
        plt.ylabel("Memory Usage (MB)", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Framework", fontsize=12)
        
        # Annotate bars with values
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        plt.show()
    
    def plot_load_time_comparison(self, output_path: Optional[str] = None) -> None:
        """
        Plot model load time comparison across frameworks.
        
        Args:
            output_path: Path to save the plot
        """
        df = self.to_dataframe().drop_duplicates(subset=["framework", "device", "model_path"])
        
        if df.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(10, 6))
        
        ax = sns.barplot(x="framework", y="load_time_seconds", hue="device", data=df)
        
        plt.title("Model Load Time Comparison", fontsize=16)
        plt.xlabel("Framework", fontsize=14)
        plt.ylabel("Load Time (seconds)", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Device", fontsize=12)
        
        # Annotate bars with values
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        plt.show()
    
    def generate_summary_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate a summary report of all profiling results.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Dictionary with summary metrics
        """
        df = self.to_dataframe()
        
        if df.empty:
            print("No data available for report generation")
            return {}
        
        # Framework comparison
        framework_summary = df.groupby("framework").agg({
            "mean_latency_ms": ["mean", "min"],
            "throughput": ["mean", "max"],
            "memory_used_mb": ["mean", "min"],
            "load_time_seconds": "mean"
        }).reset_index()
        
        # Best framework for different metrics
        best_latency = df.sort_values("mean_latency_ms").iloc[0]
        best_throughput = df.sort_values("throughput", ascending=False).iloc[0]
        best_memory = df.sort_values("memory_used_mb").iloc[0]
        best_load_time = df.sort_values("load_time_seconds").iloc[0]
        
        # Batch size scaling
        batch_scaling = df.groupby(["framework", "batch_size"]).agg({
            "mean_latency_ms": "mean",
            "throughput": "mean"
        }).reset_index()
        
        # Create summary
        summary = {
            "framework_summary": framework_summary.to_dict(),
            "best_performers": {
                "latency": {
                    "framework": best_latency["framework"],
                    "batch_size": best_latency["batch_size"],
                    "value_ms": best_latency["mean_latency_ms"]
                },
                "throughput": {
                    "framework": best_throughput["framework"],
                    "batch_size": best_throughput["batch_size"],
                    "value_items_per_sec": best_throughput["throughput"]
                },
                "memory": {
                    "framework": best_memory["framework"],
                    "batch_size": best_memory["batch_size"],
                    "value_mb": best_memory["memory_used_mb"]
                },
                "load_time": {
                    "framework": best_load_time["framework"],
                    "value_seconds": best_load_time["load_time_seconds"]
                }
            },
            "batch_scaling": batch_scaling.to_dict()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary report saved to {output_path}")
            
            # Also generate a text report
            text_report_path = output_path.replace('.json', '.txt')
            with open(text_report_path, 'w') as f:
                f.write("ML INFERENCE PROFILING SUMMARY REPORT\n")
                f.write("====================================\n\n")
                
                f.write("BEST PERFORMERS\n")
                f.write("--------------\n")
                f.write(f"Best Latency: {best_latency['framework']} (batch size {best_latency['batch_size']}): {best_latency['mean_latency_ms']:.2f} ms\n")
                f.write(f"Best Throughput: {best_throughput['framework']} (batch size {best_throughput['batch_size']}): {best_throughput['throughput']:.2f} items/sec\n")
                f.write(f"Best Memory Usage: {best_memory['framework']} (batch size {best_memory['batch_size']}): {best_memory['memory_used_mb']:.2f} MB\n")
                f.write(f"Best Load Time: {best_load_time['framework']}: {best_load_time['load_time_seconds']:.2f} seconds\n\n")
                
                f.write("FRAMEWORK COMPARISON\n")
                f.write("-------------------\n")
                f.write(df.groupby("framework").agg({
                    "mean_latency_ms": ["mean", "min", "max"],
                    "throughput": ["mean", "max"],
                    "memory_used_mb": ["mean", "min", "max"]
                }).to_string())
                f.write("\n\n")
                
                f.write("BATCH SIZE SCALING EFFICIENCY\n")
                f.write("-----------------------------\n")
                for framework in df["framework"].unique():
                    f.write(f"\n{framework}:\n")
                    framework_df = df[df["framework"] == framework]
                    baseline_latency = framework_df[framework_df["batch_size"] == 1]["mean_latency_ms"].values[0]
                    for i, row in framework_df.sort_values("batch_size").iterrows():
                        scaling_efficiency = (baseline_latency / row["mean_latency_ms"]) * row["batch_size"]
                        f.write(f"  Batch size {row['batch_size']}: {row['mean_latency_ms']:.2f} ms, " +
                                f"Throughput: {row['throughput']:.2f} items/sec, " +
                                f"Scaling efficiency: {scaling_efficiency:.2f}x\n")
            
            print(f"Text report saved to {text_report_path}")
        
        return summary
    
    def plot_scaling_efficiency(self, output_path: Optional[str] = None) -> None:
        """
        Plot scaling efficiency across batch sizes for each framework.
        
        Args:
            output_path: Path to save the plot
        """
        df = self.to_dataframe()
        
        if df.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        frameworks = df["framework"].unique()
        for framework in frameworks:
            framework_df = df[df["framework"] == framework].sort_values("batch_size")
            
            # Skip if only one batch size
            if len(framework_df) <= 1:
                continue
                
            batch_sizes = framework_df["batch_size"].values
            throughputs = framework_df["throughput"].values
            
            # Normalize to show scaling efficiency
            baseline_throughput = throughputs[0] / batch_sizes[0]  # throughput per item for batch size 1
            scaling_efficiency = [t / (bs * baseline_throughput) for t, bs in zip(throughputs, batch_sizes)]
            
            plt.plot(batch_sizes, scaling_efficiency, 'o-', label=framework)
        
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label="Linear Scaling")
        
        plt.title("Batch Size Scaling Efficiency", fontsize=16)
        plt.xlabel("Batch Size", fontsize=14)
        plt.ylabel("Scaling Efficiency\n(higher is better)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Framework", fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        plt.show()