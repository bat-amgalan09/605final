from train_deepspeed import train, CONFIG

def run_deepspeed_training():
    """Run DeepSpeed training on 2 GPUs and return metrics."""
    metrics = train(CONFIG, ds_config_path="ds_config.json")
    return metrics

if __name__ == "__main__":
    # Existing code (DDP, HF Accelerator, etc.)
    deepspeed_metrics = run_deepspeed_training()
    print("DeepSpeed Metrics:", deepspeed_metrics[-1])
