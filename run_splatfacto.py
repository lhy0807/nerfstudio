
import torch
import sys
from pathlib import Path
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.scripts.exporter import ExportGaussianSplat
from nerfstudio.utils import writer
import copy

def main():
    # 1. Configuration
    print("Setting up Splatfacto configuration...")
    # Get the default config for splatfacto
    config = copy.deepcopy(method_configs["splatfacto"])
    
    # Update configuration to match your requirements
    # "ns-train splatfacto --data ./data/nerfstudio/custom_test --pipeline.model.use_scale_regularization=True"
    
    # Set data path (using absolute path is safer)
    data_path = Path("./data/nerfstudio/custom_test_150").resolve()
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist.")
        sys.exit(1)
        
    config.pipeline.datamanager.data = data_path
    
    # Set model parameters
    # Note: splatfacto model config is in config.pipeline.model
    config.pipeline.model.use_scale_regularization = True
    
    # Set training parameters
    config.max_num_iterations = 30000  # Default is often 30k, ensures it exits
    # You can reduce this for testing purposes, e.g., 1000
    
    # IMPORTANT: Set this to True to ensure the training script exits after completion
    config.viewer.quit_on_train_completion = True
    
    # Set a timestamp or experiment name to make the output directory predictable
    # By default, nerfstudio uses current time. Let's fix it or capture it.
    config.set_timestamp()
    
    # Print where it will save
    print(f"Output directory: {config.get_base_dir()}")

    # Save the config file so it can be loaded later for export
    config.save_config()
    
    # 2. Training
    print("Starting training...")
    # TrainerConfig.setup() returns a Trainer instance
    trainer = config.setup(local_rank=0, world_size=1)
    
    # Run setup (loads data, builds model, etc.)
    trainer.setup()
    
    # Run training loop
    trainer.train()
    
    print("Training finished.")
    
    # 3. Exporting
    print("Starting export...")
    # "ns-export gaussian-splat --load-config outputs/custom_test/splatfacto/{timestamp}/config.yml --output-dir exports/splat/test --save-world-frame True"
    
    config_path = trainer.base_dir / "config.yml"
    output_dir = Path("exports/splat/test_150").resolve()
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    exporter = ExportGaussianSplat(
        load_config=config_path,
        output_dir=output_dir,
        save_world_frame=True
    )
    
    exporter.main()
    print(f"Export finished. Results in {output_dir}")

if __name__ == "__main__":
    main()
