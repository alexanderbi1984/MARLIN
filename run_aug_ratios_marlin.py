import os
import yaml
import subprocess
import sys

# Configuration
base_config_path = "config/syracuse_mil_coral_xformer_au_3class.yaml"
script_path = "train_syracuse_biovid_aux.py"
ratios = [0.0, 0.25, 0.5, 0.75, 1]

def main():
    # Ensure we are in the correct directory or paths are correct
    if not os.path.exists(base_config_path):
        print(f"Error: Base config not found at {base_config_path}")
        print(f"Current working directory: {os.getcwd()}")
        return

    # Read the base config
    try:
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return

    # Loop through ratios
    for ratio in ratios:
        print(f"\n{'='*50}")
        print(f"Starting run with train_aug_ratio = {ratio}")
        print(f"{'='*50}\n")

        # Update configuration values
        config['train_aug_ratio'] = ratio
        
        # Construct save_dir: Syracuse/xformer_mil_mma_clip_level_auxiliary/3class_x_aug_ratio_random_features
        save_dir = f"Syracuse/xformer_mil_au_clip_level_auxiliary/3class_{ratio}_aug_ratio_rgb_features"
        config['save_dir'] = save_dir
        
        print(f"Configured save_dir: {save_dir}")

        # Create a temporary config file for this run
        temp_config_filename = f"temp_config_ratio_{ratio}.yaml"
        temp_config_path = os.path.join("config", temp_config_filename)
        
        try:
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error writing temporary config file {temp_config_path}: {e}")
            continue

        # Construct the command to run the training script
        # using sys.executable ensures we use the same python interpreter
        cmd = [
            sys.executable,
            script_path,
            "--config",
            temp_config_path
        ]

        try:
            # Run the command and wait for it to complete
            subprocess.run(cmd, check=True)
            print(f"\nSuccessfully finished run for ratio {ratio}")
        except subprocess.CalledProcessError as e:
            print(f"\nError running for ratio {ratio}. Command failed with exit code {e.returncode}")
            # We continue to the next experiment even if one fails
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            # Cleanup current temp file before exit
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            sys.exit(1)
        finally:
            # Cleanup the temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    print(f"\n{'='*50}")
    print("All experiments completed.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()


