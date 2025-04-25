#!/usr/bin/env python3
# Script to run all data visualization and analysis scripts with the same dataset

import sys
import os
import subprocess

def run_all_scripts(dataset_path):
    """
    Run all data visualization and analysis scripts with the provided dataset path
    """
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the scripts to run (relative to this script's location)
    scripts = [
        os.path.join("DataVisualization", "histogram.py"),
        os.path.join("DataVisualization", "pair_plot.py"),
        os.path.join("DataVisualization", "scatter_plot.py"),
        os.path.join("DataAnalysis", "describe.py")
    ]
    
    # Run each script
    for script_path in scripts:
        full_script_path = os.path.join(current_dir, script_path)
        script_name = os.path.basename(script_path)
        
        print(f"\n{'=' * 50}")
        print(f"Running {script_name}...")
        print(f"{'=' * 50}")
        
        # Make script executable (helps but not required since we're using python3 explicitly)
        try:
            subprocess.run(["chmod", "+x", full_script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass  # Silently continue if chmod fails
        
        try:
            # Use python3 explicitly to run the script, regardless of whether it's executable
            # This is the most reliable method across platforms
            result = subprocess.run(["python3", full_script_path, dataset_path], 
                                   check=True, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            print(f"✅ {script_name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script_name}")
            print(f"Error details: {e.stderr}")
        except Exception as e:
            print(f"❌ Unexpected error running {script_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_all.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    run_all_scripts(dataset_path) 