import subprocess
import sys
import os

def run_script(command, step_name):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_name}")
    print(f"{'='*60}")
    
    try:
        # Run the command and wait for it to finish
        subprocess.run(command, shell=True, check=True)
        print(f"✅ SUCCESS: {step_name} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {step_name} failed with exit code {e.returncode}.")
        sys.exit(1) # Stop the pipeline if a step fails

def main():
    print("🌟 Starting Industrial Anomaly Detection Pipeline 🌟")
    
    # 1. Generate / Prepare Data (DISABLED - Using Real Data)
    # if os.path.exists("generate_datasets.py"):
    #     run_script(f'"{sys.executable}" generate_datasets.py', "Generating Datasets")
        
    # 2. Train the Model (Assuming the file is named src/train.py or train.py)
    if os.path.exists(os.path.join("src", "train.py")):
        run_script(f'"{sys.executable}" src/train.py', "Model Training")
    elif os.path.exists("train.py"):
        run_script(f'"{sys.executable}" train.py', "Model Training")
        
    # 3. Evaluate the Model
    if os.path.exists(os.path.join("src", "evaluate.py")):
        run_script(f'"{sys.executable}" src/evaluate.py', "Model Evaluation")
        
    # 4. Run the Streamlit Dashboard
    if os.path.exists("realtime_dashboard.py"):
        print("\n🌐 Launching Real-Time Dashboard...")
        subprocess.run(f'"{sys.executable}" -m streamlit run realtime_dashboard.py', shell=True)

if __name__ == "__main__":
    main()