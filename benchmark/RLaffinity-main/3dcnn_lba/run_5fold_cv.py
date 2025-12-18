import os
import subprocess
import sys

# Configuration
# Use the Python executable from the virtual environment
PYTHON_EXEC = sys.executable 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED_DIR = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\processed"
# Cleaned structures are shared across folds
CLEANED_DATA_DIR = os.path.join(BASE_DIR, "my_data", "cleaned") 
MASTER_CSV = os.path.join(DATA_PROCESSED_DIR, "master_table.csv")

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD:  {cmd}")
    print(f"{ '='*60}\n")
    try:
        # Use shell=True for Windows compatibility with file paths
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    # Ensure output directories exist
    os.makedirs(os.path.join(BASE_DIR, "my_data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "output_5fold"), exist_ok=True)

    for fold in range(1, 6):
        print(f"\n\n##################################################################")
        print(f"                       STARTING FOLD {fold}")
        print(f"##################################################################")

        # Define paths for this fold's split files
        train_txt = os.path.join(DATA_PROCESSED_DIR, f"fold{fold}_train.txt")
        val_txt = os.path.join(DATA_PROCESSED_DIR, f"fold{fold}_val.txt")
        test_txt = os.path.join(DATA_PROCESSED_DIR, f"fold{fold}_test.txt")
        
        # Directory to store the LMDB database for this specific fold
        lmdb_output = os.path.join(BASE_DIR, "my_data", f"fold{fold}_lmdb")
        
        # Directory to store training results
        train_output_base = os.path.join(BASE_DIR, "output_5fold", f"fold{fold}")

        # 1. Run prepare_lmdb.py
        # This creates the LMDB database indices specific to the train/val/test split of this fold
        prepare_cmd = (
            f"\"{PYTHON_EXEC}\" prepare_lmdb.py "
            f"\"{CLEANED_DATA_DIR}\" "
            f"\"{lmdb_output}\" "
            f"-s "
            f"--train_txt \"{train_txt}\" "
            f"--val_txt \"{val_txt}\" "
            f"--test_txt \"{test_txt}\" "
            f"--score_path \"{MASTER_CSV}\""
        )
        run_command(prepare_cmd, f"Generating LMDB for Fold {fold}")

        # 2. Run train.py
        # The actual data is in the 'split' subdirectory created by prepare_lmdb.py
        lmdb_split_dir = os.path.join(lmdb_output, "split")
        
        train_cmd = (
            f"\"{PYTHON_EXEC}\" train.py "
            f"--data_dir \"{lmdb_split_dir}\" "
            f"--mode test "
            f"--output_dir \"{train_output_base}\" "
            f"--num_epochs 50"  
        )
        run_command(train_cmd, f"Training and Testing Fold {fold}")

    print("\n\n##################################################################")
    print("                  ALL 5 FOLDS COMPLETED SUCCESSFULLY")
    print("##################################################################")

if __name__ == "__main__":
    main()
