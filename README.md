# Enhancing RNA-ligand Affinity Prediction using Transfer Learning

This project, developed for the CSCI 5526 course, introduces **GatorAffinity-RNA**, a deep learning model for predicting the binding affinity between RNA and small molecule ligands. It leverages transfer learning from a model pre-trained on large-scale protein-ligand synthetic data to overcome the scarcity of experimental RNA-ligand structural data.

Our results show that GatorAffinity-RNA achieves state-of-the-art performance on the PDBbind-RNA dataset, outperforming traditional docking methods and other deep learning models.

## Core Dependencies

- **Python 3.x**
- **PyTorch**
- **RDKit**: For chemical informatics.
- **PyMOL**: For 3D structure manipulation.

## Project Structure

```
.
├── GatorAffinity-main/      # Main model implementation (GatorAffinity-RNA)
│   ├── train.py             # Script for training the model
│   ├── inference.py         # Script for running inference
│   ├── data/dataset.py      # PyTorch dataset class, loads final .pkl files
│   └── environment.sh       # Shell script for an example Conda environment
├── benchmark/               # Baseline models (RSAPred, RLaffinity)
├── dataset/                 # Data preprocessing scripts and raw data
│   ├── NA-L/                # Raw PDB/SDF files for each complex (must be provided)
│   ├── INDEX_general_NL.2020R1.lst # PDBbind index file
│   ├── create_5fold_dataset.py   # Step 1: Creates CSV/TXT data splits
│   └── extract_rna_pocket.py     # Step 2: Extracts binding pockets using PyMOL
├── results/                 # Experiment results and plotting scripts
├── create_5fold_dataset.py  # (Duplicate) convenient access to the script
└── README.md
```

## Data Preprocessing Workflow

The training script `GatorAffinity-main/train.py` consumes final `.pkl` data files. However, this repository does not contain the script to generate them. The following is a reconstruction of the likely preprocessing pipeline based on the available scripts.

**Important**: The paths in the processing scripts are hardcoded (e.g., `C:\Users\Administrator\Desktop\...`). You **must** edit these scripts to point to the correct locations in your local filesystem.

### Step 1: Create Initial Data Splits

This step uses `create_5fold_dataset.py` to parse the main PDBbind index file, extract metadata, and generate CSV and TXT files that define the data splits for 5-fold cross-validation.

1.  **Prepare**: Make sure you have the `dataset/NA-L/` directory populated with the raw data and the `dataset/INDEX_general_NL.2020R1.lst` file is present.
2.  **Edit Script**: Modify the hardcoded paths in `create_5fold_dataset.py` to match your setup.
3.  **Run**:
    ```bash
    python create_5fold_dataset.py
    ```
4.  **Output**: This will generate `master_table.csv`, `5fold_splits.json`, and `fold{n}_{train/val/test}.txt` files inside the `dataset/processed/` directory.

### Step 2: Extract Binding Pockets

This step uses `extract_rna_pocket.py` to identify and extract the RNA atoms that form the binding pocket around the ligand. **This requires PyMOL to be installed and accessible.**

1.  **Edit Script**: Modify the hardcoded path to `na_l_dir` in `dataset/extract_rna_pocket.py`.
2.  **Run**:
    ```bash
    python dataset/extract_rna_pocket.py
    ```
3.  **Output**: This will create a `*_pocket_5A.pdb` file within each complex's subdirectory in `dataset/NA-L/`.

### Step 3: Generate Final `.pkl` Files (Script Missing)

**This is the missing link in the pipeline.** A script is required to perform the following actions:
1.  Read the `master_table.csv` and fold split files from Step 1.
2.  For each data entry, load the corresponding `*_pocket_5A.pdb` (from Step 2) and the ligand file.
3.  Process the 3D atomic data from the pocket and ligand, likely converting them into the graph-based data structures seen in `GatorAffinity-main/data/dataset.py` (e.g., atom types, coordinates, blocks).
4.  Combine this structured data with the affinity label (`pKd`).
5.  Save the final list of data dictionaries into fold-specific `.pkl` files (e.g., `fold1_train.pkl`, `fold1_val.pkl`, etc.) that the training script expects.

Without this script, the training process cannot be initiated.

## Training

Once you have successfully generated the `.pkl` data files, you can start the training.

1.  **Setup Environment**: Use `GatorAffinity-main/environment.sh` to create a Conda environment or install the dependencies manually (PyTorch, RDKit, etc.).
2.  **Run Training**: Execute `train.py`, providing paths to your generated data. **It is critical to override the hardcoded default paths.**

    ```bash
    cd GatorAffinity-main

    python train.py \
      --train_set_path /path/to/your/processed_data/fold1_train.pkl \
      --valid_set_path /path/to/your/processed_data/fold1_val.pkl \
      --pretrain_weights ./pretrain_model_weights.pt \
      --pretrain_config ./pretrain_model_config.json \
      --save_dir model_checkpoints/fold1 \
      --gpus 0 \
      --max_epoch 100 \
      --batch_size 4
    ```

Repeat this process for each of the 5 folds to reproduce the cross-validation results from the paper.
