
import pandas as pd
import os

# Input and Output paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\processed\master_table.csv"
OUTPUT_DIR = os.path.join(BASE_DIR, "Data_preprocessing", "my_data")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Read input
df = pd.read_csv(INPUT_CSV)

# 1. rna_data.csv
# Format: Target_RNA_sequence, Target_RNA_name, Target_RNA_ID
rna_df = df[['rna_sequence', 'pdb_id', 'pdb_id']].copy()
rna_df.columns = ['Target_RNA_sequence', 'Target_RNA_name', 'Target_RNA_ID']
# Remove duplicates if any (though we checked pdb_id is unique, rna seqs might not be, but RSAPred might want unique RNA entries? 
# The README says "RNA sequence-based features are obtained... and combined". 
# If we have duplicates in IDs, it's fine as long as unique IDs map to sequences. 
# Since pdb_id is unique, we are good.
rna_df.to_csv(os.path.join(OUTPUT_DIR, "rna_data.csv"), sep='\t', index=False)

# 2. mol_data.smi
# Format: SMILES Name
mol_df = df[['ligand_smiles', 'pdb_id']].copy()
# Check for duplicates? Since pdb_id is unique, this is fine.
# We need to ensure SMILES are valid.
mol_df.to_csv(os.path.join(OUTPUT_DIR, "mol_data.smi"), sep='\t', index=False, header=False)

# 3. sample_data.csv
# Format: Entry_ID, SMILES, Target_RNA_sequence, Molecule_name, Molecule_ID, Target_RNA_name, Target_RNA_ID, pKd
sample_df = pd.DataFrame()
sample_df['Entry_ID'] = df.index
sample_df['SMILES'] = df['ligand_smiles']
sample_df['Target_RNA_sequence'] = df['rna_sequence']
sample_df['Molecule_name'] = df['pdb_id']
sample_df['Molecule_ID'] = df['pdb_id']
sample_df['Target_RNA_name'] = df['pdb_id']
sample_df['Target_RNA_ID'] = df['pdb_id']
sample_df['pKd'] = df['pKd']

sample_df.to_csv(os.path.join(OUTPUT_DIR, "sample_data.csv"), sep='\t', index=False)

print("Data preparation complete.")
