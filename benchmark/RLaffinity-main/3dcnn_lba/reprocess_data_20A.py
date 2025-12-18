import os
import shutil
import numpy as np
import warnings
from Bio.PDB import PDBParser, MMCIFIO, Select, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress BioPython warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

# Configuration
SOURCE_ROOT = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\NA-L"
DEST_DIR = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\benchmark\RLaffinity-main\3dcnn_lba\my_data\cleaned"
CSV_PATH = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\processed\master_table.csv"
POCKET_RADIUS = 20.0

class PocketSelect(Select):
    def __init__(self, ligand_atoms, radius=20.0):
        self.ligand_coords = np.array([atom.coord for atom in ligand_atoms])
        self.radius = radius

    def accept_residue(self, residue):
        # Check distance for all atoms in residue (heavy calculation but safe)
        # Optimization: check CA/P/C4' first? RNA backbone: P, C4'
        for atom in residue:
            # Vectorized distance check
            dists = np.linalg.norm(self.ligand_coords - atom.coord, axis=1)
            if np.any(dists <= self.radius):
                return 1
        return 0

def get_pdb_ids_from_csv(csv_path):
    ids = []
    with open(csv_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if parts and len(parts) > 0:
                # Handle potential scientific notation restoration done manually or strings
                pdb_id = parts[0].lower()
                # Exclude known missing ones if they are still in CSV (though we checked they aren't)
                ids.append(pdb_id)
    return ids

def process_all():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    pdb_ids = get_pdb_ids_from_csv(CSV_PATH)
    print(f"Found {len(pdb_ids)} IDs in CSV.")
    
    parser = PDBParser(QUIET=True)
    io_cif = MMCIFIO()
    
    success_count = 0
    
    for pdb_id in pdb_ids:
        # Paths
        src_dir = os.path.join(SOURCE_ROOT, pdb_id)
        
        # Input files
        ligand_pdb = os.path.join(src_dir, f"{pdb_id}_ligand.pdb")
        nucleic_pdb = os.path.join(src_dir, f"{pdb_id}_nucleic_acid.pdb")
        ligand_sdf = os.path.join(src_dir, f"{pdb_id}_ligand.sdf")
        
        # Output files
        out_pocket_cif = os.path.join(DEST_DIR, f"{pdb_id}_pocket.cif")
        out_protein_cif = os.path.join(DEST_DIR, f"{pdb_id}_protein.cif")
        out_ligand_sdf = os.path.join(DEST_DIR, f"{pdb_id}_ligand.sdf")
        
        # Check inputs
        if not (os.path.exists(ligand_pdb) and os.path.exists(nucleic_pdb) and os.path.exists(ligand_sdf)):
            print(f"Skipping {pdb_id}: Missing source files.")
            continue
            
        try:
            # 1. Copy Ligand SDF
            shutil.copy(ligand_sdf, out_ligand_sdf)
            
            # 2. Load Structures
            ligand_struct = parser.get_structure('ligand', ligand_pdb)
            nucleic_struct = parser.get_structure('nucleic', nucleic_pdb)
            
            # Get Ligand Atoms
            ligand_atoms = list(ligand_struct.get_atoms())
            
            # 3. Extract Pocket (20A) & Save as CIF
            # We use Select class to filter residues
            selector = PocketSelect(ligand_atoms, radius=POCKET_RADIUS)
            io_cif.set_structure(nucleic_struct)
            io_cif.save(out_pocket_cif, select=selector)
            
            # 4. Convert Full Nucleic Acid to Protein CIF
            # RLaffinity usually needs the "protein" file for sequence extraction or context
            # We just save the full structure as .cif
            io_cif.set_structure(nucleic_struct)
            io_cif.save(out_protein_cif)
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"Processed {success_count} samples...")
                
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")

    print(f"Finished. Successfully processed {success_count} / {len(pdb_ids)} samples.")

if __name__ == "__main__":
    process_all()
