import os
from Bio.PDB import PDBParser, MMCIFIO
import shutil

# Config
SOURCE_ROOT = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\NA-L"
DEST_DIR = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\benchmark\RLaffinity-main\3dcnn_lba\my_data\cleaned"

samples = ["1f1t", "6e84"]

def convert_pdb_to_cif(pdb_path, cif_path, structure_id):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_id, pdb_path)
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(cif_path)
        print(f"Converted {pdb_path} -> {cif_path}")
    except Exception as e:
        print(f"Failed to convert {pdb_path}: {e}")

def process_sample(pdb_id):
    print(f"Processing {pdb_id}...")
    src_dir = os.path.join(SOURCE_ROOT, pdb_id)
    
    # 1. Ligand (SDF) - Copy
    src_lig = os.path.join(src_dir, f"{pdb_id}_ligand.sdf")
    dst_lig = os.path.join(DEST_DIR, f"{pdb_id}_ligand.sdf")
    if os.path.exists(src_lig):
        shutil.copy(src_lig, dst_lig)
        print(f"Copied ligand to {dst_lig}")
    else:
        print(f"Warning: Ligand not found at {src_lig}")

    # 2. Pocket (PDB -> CIF)
    # Source is _pocket_5A.pdb
    src_pocket = os.path.join(src_dir, f"{pdb_id}_pocket_5A.pdb")
    dst_pocket = os.path.join(DEST_DIR, f"{pdb_id}_pocket.cif")
    if os.path.exists(src_pocket):
        convert_pdb_to_cif(src_pocket, dst_pocket, pdb_id)
    else:
        print(f"Warning: Pocket not found at {src_pocket}")

    # 3. Protein/Nucleic Acid (PDB -> CIF)
    # Source is _nucleic_acid.pdb -> _protein.cif
    src_prot = os.path.join(src_dir, f"{pdb_id}_nucleic_acid.pdb")
    dst_prot = os.path.join(DEST_DIR, f"{pdb_id}_protein.cif")
    if os.path.exists(src_prot):
        convert_pdb_to_cif(src_prot, dst_prot, pdb_id)
    else:
        print(f"Warning: Nucleic acid file not found at {src_prot}")

if __name__ == "__main__":
    for s in samples:
        process_sample(s)
