from rdkit import Chem
from rdkit.Chem import AllChem
import sys

def convert_smi_to_sdf(smi_file, sdf_file):
    print(f"Converting {smi_file} to {sdf_file} using RDKit...")
    
    with open(smi_file, 'r') as f:
        lines = f.readlines()
    
    writer = Chem.SDWriter(sdf_file)
    
    count = 0
    failed = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split('\t')
        if len(parts) < 2:
             parts = line.split() 
        
        if len(parts) >= 1:
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else f"Mol_{{count}}"
            
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol.SetProp("_Name", name)
                mol = Chem.AddHs(mol)
                # Generate 3D conformer
                # usage of random seed for reproducibility
                params = AllChem.ETKDG()
                params.randomSeed = 0xf00d
                res = AllChem.EmbedMolecule(mol, params)
                
                if res == 0:
                    writer.write(mol)
                    count += 1
                else:
                    print(f"Warning: Failed to embed molecule {name}. Writing 2D/flat structure.")
                    # Compute 2D coords so it's valid SDF
                    AllChem.Compute2DCoords(mol)
                    writer.write(mol)
                    failed += 1
            else:
                print(f"Error: Failed to parse SMILES: {smi}")
                failed += 1
        
    writer.close()
    print(f"Finished. Processed {count} molecules with 3D coords. {failed} fallbacks/failures.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_sdf.py <input.smi> <output.sdf>")
    else:
        convert_smi_to_sdf(sys.argv[1], sys.argv[2])