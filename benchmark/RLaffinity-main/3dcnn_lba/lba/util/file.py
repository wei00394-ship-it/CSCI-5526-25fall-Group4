"""File-related utilities."""
import os
from pathlib import Path

def find_files(path, suffix, relative=None):
    """
    Find all files in path with given suffix.

    :param path: Directory in which to find files.
    :type path: Union[str, Path]
    :param suffix: Suffix determining file type to search for.
    :type suffix: str
    :param relative: Flag to indicate whether to return absolute or relative path.

    :return: list of paths to all files with suffix sorted by their names.
    :rtype: list[Path]
    """
    path = Path(path)
    files = []
    
    # Walk through directory
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(f".{suffix}"):
                full_path = Path(root) / filename
                if relative:
                    files.append(full_path.relative_to(path))
                else:
                    files.append(full_path)
    
    # Sort to ensure deterministic order, matching 'sort' command behavior
    files.sort(key=lambda p: str(p))
    return files

def get_ligand_code(path):
    """
    Extract ligand code from filename.
    
    :param path: Path to file.
    :type path: str

    :return: Ligand code.
    :rtype: str
    """
    # Original logic: str(path).split('/')[-1][5:-4]
    # This assumes filename format PDB_LIGAND.EXT (e.g. 1arj_ligand.sdf)
    return Path(path).name[5:-4]

def get_pdb_code(path):
    """
    Extract 4-character PDB ID code from full path.

    :param path: Path to file.
    :type path: str

    :return: PDB ID.
    :rtype: str
    """
    # Original logic: path.split('/')[-1][:4].lower()
    return Path(path).name[:4].lower()


def get_pdb_name(path):
    """
    Extract filename for PDB file from full path.

    :param path: Path to file.
    :type path: str

    :return: Filename.
    :rtype: str
    """
    return Path(path).name