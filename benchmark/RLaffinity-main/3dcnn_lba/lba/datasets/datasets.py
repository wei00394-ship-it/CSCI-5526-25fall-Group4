import os
import lmdb
import msgpack
import gzip
import io
import logging
import torch
import json
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tqdm
from lba.util import formats as fo

logger = logging.getLogger(__name__)

def serialize(x, serialization_format='msgpack'):
    if serialization_format == 'msgpack':
        return msgpack.packb(x, use_bin_type=True)
    elif serialization_format == 'json':
        # Handle numpy/pandas types for JSON serialization
        def default(o):
            if isinstance(o, (np.integer, np.int64)):
                return int(o)
            if isinstance(o, (np.floating, np.float64)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient='split')
            raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')
        return json.dumps(x, default=default).encode('utf-8')
    else:
        raise NotImplementedError

def deserialize(x, serialization_format='msgpack'):
    if serialization_format == 'msgpack':
        return msgpack.unpackb(x, raw=False)
    elif serialization_format == 'json':
        return json.loads(x.decode('utf-8'))
    else:
        raise NotImplementedError(f"Unknown format {serialization_format}")

class LMDBDataset(Dataset):
    def __init__(self, path, transform=None):
        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        self.transform = transform
        with self.env.begin() as txn:
            self.length = int(txn.get(b'num_examples').decode())
            self.serialization_format = txn.get(b'serialization_format').decode()

    def ids(self):
        # Iterate to find all ids. This is slow for large datasets but necessary if not stored separately.
        # For better performance, id_to_idx should be stored in LMDB.
        ids = []
        with self.env.begin() as txn:
            for i in range(self.length):
                value = txn.get(str(i).encode())
                with gzip.GzipFile(fileobj=io.BytesIO(value)) as f:
                    item = deserialize(f.read(), self.serialization_format)
                    ids.append(item['id'])
        return ids

    def ids_to_indices(self, ids):
        # Reverse mapping
        current_ids = self.ids()
        id_map = {id_: i for i, id_ in enumerate(current_ids)}
        return [id_map[id_] for id_ in ids if id_ in id_map]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError(index)
        
        with self.env.begin() as txn:
            value = txn.get(str(index).encode())
            if value is None:
                raise RuntimeError(f"Key {index} not found in LMDB")
            
            with gzip.GzipFile(fileobj=io.BytesIO(value)) as f:
                item = deserialize(f.read(), self.serialization_format)
        
        if self.transform:
            item = self.transform(item)
        return item

class SimpleFileDataset(Dataset):
    def __init__(self, file_list, file_type, transform=None, **kwargs):
        self.file_list = file_list
        self.file_type = file_type
        self.transform = transform
        self.kwargs = kwargs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        
        if self.file_type == 'pdb':
            # Assuming fo.read_any returns a Bio.PDB structure
            # And we need to convert it to DataFrame for LBA
            bp = fo.read_any(file_path)
            df = fo.bp_to_df(bp)
            item = {'atoms': df, 'id': os.path.basename(file_path), 'file_path': file_path}
            
        elif self.file_type == 'sdf':
            # Assuming fo.read_sdf_to_mol returns RDKit mols
            mols = fo.read_sdf_to_mol(file_path, **self.kwargs)
            if not mols:
                raise RuntimeError(f"No molecule found in {file_path}")
            mol = mols[0]
            # Conversion logic similar to lba.util.formats.mol_to_df
            # Note: fo.mol_to_df is available
            df = fo.mol_to_df(mol)
            
            # Extract bonds if needed
            bonds = fo.get_bonds_list_from_mol(mol) if self.kwargs.get('include_bonds') else None
            
            item = {'atoms': df, 'bonds': bonds, 'id': os.path.basename(file_path), 'file_path': file_path}
            
        else:
            raise NotImplementedError(f"File type {self.file_type} not supported in SimpleFileDataset")

        if self.transform:
            item = self.transform(item)
        return item

def load_dataset(path, format='lmdb', transform=None, **kwargs):
    if format == 'lmdb':
        return LMDBDataset(path, transform)
    elif format == 'pdb':
        # path is a list of files
        return SimpleFileDataset(path, 'pdb', transform, **kwargs)
    elif format == 'sdf':
        return SimpleFileDataset(path, 'sdf', transform, **kwargs)
    else:
        raise NotImplementedError(f"Unknown format {format}")

def make_lmdb_dataset(dataset, output_lmdb, filter_fn=None, serialization_format='json'):
    num_examples = len(dataset)
    logger.info(f'{num_examples} examples')
    
    # map_size 10GB
    env = lmdb.open(str(output_lmdb), map_size=int(1e10))
    
    with env.begin(write=True) as txn:
        try:
            i = 0
            for idx in tqdm.tqdm(range(num_examples)):
                try:
                    x = dataset[idx]
                    if filter_fn is not None and filter_fn(x):
                        continue
                        
                    # Add types info (mocking original behavior)
                    # x['types'] = {key: str(type(val)) for key, val in x.items()}
                    
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                        f.write(serialize(x, serialization_format))
                    compressed = buf.getvalue()
                    
                    txn.put(str(i).encode(), compressed)
                    i += 1
                except Exception as e:
                    logger.warning(f"Skipping index {idx} due to error: {e}")
                    continue
                    
            txn.put(b'num_examples', str(i).encode())
            txn.put(b'serialization_format', serialization_format.encode())
        except Exception as e:
            logger.error(f"Error creating LMDB: {e}")
            raise e

class CombinedLMDBStore:
    def __init__(self, datasets):
        self.datasets = datasets # List of LMDBDataset
    def close(self):
        for d in self.datasets:
            if hasattr(d, 'env'):
                d.env.close()

def build_datasets_from_three_csvs(data_dir, csv_train, csv_val, csv_test, pdb_col, transform_factory, random_seed):
    """
    Load train/val/test datasets from split LMDBs.
    Ignores CSV arguments as the data is assumed to be already split in data_dir.
    """
    train_path = os.path.join(data_dir, 'data', 'train')
    val_path = os.path.join(data_dir, 'data', 'val')
    test_path = os.path.join(data_dir, 'data', 'test')
    
    logger.info(f"Loading datasets from {data_dir}")
    
    train_dset = load_dataset(train_path, 'lmdb', transform=transform_factory('train', random_seed))
    val_dset = load_dataset(val_path, 'lmdb', transform=transform_factory('val', random_seed))
    test_dset = load_dataset(test_path, 'lmdb', transform=transform_factory('test', random_seed))
    
    store = CombinedLMDBStore([train_dset, val_dset, test_dset])
    
    return train_dset, val_dset, test_dset, store

# Dummy implementations for missing symbols if strictly needed imports exist
class PDBDataset(Dataset): pass
class PTGDataset(Dataset): pass
class SilentDataset(Dataset): pass
def get_file_list(*args, **kwargs): pass
def extract_coordinates_as_numpy_arrays(*args, **kwargs): pass