import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import Descriptors
import json

def parse_binding_affinity(binding_str):
    """
    解析亲和力字符串并转换为pKd值
    支持格式: Kd=1mM, Ki=342uM, IC50=0.33uM等
    返回: pKd值 (即 -log10(Kd in M))
    """
    binding_str = binding_str.strip()

    # 提取数值和单位
    # 匹配模式: Kd=1mM, Ki=342uM, IC50=0.33uM, Kd~10nM, Kd<0.18nM等
    pattern = r'(Kd|Ki|IC50)\s*[=~<>]+\s*([\d.]+)\s*([a-zA-Z]+)'
    match = re.search(pattern, binding_str)

    if not match:
        return None

    value = float(match.group(2))
    unit = match.group(3).upper()

    # 转换单位到M (摩尔)
    unit_conversion = {
        'M': 1,
        'MM': 1e-3,  # 毫摩尔
        'UM': 1e-6,  # 微摩尔
        'NM': 1e-9,  # 纳摩尔
        'PM': 1e-12, # 皮摩尔
    }

    if unit not in unit_conversion:
        print(f"Warning: Unknown unit {unit} in {binding_str}")
        return None

    # 转换为摩尔浓度
    kd_in_molar = value * unit_conversion[unit]

    # 计算pKd = -log10(Kd)
    pkd = -np.log10(kd_in_molar)

    return pkd

def parse_index_file(index_file_path):
    """
    解析INDEX文件,提取PDB ID和亲和力数据
    """
    data = {}

    with open(index_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line.startswith('#') or not line:
                continue

            # 解析格式: 1arj   NMR  1996  Kd=1mM        // 1arj.pdf (ARG) ...
            parts = line.split()
            if len(parts) < 4:
                continue

            pdb_id = parts[0]
            resolution = parts[1]
            year = parts[2]
            binding_data = parts[3]

            # 转换为pKd
            pkd = parse_binding_affinity(binding_data)

            if pkd is not None:
                data[pdb_id] = {
                    'pdb_id': pdb_id,
                    'resolution': resolution,
                    'year': year,
                    'binding_data': binding_data,
                    'pKd': pkd
                }

    return data

def extract_rna_sequence_from_pdb(pdb_file_path):
    """
    从PDB文件中提取RNA序列
    """
    nucleotides = []
    seen_residues = set()

    with open(pdb_file_path, 'r') as f:
        for line in f:
            # 解析SEQRES记录 (更准确的序列信息)
            if line.startswith('SEQRES'):
                chain = line[11:12].strip()
                residues = line[19:].split()
                for res in residues:
                    if res in ['A', 'U', 'G', 'C', 'T']:
                        nucleotides.append(res)
                continue

    # 如果SEQRES有数据就用SEQRES
    if nucleotides:
        return ''.join(nucleotides)

    # 否则从ATOM记录提取
    nucleotides = []
    seen_residues = set()

    with open(pdb_file_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # 解析PDB ATOM行
                chain = line[21:22].strip()
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()

                # RNA核苷酸
                nucleotide_map = {
                    'G': 'G', 'A': 'A', 'C': 'C', 'U': 'U', 'T': 'T',
                    'DG': 'G', 'DA': 'A', 'DC': 'C', 'DT': 'T',
                    'RG': 'G', 'RA': 'A', 'RC': 'C', 'RU': 'U'
                }

                if res_name in nucleotide_map:
                    res_id = f"{chain}_{res_seq}"
                    if res_id not in seen_residues:
                        seen_residues.add(res_id)
                        nucleotides.append((chain, int(res_seq), nucleotide_map[res_name]))

    # 按链和序列号排序
    nucleotides.sort(key=lambda x: (x[0], x[1]))

    # 提取序列
    sequence = ''.join([n[2] for n in nucleotides])

    return sequence

def extract_ligand_smiles_from_sdf(sdf_file_path):
    """
    从SDF文件中提取配体并转换为SMILES
    """
    try:
        # 使用RDKit读取SDF文件
        suppl = Chem.SDMolSupplier(sdf_file_path)

        for mol in suppl:
            if mol is not None:
                # 转换为SMILES
                smiles = Chem.MolToSmiles(mol)
                return smiles

        return None
    except Exception as e:
        print(f"Error reading SDF file {sdf_file_path}: {e}")
        return None

def process_dataset(dataset_dir, index_file):
    """
    处理整个数据集
    """
    # 解析INDEX文件
    print("Parsing INDEX file...")
    binding_data_dict = parse_index_file(index_file)
    print(f"Found {len(binding_data_dict)} entries with binding data in INDEX")

    # 数据目录
    nal_dir = os.path.join(dataset_dir, 'NA-L')

    # 获取所有可用的样本目录
    available_samples = [d for d in os.listdir(nal_dir)
                         if os.path.isdir(os.path.join(nal_dir, d))]

    print(f"Found {len(available_samples)} sample directories in NA-L")

    results = []
    processed_count = 0
    skipped_count = 0

    for pdb_id in available_samples:
        # 检查是否在INDEX中
        if pdb_id not in binding_data_dict:
            skipped_count += 1
            continue

        sample_dir = os.path.join(nal_dir, pdb_id)

        # 文件路径
        rna_pdb_file = os.path.join(sample_dir, f"{pdb_id}_nucleic_acid.pdb")
        ligand_sdf_file = os.path.join(sample_dir, f"{pdb_id}_ligand.sdf")

        # 检查文件是否存在
        if not os.path.exists(rna_pdb_file):
            print(f"Warning: RNA PDB file not found for {pdb_id}")
            continue

        if not os.path.exists(ligand_sdf_file):
            print(f"Warning: Ligand SDF file not found for {pdb_id}")
            continue

        print(f"Processing {pdb_id}...")

        # 提取RNA序列
        rna_seq = extract_rna_sequence_from_pdb(rna_pdb_file)

        # 提取配体SMILES
        ligand_smiles = extract_ligand_smiles_from_sdf(ligand_sdf_file)

        if not rna_seq:
            print(f"Warning: No RNA sequence extracted for {pdb_id}")
            continue

        if not ligand_smiles:
            print(f"Warning: No ligand SMILES extracted for {pdb_id}")
            continue

        # 合并数据
        entry_data = binding_data_dict[pdb_id]
        results.append({
            'pdb_id': pdb_id,
            'rna_sequence': rna_seq,
            'ligand_smiles': ligand_smiles,
            'resolution': entry_data['resolution'],
            'year': entry_data['year'],
            'binding_data': entry_data['binding_data'],
            'pKd': entry_data['pKd']
        })

        processed_count += 1

    print(f"\n=== Processing Summary ===")
    print(f"Total samples in INDEX: {len(binding_data_dict)}")
    print(f"Total sample directories: {len(available_samples)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (not in INDEX): {skipped_count}")

    return results

def create_5fold_split_with_validation(data, random_seed=42, val_ratio=0.125):
    """
    创建5折交叉验证划分,并为每折从训练集中划分出验证集

    参数:
        data: 数据列表
        random_seed: 随机种子
        val_ratio: 验证集占训练集的比例 (默认0.125,即训练集的12.5%)

    对于140个样本:
        - 测试集: 28个 (20%)
        - 剩余112个中划分:
          - 训练集: 98个 (70% of total)
          - 验证集: 14个 (10% of total)
    """
    np.random.seed(random_seed)

    # 创建索引数组
    n_samples = len(data)
    indices = np.arange(n_samples)

    # 使用KFold进行5折划分
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    fold_splits = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(kfold.split(indices)):
        # 从训练集中再划分出验证集
        # 使用不同的随机种子以确保每折的验证集划分不同
        np.random.seed(random_seed + fold_idx)

        # 打乱训练+验证索引
        np.random.shuffle(trainval_idx)

        # 计算验证集大小
        n_val = int(len(trainval_idx) * val_ratio)

        # 划分训练集和验证集
        val_idx = trainval_idx[:n_val]
        train_idx = trainval_idx[n_val:]

        fold_splits.append({
            'fold': fold_idx + 1,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'test_indices': test_idx.tolist(),
            'train_ids': [data[i]['pdb_id'] for i in train_idx],
            'val_ids': [data[i]['pdb_id'] for i in val_idx],
            'test_ids': [data[i]['pdb_id'] for i in test_idx],
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx)
        })

    return fold_splits

def main():
    # 设置路径
    dataset_dir = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset"
    index_file = os.path.join(dataset_dir, "INDEX_general_NL.2020R1.lst")

    # 检查路径是否存在
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return

    if not os.path.exists(index_file):
        print(f"Error: INDEX file not found: {index_file}")
        return

    # 处理数据集
    results = process_dataset(dataset_dir, index_file)

    if not results:
        print("No data processed!")
        return

    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {len(results)}")

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 输出目录
    output_dir = os.path.join(dataset_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # 保存总表
    master_table_path = os.path.join(output_dir, "master_table.csv")
    df.to_csv(master_table_path, index=False)
    print(f"\nMaster table saved to: {master_table_path}")

    # 显示统计信息
    print(f"\npKd statistics:")
    print(f"  Range: {df['pKd'].min():.2f} - {df['pKd'].max():.2f}")
    print(f"  Mean: {df['pKd'].mean():.2f} ± {df['pKd'].std():.2f}")
    print(f"  Median: {df['pKd'].median():.2f}")

    # 创建5折划分(包含验证集)
    print(f"\n=== Creating 5-fold splits with validation set ===")
    fold_splits = create_5fold_split_with_validation(results, random_seed=42, val_ratio=0.125)

    # 保存fold划分 (JSON格式)
    fold_output_path = os.path.join(output_dir, "5fold_splits.json")
    with open(fold_output_path, 'w') as f:
        json.dump(fold_splits, f, indent=2)
    print(f"5-fold splits saved to: {fold_output_path}")

    # 保存每个fold的ID列表到单独的文件
    for fold in fold_splits:
        fold_num = fold['fold']

        # 保存训练集ID
        train_file = os.path.join(output_dir, f"fold{fold_num}_train.txt")
        with open(train_file, 'w') as f:
            for pdb_id in fold['train_ids']:
                f.write(f"{pdb_id}\n")

        # 保存验证集ID
        val_file = os.path.join(output_dir, f"fold{fold_num}_val.txt")
        with open(val_file, 'w') as f:
            for pdb_id in fold['val_ids']:
                f.write(f"{pdb_id}\n")

        # 保存测试集ID
        test_file = os.path.join(output_dir, f"fold{fold_num}_test.txt")
        with open(test_file, 'w') as f:
            for pdb_id in fold['test_ids']:
                f.write(f"{pdb_id}\n")

        print(f"Fold {fold_num}: {fold['n_train']} train, {fold['n_val']} val, {fold['n_test']} test")

    print(f"\n=== All files saved to: {output_dir} ===")
    print(f"Files created:")
    print(f"  - master_table.csv (main dataset)")
    print(f"  - 5fold_splits.json (fold information)")
    print(f"  - fold1_train.txt, fold1_val.txt, fold1_test.txt")
    print(f"  - fold2_train.txt, fold2_val.txt, fold2_test.txt")
    print(f"  - ... (fold 3-5)")

    # 计算并显示数据集划分比例
    total_samples = len(results)
    avg_train = sum(f['n_train'] for f in fold_splits) / len(fold_splits)
    avg_val = sum(f['n_val'] for f in fold_splits) / len(fold_splits)
    avg_test = sum(f['n_test'] for f in fold_splits) / len(fold_splits)

    print(f"\n=== Data Split Ratios ===")
    print(f"Total samples: {total_samples}")
    print(f"Average per fold:")
    print(f"  Train: {avg_train:.1f} ({avg_train/total_samples*100:.1f}%)")
    print(f"  Val:   {avg_val:.1f} ({avg_val/total_samples*100:.1f}%)")
    print(f"  Test:  {avg_test:.1f} ({avg_test/total_samples*100:.1f}%)")

if __name__ == "__main__":
    main()
