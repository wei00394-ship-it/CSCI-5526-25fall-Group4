"""
RSAPred简化版训练脚本 - 使用传统机器学习方法
基于RNA序列和SMILES的特征进行亲和力预测
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib
import json

# 简单的特征提取
def extract_rna_features(sequence):
    """从RNA序列提取简单特征"""
    features = {}

    # 长度
    features['length'] = len(sequence)

    # 碱基组成
    for base in ['A', 'U', 'G', 'C', 'T']:
        features[f'{base}_count'] = sequence.count(base)
        features[f'{base}_freq'] = sequence.count(base) / len(sequence) if len(sequence) > 0 else 0

    # GC含量
    gc_count = sequence.count('G') + sequence.count('C')
    features['GC_content'] = gc_count / len(sequence) if len(sequence) > 0 else 0

    # 二核苷酸
    dinucs = ['AA', 'AU', 'AG', 'AC', 'UA', 'UU', 'UG', 'UC',
              'GA', 'GU', 'GG', 'GC', 'CA', 'CU', 'CG', 'CC']
    for dinuc in dinucs:
        count = sum(1 for i in range(len(sequence)-1) if sequence[i:i+2] == dinuc)
        features[f'dinuc_{dinuc}'] = count

    return features

def extract_smiles_features(smiles):
    """从SMILES提取简单特征"""
    features = {}

    # 长度
    features['smiles_length'] = len(smiles)

    # 字符统计
    features['C_count'] = smiles.count('C')
    features['N_count'] = smiles.count('N')
    features['O_count'] = smiles.count('O')
    features['ring_count'] = smiles.count('1') + smiles.count('2') + smiles.count('3')
    features['double_bond'] = smiles.count('=')
    features['triple_bond'] = smiles.count('#')
    features['branch'] = smiles.count('(')

    # 芳香性
    features['aromatic'] = smiles.count('c') + smiles.count('n') + smiles.count('o')

    return features

def prepare_features(df):
    """准备特征矩阵"""
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        # RNA特征
        rna_feats = extract_rna_features(row['rna_sequence'])

        # SMILES特征
        smiles_feats = extract_smiles_features(row['ligand_smiles'])

        # 合并特征
        combined = {**rna_feats, **smiles_feats}
        X_list.append(combined)
        y_list.append(row['pKd'])

    # 转换为DataFrame
    X = pd.DataFrame(X_list)
    y = np.array(y_list)

    return X, y

def evaluate_predictions(y_true, y_pred):
    """计算评估指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson': pearson
    }

def train_and_evaluate_fold(fold_num, data_dir, output_dir):
    """训练和评估单个fold"""
    print(f"\n{'='*60}")
    print(f"Processing Fold {fold_num}")
    print(f"{'='*60}")

    # 加载数据
    train_df = pd.read_csv(os.path.join(data_dir, f"fold{fold_num}", "train_data.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, f"fold{fold_num}", "val_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, f"fold{fold_num}", "test_data.csv"))

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 准备特征
    print("Extracting features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    # 确保特征一致
    common_features = X_train.columns
    X_val = X_val[common_features]
    X_test = X_test[common_features]

    # 填充缺失值
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    print(f"Feature dim: {X_train.shape[1]}")

    # 训练模型
    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 预测
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # 评估
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存模型
    fold_output_dir = os.path.join(output_dir, f"fold{fold_num}")
    os.makedirs(fold_output_dir, exist_ok=True)

    model_path = os.path.join(fold_output_dir, "model.pkl")
    joblib.dump(model, model_path)

    # 保存预测结果
    predictions = {
        'val': {
            'pdb_ids': val_df['pdb_id'].tolist(),
            'y_true': y_val.tolist(),
            'y_pred': y_val_pred.tolist()
        },
        'test': {
            'pdb_ids': test_df['pdb_id'].tolist(),
            'y_true': y_test.tolist(),
            'y_pred': y_test_pred.tolist()
        }
    }

    with open(os.path.join(fold_output_dir, "predictions.json"), 'w') as f:
        json.dump(predictions, f, indent=2)

    # 保存指标
    metrics = {
        'val': val_metrics,
        'test': test_metrics
    }

    with open(os.path.join(fold_output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    return test_metrics

def main():
    data_dir = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\benchmark\RSAPred-main\data"
    output_dir = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\benchmark\RSAPred-main\results"

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # 训练所有5个fold
    for fold_num in range(1, 6):
        try:
            test_metrics = train_and_evaluate_fold(fold_num, data_dir, output_dir)
            all_results.append(test_metrics)
        except Exception as e:
            print(f"\nError in Fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    if all_results:
        print("\n" + "="*60)
        print("OVERALL RESULTS (Average across folds)")
        print("="*60)

        avg_metrics = {}
        for metric in ['RMSE', 'MAE', 'R2', 'Pearson']:
            values = [r[metric] for r in all_results]
            avg = np.mean(values)
            std = np.std(values)
            avg_metrics[metric] = {'mean': avg, 'std': std}
            print(f"{metric}: {avg:.4f} ± {std:.4f}")

        # 保存汇总结果
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump({
                'fold_results': all_results,
                'average': avg_metrics
            }, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
