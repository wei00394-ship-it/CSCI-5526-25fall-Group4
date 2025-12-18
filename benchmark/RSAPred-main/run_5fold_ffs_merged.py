import pandas as pd
import numpy as np
import sys
import os
import itertools
from itertools import combinations
from sklearn import linear_model
from scipy.stats import pearsonr, spearmanr
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import pickle

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_DATA = os.path.join(BASE_DIR, r"Data_preprocessing\my_data\Final_sample_dataset_v1.csv")
SPLIT_DIR = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\processed"
RESULTS_FILE = os.path.join(BASE_DIR, "5fold_results_ffs_merged_metrics.csv")

# RNA Features List
RNA_FEATURES = [
    "A", "G", "C", "U", "AA", "AG", "AC", "AU", "GA", "GG", "GC", "GU", "CA", "CG", "CC", "CU", 
    "UA", "UG", "UC", "UU", "AAA", "AAG", "AAC", "AAU", "AGA", "AGC", "AGU", "ACA", "ACC", "ACU", 
    "AUG", "AUC", "AUU", "GAA", "GAG", "GAC", "GAU", "GGA", "GGG", "GGC", "GGU", "GCA", "GCG", 
    "GCC", "GCU", "GUA", "GUG", "GUC", "GUU", "CAA", "CAG", "CAC", "CAU", "CGA", "CGG", "CGC", 
    "CGU", "CCA", "CCG", "CCC", "CCU", "CUA", "CUG", "CUC", "UAC", "UAU", "UGA", "UGG", "UGC", 
    "UGU", "UCA", "UCG", "UCC", "UCU", "UUG", "UUC", "UUU", "AAAG", "AAAU", "AAGU", "AAUG", "AAUC", 
    "AGCA", "AGCG", "AGUG", "ACAC", "ACAU", "ACCC", "ACUA", "ACUG", "AUGG", "AUGU", "AUCU", "AUUC", 
    "AUUU", "GAAA", "GAAU", "GAGC", "GACA", "GACU", "GAUG", "GGAA", "GGAG", "GGGA", "GGGC", "GGGU", 
    "GGCA", "GGCC", "GGUU", "GCAA", "GCAC", "GCGA", "GCGU", "GCUG", "GUGG", "GUGC", "GUGU", "GUCA", 
    "GUCC", "GUUG", "CAAA", "CAAG", "CAAC", "CAGU", "CACC", "CACU", "CAUG", "CAUU", "CGAC", "CGCU", 
    "CCAA", "CCAG", "CUGA", "CUGC", "CUGU", "CUCA", "UAUU", "UGAA", "UGAG", "UGAC", "UGAU", "UGGA", 
    "UGGG", "UGGC", "UGCG", "UGCU", "UGUG", "UGUC", "UGUU", "UCAA", "UCAC", "UCAU", "UCCA", "UCUG", 
    "UCUC", "UUGA", "UUGG", "UUCU", "UUUG", "DNC_AA", "DNC_AG", "DNC_AC", "DNC_AU", "DNC_GA", 
    "DNC_GG", "DNC_GC", "DNC_GU", "DNC_CA", "DNC_CG", "DNC_CC", "DNC_CU", "DNC_UA", "DNC_UG", 
    "DNC_UC", "DNC_UU", "DNC_Feat_17", "DNC_Feat_18", "DNC_Feat_19", "DNC_Feat_20", "DNC_Feat_21", 
    "DNC_Feat_22", "DNC_Feat_23", "DNC_Feat_24", "A(((", "A((.", "A(..", "A(.( ", "A.(( ", "A..(", 
    "A...", "G(((", "G((.", "G(..", "G(.( ", "G.(( ", "G..(", "G...", "C(((", "C((.", "C(.( ", 
    "C.(( ", "C.(. ", "C..(", "C...", "U(((", "U((.", "U(..", "U(.( ", "U.(( ", "U..(", "U...", 
    "A,A", "A,C", "A,U", "A,A-U", "A,U-A", "A,C-G", "G,A", "G,G", "G,U", "G,A-U", "G,G-C", 
    "G,C-G", "C,A", "C,C", "C,U", "C,U-A", "C,G-C", "C,C-G", "U,A", "U,G", "U,C", "U,U", 
    "U,G-C", "U,G-U", "A-U,A", "A-U,C", "A-U,A-U", "A-U,U-A", "A-U,G-C", "A-U,C-G", "A-U,U-G", 
    "U-A,A", "U-A,G", "U-A,U", "U-A,U-A", "U-A,G-C"
]

def is_rna_feature(feat_name):
    if feat_name in RNA_FEATURES: return True
    if feat_name.startswith("DNC_") or feat_name.startswith("TNC_"): return True
    if any(c in "AGCU" for c in feat_name) and len(feat_name) < 10 and not "bond" in feat_name.lower(): return True 
    return False

def evaluate_combo(combo, X_data, y_data):
    try:
        X = X_data[list(combo)].values
        model = linear_model.LinearRegression()
        model.fit(X, y_data)
        y_pred = model.predict(X)
        pearson_corr, _ = pearsonr(y_data, y_pred)
        return (combo, pearson_corr)
    except Exception as e:
        return (combo, -1.0)

def get_low_corr_pairs(df, threshold=0.8):
    print("Computing correlation matrix...")
    corr_matrix = df.corr().abs()
    valid_pairs = set()
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr_matrix.iloc[i, j] < threshold:
                valid_pairs.add(frozenset([cols[i], cols[j]]))
    print(f"Found {len(valid_pairs)} valid low-correlation pairs.")
    return valid_pairs

def run_ffs_for_fold(fold, train_df, test_df, feat_cols):
    print(f"--- FFS for Fold {fold} ---")
    X_train_full = train_df[feat_cols]
    y_train = train_df['pKd'].values
    
    # 1. Initial Screening (Univariate) using ONLY training data
    print("Step 1: Univariate Screening (Train Only)")
    corrs = []
    for f in feat_cols:
        try:
            # Check for constant variance to avoid warnings
            if X_train_full[f].nunique() <= 1:
                continue
            c, _ = pearsonr(X_train_full[f], y_train)
            if np.isnan(c): continue
            corrs.append((f, abs(c)))
        except:
            pass
    corrs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top 100 RNA and top 100 Mol features
    top_rna = [f for f, c in corrs if is_rna_feature(f)][:100]
    top_mol = [f for f, c in corrs if not is_rna_feature(f)][:100]
    
    # Fill up if not enough
    if len(top_rna) < 10: top_rna = [f for f, c in corrs if is_rna_feature(f)][:50] 
    
    pool_features = top_rna + top_mol
    print(f"Selected {len(pool_features)} features for FFS pool ({len(top_rna)} RNA, {len(top_mol)} Mol).")
    
    # 2. Compute Allowed Pairs for this pool (Using Train Only)
    df_pool = X_train_full[pool_features]
    valid_pairs = get_low_corr_pairs(df_pool, threshold=0.8)
    
    best_models = [] 
    
    def sequential_eval(candidates):
        results = []
        for i, combo in enumerate(candidates):
            if i % 5000 == 0 and i > 0: print(f"Evaluated {i} candidates...", end='\r')
            
            rna_count = sum(1 for f in combo if is_rna_feature(f))
            if rna_count == 0 or rna_count == len(combo):
                continue 
            
            res = evaluate_combo(combo, df_pool, y_train)
            if res[1] > 0:
                results.append(res)
        if len(candidates) > 5000: print("")
        return results

    # Level 2 (Pairs)
    print("Step 2: Evaluating Pairs (n=2)")
    candidates_2 = [tuple(c) for c in combinations(pool_features, 2) if frozenset(c) in valid_pairs]
    print(f"Evaluating {len(candidates_2)} pairs...")
    results_2 = sequential_eval(candidates_2)
    results_2.sort(key=lambda x: x[1], reverse=True)
    
    if not results_2:
        print("No valid pairs found.")
        best_combo = [corrs[0][0]]
    else:
        print(f"Top Pair Corr (Train): {results_2[0][1]:.4f}")
        best_models.extend(results_2[:50]) 
        
        # Level 3 (Triplets)
        print("Step 3: Evaluating Triplets (n=3)")
        candidates_3 = []
        for pair_res in results_2[:50]: 
            pair = pair_res[0]
            for feat in pool_features:
                if feat in pair: continue
                new_triplet = pair + (feat,)
                if frozenset({feat, pair[0]}) in valid_pairs and frozenset({feat, pair[1]}) in valid_pairs:
                    candidates_3.append(tuple(sorted(new_triplet)))
        
        candidates_3 = list(set(candidates_3)) 
        print(f"Evaluating {len(candidates_3)} triplets...")
        results_3 = sequential_eval(candidates_3)
        results_3.sort(key=lambda x: x[1], reverse=True)
        
        if results_3:
            print(f"Top Triplet Corr (Train): {results_3[0][1]:.4f}")
            best_models.extend(results_3[:20])
            
            # Level 4
            print("Step 4: Evaluating Quadruplets (n=4)")
            candidates_4 = []
            for trip_res in results_3[:20]:
                trip = trip_res[0]
                for feat in pool_features:
                    if feat in trip: continue
                    if all(frozenset({feat, x}) in valid_pairs for x in trip):
                        candidates_4.append(tuple(sorted(trip + (feat,))))
            
            candidates_4 = list(set(candidates_4))
            print(f"Evaluating {len(candidates_4)} quadruplets...")
            results_4 = sequential_eval(candidates_4)
            results_4.sort(key=lambda x: x[1], reverse=True)
            
            if results_4:
                print(f"Top Quad Corr (Train): {results_4[0][1]:.4f}")
                best_models.extend(results_4[:10])

        best_models.sort(key=lambda x: x[1], reverse=True)
        best_combo = best_models[0][0]

    print(f"Best Combination selected: {best_combo}")
    
    # Final Training on Combined Train (Train+Val)
    X_train = train_df[list(best_combo)].values
    y_train = train_df['pKd'].values
    
    # Evaluation on Completely Unseen Test Set
    X_test = test_df[list(best_combo)].values
    y_test = test_df['pKd'].values
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    p_corr, _ = pearsonr(y_test, y_pred)
    s_corr, _ = spearmanr(y_test, y_pred)
    
    print(f"Fold {fold} Test Results: RMSE={rmse:.4f}, MAE={mae:.4f}, Pearson={p_corr:.4f}, Spearman={s_corr:.4f}")
    
    return {
        'fold': fold,
        'features': str(best_combo),
        'rmse': rmse,
        'mae': mae,
        'pearson': p_corr,
        'spearman': s_corr
    }

def run():
    print(f"Reading dataset from {FULL_DATA}")
    df_full = pd.read_csv(FULL_DATA, sep='\t')
    
    non_feat = ['Entry_ID', 'SMILES', 'Target_RNA_sequence', 'Molecule_name', 'Molecule_ID', 'Target_RNA_name', 'Target_RNA_ID', 'name', 'pKd']
    feat_cols = [c for c in df_full.columns if c not in non_feat]
    
    for c in feat_cols:
        df_full[c] = pd.to_numeric(df_full[c], errors='coerce')
    df_full.fillna(0, inplace=True)
    
    results = []
    
    for fold in range(1, 6):
        train_file = os.path.join(SPLIT_DIR, f"fold{fold}_train.txt")
        val_file = os.path.join(SPLIT_DIR, f"fold{fold}_val.txt")
        test_file = os.path.join(SPLIT_DIR, f"fold{fold}_test.txt")
        
        # MERGING TRAIN + VAL
        with open(train_file, 'r') as f: t_ids = [l.strip() for l in f.readlines()]
        with open(val_file, 'r') as f: v_ids = [l.strip() for l in f.readlines()]
        with open(test_file, 'r') as f: test_ids = [l.strip() for l in f.readlines()]
        
        train_ids = t_ids + v_ids # Merged Training Set
        
        # STRICT SPLIT
        train_df = df_full[df_full['Target_RNA_ID'].isin(train_ids)].copy()
        test_df = df_full[df_full['Target_RNA_ID'].isin(test_ids)].copy()
        
        print(f"\nFold {fold}: Combined Train size: {len(train_df)} (Train {len(t_ids)} + Val {len(v_ids)}), Test size: {len(test_df)}")
        
        if len(train_df) == 0 or len(test_df) == 0:
            print("Error: Empty dataset.")
            continue

        res = run_ffs_for_fold(fold, train_df, test_df, feat_cols)
        results.append(res)

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_FILE, index=False)
    print("\nAll Done. Results saved to", RESULTS_FILE)
    print(res_df)

if __name__ == "__main__":
    run()