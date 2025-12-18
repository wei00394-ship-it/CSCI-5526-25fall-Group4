
import pandas as pd
import numpy as np
import sys
import os
import itertools
from itertools import combinations
from sklearn import linear_model
from scipy.stats import pearsonr
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import pickle

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_DATA = os.path.join(BASE_DIR, r"Data_preprocessing\my_data\Final_sample_dataset_v1.csv")
SPLIT_DIR = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\processed"
RESULTS_FILE = os.path.join(BASE_DIR, "5fold_results_ffs.csv")

# RNA Features List (from process_regression_output_v1.py)
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
    "DNC_Feat_22", "DNC_Feat_23", "DNC_Feat_24", "A((( ", "A((. ", "A(.. ", "A(.( ", "A.(( ", "A..( ", "A... ", 
    "G((( ", "G((. ", "G(.. ", "G(.( ", "G.(( ", "G..( ", "G... ", 
    "C((( ", "C((. ", "C(.( ", "C.(( ", "C.(. ", "C..( ", "C... ", 
    "U((( ", "U((. ", "U(.. ", "U(.( ", "U.(( ", "U..( ", "U... ", 
    "A,A", "A,C", "A,U", "A,A-U", "A,U-A", "A,C-G", "G,A", "G,G", "G,U", "G,A-U", "G,G-C", 
    "G,C-G", "C,A", "C,C", "C,U", "C,U-A", "C,G-C", "C,C-G", "U,A", "U,G", "U,C", "U,U", 
    "U,G-C", "U,G-U", "A-U,A", "A-U,C", "A-U,A-U", "A-U,U-A", "A-U,G-C", "A-U,C-G", "A-U,U-G", 
    "U-A,A", "U-A,G", "U-A,U", "U-A,U-A", "U-A,G-C"
] # (truncated manually, but logic will use set membership)
# Note: The full list is extremely long. Better logic: check if feature starts with certain prefixes or is NOT in Mol features.
# Assuming Mol features are from Mordred, they have specific names.
# The script uses this list to ENFORCE RNA+Mol combination.
# I will try to deduce RNA features dynamically or allow flexible check.

def is_rna_feature(feat_name):
    # Heuristic based on standard RNA feature names in RSAPred
    if feat_name in RNA_FEATURES: return True
    # Also checks common prefixes if list incomplete
    if feat_name.startswith("DNC_") or feat_name.startswith("TNC_"): return True
    if any(c in "AGCU" for c in feat_name) and len(feat_name) < 10 and not "bond" in feat_name.lower(): return True # e.g. "AA", "G((."
    return False

def evaluate_combo(combo, X_data, y_data):
    # X_data is a DataFrame or dict-like access
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
    
    # 1. Initial Screening (Univariate)
    print("Step 1: Univariate Screening")
    corrs = []
    for f in feat_cols:
        try:
            c, _ = pearsonr(X_train_full[f], y_train)
            corrs.append((f, abs(c)))
        except:
            pass
    corrs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top 100 RNA and top 100 Mol features
    top_rna = [f for f, c in corrs if is_rna_feature(f)][:100]
    top_mol = [f for f, c in corrs if not is_rna_feature(f)][:100]
    
    # Fill up if not enough
    if len(top_rna) < 10: top_rna = [f for f, c in corrs if is_rna_feature(f)][:50] # Fallback
    
    pool_features = top_rna + top_mol
    print(f"Selected {len(pool_features)} features for FFS pool ({len(top_rna)} RNA, {len(top_mol)} Mol).")
    
    # 2. Compute Allowed Pairs for this pool
    df_pool = X_train_full[pool_features]
    valid_pairs = get_low_corr_pairs(df_pool, threshold=0.8)
    
    best_models = [] # List of (combo, corr)
    
    # Helper for parallel eval
    def parallel_eval(candidates):
        results = []
        # Use process pool
        with ProcessPoolExecutor(max_workers=4) as executor:
            # We need to pass dataframe, but passing large DF to workers is slow.
            # Optimization: Pass numpy array and indices? 
            # For simplicity with this dataset size (~100 rows), passing slices is okay?
            # Actually, let's just run sequential if list is small, or chunks.
            # 20k candidates is fine for sequential if simple linear regression.
            # Sklearn LR is fast.
            for combo in candidates:
                # Enforce RNA+Mol constraint
                rna_count = sum(1 for f in combo if is_rna_feature(f))
                if rna_count == 0 or rna_count == len(combo):
                    continue # Skip if not mixed
                
                res = evaluate_combo(combo, df_pool, y_train)
                if res[1] > 0:
                    results.append(res)
        return results

    def sequential_eval(candidates):
        results = []
        for i, combo in enumerate(candidates):
            if i % 5000 == 0: print(f"Evaluated {i} candidates...", end='\r')
            
            # Enforce RNA+Mol constraint
            rna_count = sum(1 for f in combo if is_rna_feature(f))
            if rna_count == 0 or rna_count == len(combo):
                continue 
            
            res = evaluate_combo(combo, df_pool, y_train)
            if res[1] > 0:
                results.append(res)
        print("")
        return results

    # Level 2 (Pairs)
    print("Step 2: Evaluating Pairs (n=2)")
    candidates_2 = [tuple(c) for c in combinations(pool_features, 2) if frozenset(c) in valid_pairs]
    print(f"Evaluating {len(candidates_2)} pairs...")
    results_2 = sequential_eval(candidates_2)
    results_2.sort(key=lambda x: x[1], reverse=True)
    
    if not results_2:
        print("No valid pairs found. Using univariate best.")
        return corrs[0][0] # Fallback
        
    print(f"Top Pair Corr: {results_2[0][1]:.4f}")
    best_models.extend(results_2[:50]) # Keep top 50 pairs
    
    # Level 3 (Triplets)
    print("Step 3: Evaluating Triplets (n=3)")
    # Form triplets from features present in top 50 pairs + high performing singles
    # Heuristic: Take top 50 pairs, try adding 1 feature from top_features pool
    top_feats_from_pairs = set()
    for m in results_2[:50]:
        top_feats_from_pairs.update(m[0])
    
    candidates_3 = []
    for pair_res in results_2[:50]: # Extend top 50 pairs
        pair = pair_res[0]
        for feat in pool_features:
            if feat in pair: continue
            
            # Check valid pairs
            new_triplet = pair + (feat,)
            # Check if (feat, pair[0]) and (feat, pair[1]) are valid
            if frozenset({feat, pair[0]}) in valid_pairs and frozenset({feat, pair[1]}) in valid_pairs:
                # Avoid duplicates (sort)
                candidates_3.append(tuple(sorted(new_triplet)))
    
    candidates_3 = list(set(candidates_3)) # Unique
    print(f"Evaluating {len(candidates_3)} triplets...")
    results_3 = sequential_eval(candidates_3)
    results_3.sort(key=lambda x: x[1], reverse=True)
    
    if results_3:
        print(f"Top Triplet Corr: {results_3[0][1]:.4f}")
        best_models.extend(results_3[:20])
    
    # Level 4? Maybe stop at 3 or 4. Let's try 4.
    print("Step 4: Evaluating Quadruplets (n=4)")
    candidates_4 = []
    if results_3:
        for trip_res in results_3[:20]:
            trip = trip_res[0]
            for feat in pool_features:
                if feat in trip: continue
                # Check cliques
                if all(frozenset({feat, x}) in valid_pairs for x in trip):
                    candidates_4.append(tuple(sorted(trip + (feat,))))
        
        candidates_4 = list(set(candidates_4))
        print(f"Evaluating {len(candidates_4)} quadruplets...")
        results_4 = sequential_eval(candidates_4)
        results_4.sort(key=lambda x: x[1], reverse=True)
        
        if results_4:
            print(f"Top Quad Corr: {results_4[0][1]:.4f}")
            best_models.extend(results_4[:10])

    # Final Selection
    best_models.sort(key=lambda x: x[1], reverse=True)
    best_combo = best_models[0][0]
    print(f"Best Combination found: {best_combo} (Corr: {best_models[0][1]:.4f})")
    
    # Train and Test
    X_train = train_df[list(best_combo)].values
    y_train = train_df['pKd'].values
    X_test = test_df[list(best_combo)].values
    y_test = test_df['pKd'].values
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    p_corr, _ = pearsonr(y_test, y_pred)
    
    return {
        'fold': fold,
        'features': str(best_combo),
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'pearson': p_corr
    }

def run():
    print(f"Reading dataset from {FULL_DATA}")
    df_full = pd.read_csv(FULL_DATA, sep='\t')
    
    # Drop non-numeric
    non_feat = ['Entry_ID', 'SMILES', 'Target_RNA_sequence', 'Molecule_name', 'Molecule_ID', 'Target_RNA_name', 'Target_RNA_ID', 'name', 'pKd']
    feat_cols = [c for c in df_full.columns if c not in non_feat]
    
    # Ensure numeric
    for c in feat_cols:
        df_full[c] = pd.to_numeric(df_full[c], errors='coerce')
    
    # Fill NA
    df_full.fillna(0, inplace=True)
    
    results = []
    
    for fold in range(1, 6):
        train_file = os.path.join(SPLIT_DIR, f"fold{fold}_train.txt")
        test_file = os.path.join(SPLIT_DIR, f"fold{fold}_test.txt")
        
        with open(train_file, 'r') as f:
            train_ids = [l.strip() for l in f.readlines()]
        with open(test_file, 'r') as f:
            test_ids = [l.strip() for l in f.readlines()]
            
        train_df = df_full[df_full['Target_RNA_ID'].isin(train_ids)].copy()
        test_df = df_full[df_full['Target_RNA_ID'].isin(test_ids)].copy()
        
        res = run_ffs_for_fold(fold, train_df, test_df, feat_cols)
        results.append(res)
        print(f"Fold {fold} Final: {res}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)
    print(res_df)

if __name__ == "__main__":
    run()
