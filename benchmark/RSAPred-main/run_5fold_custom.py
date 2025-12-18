
import pandas as pd
import sys
import os
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from scipy.stats import pearsonr

# Path to full dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_DATA = os.path.join(BASE_DIR, r"Data_preprocessing\my_data\Final_sample_dataset_v1.csv")
SPLIT_DIR = r"C:\Users\Administrator\Desktop\Class\CSCI5526\final\dataset\processed"

# Output for results
RESULTS_FILE = os.path.join(BASE_DIR, "5fold_results.csv")

def pearson_func(data1, data2):
    if len(data1) < 2: return 0.0
    pearson_corr, p_value = pearsonr(data1, data2)
    return pearson_corr

pearson_r_scorer = make_scorer(pearson_func)

def run():
    results = []

    print(f"Reading dataset from {FULL_DATA}")
    df_full = pd.read_csv(FULL_DATA, sep='\t')
    
    # Identify feature columns
    # Exclude metadata
    non_feat = ['Entry_ID', 'SMILES', 'Target_RNA_sequence', 'Molecule_name', 'Molecule_ID', 'Target_RNA_name', 'Target_RNA_ID', 'name', 'pKd']
    feat_cols = [c for c in df_full.columns if c not in non_feat]
    
    # Drop columns that are non-numeric or all NaN (just in case)
    # df_full[feat_cols] = df_full[feat_cols].apply(pd.to_numeric, errors='coerce')
    # df_full.dropna(axis=1, how='all', inplace=True)
    # Re-evaluate feat_cols
    feat_cols = [c for c in df_full.columns if c not in non_feat]
    print(f"Initial feature count: {len(feat_cols)}")

    for fold in range(1, 6):
        print(f"\nProcessing Fold {fold}...")
        train_file = os.path.join(SPLIT_DIR, f"fold{fold}_train.txt")
        test_file = os.path.join(SPLIT_DIR, f"fold{fold}_test.txt")
        
        try:
            with open(train_file, 'r') as f:
                train_ids = [line.strip() for line in f.readlines()]
            with open(test_file, 'r') as f:
                test_ids = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Fold files not found for fold {fold}. Skipping.")
            continue
            
        # Filter
        train_df = df_full[df_full['Target_RNA_ID'].isin(train_ids)].copy()
        test_df = df_full[df_full['Target_RNA_ID'].isin(test_ids)].copy()
        
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        if len(train_df) == 0 or len(test_df) == 0:
            print("Error: Empty train or test set. Check IDs.")
            continue

        X_train = train_df[feat_cols].values
        Y_train = train_df['pKd'].values
        X_test = test_df[feat_cols].values
        Y_test = test_df['pKd'].values
        
        # Handle NaNs/Infinity in features
        X_train = np.nan_to_num(X_train.astype(float))
        X_test = np.nan_to_num(X_test.astype(float))
        
        # Remove constant features in training set
        from sklearn.feature_selection import VarianceThreshold
        sel = VarianceThreshold(threshold=0)
        sel.fit(X_train)
        X_train_var = sel.transform(X_train)
        X_test_var = sel.transform(X_test)
        print(f"Features after removing constant variance: {X_train_var.shape[1]}")
        
        # Feature Selection (RFECV)
        print("Running Feature Selection (RFECV)...")
        mlr = linear_model.LinearRegression()
        # step=0.2 remove 20% features at each step for speed
        # min_features_to_select=10
        rfecv = RFECV(estimator=mlr, cv=5, scoring=pearson_r_scorer, step=0.2, min_features_to_select=10, n_jobs=1) 
        rfecv.fit(X_train_var, Y_train)
        
        print(f"Optimal features: {rfecv.n_features_}")
        
        # Train Final Model
        mlr_final = linear_model.LinearRegression()
        X_train_sel = rfecv.transform(X_train_var)
        X_test_sel = rfecv.transform(X_test_var)
        
        mlr_final.fit(X_train_sel, Y_train)
        
        # Predict
        Y_pred = mlr_final.predict(X_test_sel)
        
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        pearson_corr, _ = pearsonr(Y_test, Y_pred)
        
        print(f"Fold {fold} Results: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}, Pearson={pearson_corr:.4f}")
        
        results.append({
            'fold': fold,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'pearson': pearson_corr
        })

    # Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_FILE, index=False)
    print("\nAll Done. Results saved to", RESULTS_FILE)
    print(res_df)

if __name__ == "__main__":
    run()
