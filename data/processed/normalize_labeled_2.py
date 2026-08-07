"""
===============================================================================
FINANCIAL DATASET NORMALIZATION PIPELINE
===============================================================================
Description:
    This script processes time-series financial market data from 'labeled_2.csv'
    and applies feature-specific normalization methods to optimize the dataset
    for machine learning model training.

Normalization Strategy:
    1. Trading Volumes (Log1p + StandardScaler):
       Reduces heavy right-tail skewness using a log(1+x) transform before 
       standardizing to mean 0 and variance 1.
    2. Volatility Indices (RobustScaler):
       Scales using median and interquartile range (IQR) to prevent market 
       panic spikes/outliers from distorting scale parameters.
    3. Prices, Deltas & Percent Changes (StandardScaler):
       Standardizes raw prices, deltas, and percentage returns around 0 with 
       a standard deviation of 1.
    4. Identifiers & Targets:
       Leaves 'date' and 'label' untouched.

Outputs:
    - 'labeled_2_normalized.csv' : Processed dataset ready for training.
    - 'scalers.joblib'           : Serialized scaler parameters for future 
                                   inference/prediction transforms.
===============================================================================
"""
import numpy as np
import pandas as pd
#import joblib
#from sklearn.preprocessing import RobustScaler, StandardScaler


def prepare_financial_dataset(
    input_filepath='data/processed/labeled_2.csv',
    output_filepath='data/processed/labeled_2_normalized.csv',
):
    """Loads financial dataset, applies non-leaking mathematical transformations

    (Log1p on volume features), and exports the dataset.

    Note: Parametric scaling (StandardScaler, RobustScaler) is deliberately
    omitted here to prevent train-test data leakage.
    """
    # 1. Load Dataset
    print(f"Loading raw features from '{input_filepath}'...")
    df = pd.read_csv(input_filepath)
    df_prepared = df.copy()

    # 2. Apply non-leaking mathematical transforms (Log1p on Trading Volume)
    raw_volume_cols = [
        'amzn_volume',
        'amzn_volume_lag1',
        'ixic_volume',
        'ixic_volume_lag1',
    ]

    print("Applying Log1p transformation to Trading Volume features...")
    for col in raw_volume_cols:
        if col in df_prepared.columns:
            df_prepared[col] = np.log1p(df_prepared[col].values)

    # 3. Export Prepared Dataset
    # Scalers (StandardScaler / RobustScaler) will be fitted exclusively inside train_2.py on X_train
    df_prepared.to_csv(output_filepath, index=False)

    print("\n--- Feature Preparation Complete ---")
    print(f"✅ Clean dataset saved to : '{output_filepath}'")
    print("   Standard/Robust scaling deferred to train_2.py post train/test split.\n")

    return df_prepared


if __name__ == '__main__':
    # Execute feature preparation pipeline
    prepared_df = prepare_financial_dataset()

    # Display sample output
    print("Sample Prepared Output (First 3 Rows):")
    print(prepared_df[['date', 'amzn', 'amzn_volume', 'vixcls', 'label']].head(3))


'''
def normalize_financial_dataset(
    input_filepath='labeled_2.csv',
    output_filepath='labeled_2_normalized.csv',
    scaler_export_path='scalers.joblib',
):
    """Loads financial dataset, applies column-specific normalization,

    saves the processed dataset and fitted scalers.
    """
    # 1. Load Dataset
    print(f"Loading data from '{input_filepath}'...")
    df = pd.read_csv(input_filepath)
    df_scaled = df.copy()

    # 2. Categorize Columns
    raw_volume_cols = [
        'amzn_volume',
        'amzn_volume_lag1',
        'ixic_volume',
        'ixic_volume_lag1',
    ]
    volatility_cols = ['vixcls', 'vixcls_lag1', 'vxazn', 'vxazn_lag1']
    non_scalable_cols = ['date', 'label']

    # Identify all remaining numeric columns (prices, deltas, change_pcts, returns)
    standard_cols = [
        col
        for col in df.columns
        if col not in raw_volume_cols + volatility_cols + non_scalable_cols
    ]

    # Dictionary to store scaler instances for future inference / inverse transforms
    fitted_scalers = {}

    # 3. Scale Raw Trading Volumes (Log1p + StandardScaler)
    print("Normalizing Trading Volume features (Log1p + StandardScaler)...")
    fitted_scalers['volume_scaler'] = StandardScaler()
    log_transformed_volumes = np.log1p(df[raw_volume_cols].values)
    df_scaled[raw_volume_cols] = fitted_scalers[
        'volume_scaler'
    ].fit_transform(log_transformed_volumes)

    # 4. Scale Volatility Indices (RobustScaler)
    print("Normalizing Volatility Indices (RobustScaler)...")
    fitted_scalers['volatility_scaler'] = RobustScaler()
    df_scaled[volatility_cols] = fitted_scalers[
        'volatility_scaler'
    ].fit_transform(df[volatility_cols])

    # 5. Scale Prices, Percent Changes, Deltas & Returns (StandardScaler)
    print("Normalizing Prices, Deltas & Percentage Changes (StandardScaler)...")
    fitted_scalers['standard_scaler'] = StandardScaler()
    df_scaled[standard_cols] = fitted_scalers['standard_scaler'].fit_transform(
        df[standard_cols]
    )

    # 6. Save Normalized Dataset & Scaler Pipeline
    df_scaled.to_csv(output_filepath, index=False)
    joblib.dump(fitted_scalers, scaler_export_path)

    print("\n--- Normalization Complete ---")
    print(f"Normalized CSV saved to : '{output_filepath}'")
    print(f"Fitted scalers saved to : '{scaler_export_path}'\n")

    return df_scaled


if __name__ == '__main__':
    # Execute normalization pipeline
    normalized_df = normalize_financial_dataset()

    # Display sample output
    print("Sample Normalized Output (First 3 Rows):")
    print(normalized_df[['date', 'amzn', 'amzn_volume', 'vixcls', 'label']].head(3))
'''



