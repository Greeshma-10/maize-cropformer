import pandas as pd
from sklearn.model_selection import train_test_split
import traceback

# --- IMPORTANT: CHOOSE YOUR TRAIT ---
# Make sure this is the same trait you used for feature selection.
TARGET_TRAIT = 'DTT'
# ------------------------------------

# --- Configuration ---
feature_selected_geno_file = 'chr10_top10k_snps.csv'
phenotype_file = 'Ncii_2015_5locs_hybrids.txt'
# ---------------------

print("🚀 Starting Phase 2: Splitting data into training and testing sets...")

try:
    # 1. Load the data
    print(f"Loading feature-selected genotype data from '{feature_selected_geno_file}'...")
    X = pd.read_csv(feature_selected_geno_file, index_col=0)

    print(f"Loading and averaging phenotype data from '{phenotype_file}'...")
    pheno_df = pd.read_csv(phenotype_file, sep='\\s+')
    
    # --- THIS IS THE NEW, CORRECTED PART ---
    # Group by the sample ID and calculate the mean for the target trait
    y_series = pheno_df.groupby('Lineid')[TARGET_TRAIT].mean()
    # ---------------------------------------

    # 2. Align the datasets
    print("Aligning final genotype and phenotype data...")
    common_samples = X.index.intersection(y_series.index)
    X = X.loc[common_samples]
    y = y_series.loc[common_samples]
    
    # Drop any remaining missing values (e.g., if mean results in NaN)
    valid_indices = y.dropna().index
    y = y.loc[valid_indices]
    X = X.loc[valid_indices]

    print(f"Final dataset contains {len(X)} samples.")

    # 3. Split the data (80% train, 20% test)
    print("Splitting data (80% training, 20% testing)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # 4. Save the four new files
    print("Saving output files...")
    X_train.to_csv('X_train.csv')
    y_train.to_csv('y_train.csv', header=True)
    X_test.to_csv('X_test.csv')
    y_test.to_csv('y_test.csv', header=True)
    
    print("✅ Success! Data splitting is complete.")

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    traceback.print_exc()