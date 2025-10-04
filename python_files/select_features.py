import pandas as pd
from minepy import MINE
from tqdm import tqdm
import traceback
import os
from joblib import Parallel, delayed

# --- IMPORTANT: CHOOSE YOUR TRAIT ---
TARGET_TRAIT = 'DTT'
# ------------------------------------

# --- Configuration ---
encoded_geno_file = 'chr10_encoded.csv'
phenotype_file = 'Ncii_2015_5locs_hybrids.txt'
output_file_top_snps = 'chr10_top10k_snps.csv'
checkpoint_file = 'mic_scores_checkpoint.csv'
N_JOBS = -1  # Use all available CPU cores
BATCH_SIZE = 500  # Number of SNPs to process and save per checkpoint

# --- THIS IS THE NEW INPUT FILE ---
prefiltered_snp_list_file = 'top30k_pearson_snps.txt'
# ----------------------------------

print("🚀 Starting Step 2: Final MIC Selection on Pre-filtered SNPs...")

try:
    # 1. Load phenotype data and the PRE-FILTERED list of SNPs
    print(f"Loading phenotype data from '{phenotype_file}'...")
    pheno_df = pd.read_csv(phenotype_file, sep='\\s+', index_col='Lineid')
    
    # --- THIS IS THE KEY CHANGE ---
    print(f"Reading pre-filtered marker list from '{prefiltered_snp_list_file}'...")
    all_snps = pd.read_csv(prefiltered_snp_list_file, header=None)[0].tolist()
    # -----------------------------

    # 2. Load or initialize MIC scores from checkpoint
    mic_scores = {}
    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint file: '{checkpoint_file}'")
        checkpoint_df = pd.read_csv(checkpoint_file)
        mic_scores = pd.Series(checkpoint_df['mic'].values, index=checkpoint_df['snp']).to_dict()
        print(f"Loaded {len(mic_scores)} completed scores.")
    else:
        with open(checkpoint_file, 'w') as f:
            f.write("snp,mic\n")

    # 3. Determine remaining SNPs to process from the 30k list
    processed_snps = set(mic_scores.keys())
    remaining_snps = [snp for snp in all_snps if snp not in processed_snps]

    # (The rest of the script is identical to the last parallel version)
    def compute_mic(snp_data, pheno_data):
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(snp_data, pheno_data)
        return mine.mic()

    if not remaining_snps:
        print("🎉 All MIC scores already calculated. Proceeding to selection.")
    else:
        print(f"⚡ Calculating MIC scores for {len(remaining_snps)} remaining markers...")
        
        with open(checkpoint_file, 'a', buffering=1) as f_checkpoint:
            for i in tqdm(range(0, len(remaining_snps), BATCH_SIZE), desc="Processing batches"):
                batch_snps = remaining_snps[i:i+BATCH_SIZE]
                
                batch_geno_df = pd.read_csv(encoded_geno_file, index_col=0, usecols=['IID'] + batch_snps)
                
                common_samples = batch_geno_df.index.intersection(pheno_df.index)
                batch_geno_df = batch_geno_df.loc[common_samples]
                pheno_series_aligned = pheno_df.loc[common_samples, TARGET_TRAIT].dropna()
                batch_geno_df = batch_geno_df.loc[pheno_series_aligned.index]

                results = Parallel(n_jobs=N_JOBS)(
                    delayed(compute_mic)(batch_geno_df[snp].values, pheno_series_aligned.values)
                    for snp in batch_snps
                )

                for snp, score in zip(batch_snps, results):
                    mic_scores[snp] = score
                    f_checkpoint.write(f"{snp},{score}\n")

    # 5. Select the top 10,000 SNPs
    print("🔎 Selecting the top 10,000 markers...")
    top_snps = sorted(mic_scores, key=mic_scores.get, reverse=True)[:10000]
    
    print("Loading final data for top 10k SNPs...")
    final_df = pd.read_csv(encoded_geno_file, index_col=0, usecols=['IID'] + top_snps)
    
    # 6. Save the final result
    print(f"💾 Saving top 10,000 markers to '{output_file_top_snps}'...")
    final_df.to_csv(output_file_top_snps)

    print("✅ Success! Your final feature-selected dataset is ready.")

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    traceback.print_exc()