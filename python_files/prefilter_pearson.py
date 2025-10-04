import pandas as pd
from tqdm import tqdm
import traceback

# --- Configuration ---
TARGET_TRAIT = 'DTT'
encoded_geno_file = 'chr10_encoded.csv'
phenotype_file = 'Ncii_2015_5locs_hybrids.txt'
output_snp_list_file = 'top30k_pearson_snps.txt'
TOP_N_PEARSON = 30000
BATCH_SIZE = 10000
# ---------------------

print("🚀 Starting Step 1: Fast Pre-filtering with Pearson Correlation...")

try:
    # 1. Load phenotype data
    print(f"Loading phenotype data from '{phenotype_file}'...")
    pheno_df = pd.read_csv(phenotype_file, sep='\\s+', index_col='Lineid')
    
    # 2. Get the full list of SNP columns from the large CSV header
    print(f"Reading marker list from '{encoded_geno_file}'...")
    all_snps = pd.read_csv(encoded_geno_file, nrows=0, engine='python').columns[1:]

    # 3. Align phenotype to all potential samples in the genotype file
    all_samples = pd.read_csv(encoded_geno_file, index_col=0, usecols=[0]).index
    common_samples = all_samples.intersection(pheno_df.index)
    pheno_series = pheno_df.loc[common_samples, TARGET_TRAIT].dropna()
    
    print(f"Aligned data. Found {len(pheno_series)} samples with valid phenotypes.")

    # 4. Process SNPs in batches to calculate correlation
    all_correlations = pd.Series(dtype=float)
    
    for i in tqdm(range(0, len(all_snps), BATCH_SIZE), desc="Calculating Pearson Correlation in batches"):
        batch_snps = all_snps[i:i+BATCH_SIZE].tolist()
        
        # Read only the necessary columns for this batch + the aligned samples
        batch_geno_df = pd.read_csv(encoded_geno_file, index_col=0, usecols=['IID'] + batch_snps)
        batch_geno_df = batch_geno_df.loc[pheno_series.index]
        
        # Calculate correlation for this batch (fast and vectorized)
        batch_corrs = batch_geno_df.corrwith(pheno_series)
        all_correlations = pd.concat([all_correlations, batch_corrs])

    # 5. Select top N SNPs based on the absolute correlation value
    print(f"Selecting top {TOP_N_PEARSON} SNPs based on absolute correlation...")
    top_snps = all_correlations.abs().nlargest(TOP_N_PEARSON).index.tolist()

    # 6. Save the list of top SNP names to a file
    print(f"💾 Saving list of top SNPs to '{output_snp_list_file}'...")
    with open(output_snp_list_file, 'w') as f:
        for snp_name in top_snps:
            f.write(f"{snp_name}\n")
            
    print("✅ Step 1 complete!")

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    traceback.print_exc()