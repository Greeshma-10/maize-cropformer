import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback

# --- Configuration ---
raw_genotype_file = 'chr10_text_genotypes.raw'
bim_file = 'chr10_final_qc.bim'
output_csv = 'chr10_encoded.csv'
# This is the number of rows (samples) to process at a time from the large .raw file
chunksize_samples = 1000 
# -------------------

print("🟢 Starting final optimized and memory-safe 0-9 encoding...")

try:
    # 1. Load BIM file for marker information
    print("Reading BIM file...")
    bim = pd.read_csv(bim_file, sep='\t', header=None, names=['chrom', 'snp', 'cm', 'pos', 'a1', 'a2'])
    snp_order = bim['snp'].values
    num_snps = len(snp_order)

    # 2. Define the 0-9 encoding map
    encoding_map = {
        'AA': 0, 'AT': 1, 'TA': 1, 'AC': 2, 'CA': 2, 'AG': 3, 'GA': 3,
        'TT': 4, 'TC': 5, 'CT': 5, 'TG': 6, 'GT': 6, 'CC': 7, 'CG': 8,
        'GC': 8, 'GG': 9
    }

    # 3. Pre-compute the 0,1,2 -> 0-9 conversion matrix for all SNPs (your excellent idea)
    def compute_encoding_matrix(a1, a2):
        # PLINK .raw format: 2=hom_ref, 1=het, 0=hom_alt
        return np.array([
            encoding_map.get(a1 + a1, -1),      # Code for 0 (hom_alt)
            encoding_map.get(a1 + a2, -1),      # Code for 1 (het)
            encoding_map.get(a2 + a2, -1)       # Code for 2 (hom_ref)
        ])

    encoding_matrix = np.array([compute_encoding_matrix(a1, a2) for a1, a2 in zip(bim['a1'], bim['a2'])])
    print("✅ Encoding matrix ready.")

    # 4. Prepare the final CSV file with just the header
    print(f"Preparing output file '{output_csv}'...")
    # Get the sample order from the .fam file to be safe
    fam = pd.read_csv('chr10_final_qc.fam', sep=' ', header=None)
    sample_order = fam[1].values
    
    header = 'IID,' + ','.join(snp_order)
    with open(output_csv, 'w') as f:
        f.write(header + '\n')

    # 5. Process the large .raw file in chunks (MEMORY-SAFE)
    print(f"Processing '{raw_genotype_file}' in chunks of {chunksize_samples} samples...")
    
    # This creates an iterator that reads the file from disk in chunks, avoiding MemoryError
    chunk_iterator = pd.read_csv(raw_genotype_file, sep=' ', index_col='IID', chunksize=chunksize_samples)

    for chunk_df in tqdm(chunk_iterator, desc="Processing sample chunks"):
        # Isolate just the SNP columns
        snp_cols = chunk_df.columns[5:]
        X = chunk_df[snp_cols].values.astype(int)
        
        # Use your fast vectorized encoding method on the chunk
        # Note the transpose for correct indexing: encoding_matrix is (n_snps, 3), X is (n_samples, n_snps)
        encoded_chunk = encoding_matrix.T[X, np.arange(num_snps)]
        
        # Create a DataFrame with the correct index and columns
        encoded_chunk_df = pd.DataFrame(encoded_chunk, index=chunk_df.index, columns=snp_order)
        
        # Append the processed chunk to the final CSV
        encoded_chunk_df.to_csv(output_csv, mode='a', header=False)

    print(f"✅ Success! Encoding complete. File saved as '{output_csv}'.")

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    print("\n--- FULL TRACEBACK ---")
    traceback.print_exc()