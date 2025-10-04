import pandas as pd
import numpy as np

# --- Configuration ---
# File with the correct 10,000 SNP columns
feature_file = '/content/drive/MyDrive/crop_former_model/chr10_top10k_snps.csv'

# Name of our new simulated data file
output_simulated_file = '/content/drive/MyDrive/crop_former_model/new_simulated_maize.csv'

# How many new samples to simulate
num_new_samples = 5
# ---------------------

print("Creating a simulated data file for prediction...")

# Read just the header of the feature file to get the column names
try:
    snp_columns = pd.read_csv(feature_file, nrows=0).columns[1:] # Skip the first 'IID' column
    
    # Create some new, fake sample IDs
    new_sample_ids = [f'Maize_Sample_{i+1}' for i in range(num_new_samples)]
    
    # Create a DataFrame with the correct shape
    simulated_df = pd.DataFrame(index=new_sample_ids, columns=snp_columns)
    simulated_df.index.name = 'IID'
    
    # Fill it with random genotype data (the 0-9 encoding)
    # This simulates having new genotype data for the same SNPs
    simulated_data = np.random.randint(0, 10, size=simulated_df.shape)
    simulated_df[:] = simulated_data
    
    # Save the simulated data to a new CSV file
    simulated_df.to_csv(output_simulated_file)
    
    print(f"✅ Success! Created '{output_simulated_file}' with {num_new_samples} simulated samples.")

except FileNotFoundError:
    print(f"❌ Error: Make sure '{feature_file}' exists in your Google Drive.")