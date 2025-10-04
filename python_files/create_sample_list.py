import pandas as pd

# --- Configuration ---
# The name of your large phenotype file
input_phenotype_file = "C:/Users/GREESHMA/Desktop/cropformer/Ncii_2015_5locs_hybrids.txt"

# The name of the file we want to create for PLINK
output_sample_file = 'samples_to_keep.txt'

# The name of the column that contains the sample IDs
id_column_name = 'Lineid'
# -------------------

try:
    # Read the file. The separator '\s+' is robust and handles tabs or spaces.
    pheno_df = pd.read_csv(input_phenotype_file, sep='\s+')
    
    # Check if the specified ID column exists in the file
    if id_column_name in pheno_df.columns:
        # Select just the column with the sample IDs
        sample_ids = pheno_df[id_column_name]
        
        # Save the list of IDs to the output file
        # We set header=False and index=False so ONLY the IDs are written
        sample_ids.to_csv(output_sample_file, header=False, index=False)
        
        print(f"✅ Success! Created '{output_sample_file}' with {len(sample_ids)} sample IDs.")
        
    else:
        print(f"❌ Error: Column '{id_column_name}' not found in the file.")
        print(f"Please check the file. Found columns: {pheno_df.columns.tolist()}")

except FileNotFoundError:
    print(f"❌ Error: The file '{input_phenotype_file}' was not found.")
    print("Please make sure it's in the same directory as this script.")