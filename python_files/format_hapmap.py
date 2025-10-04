# A script to add a proper HapMap header AND placeholder marker columns.
# This version also standardizes the delimiters in the data to prevent column mismatch errors.

# --- Configuration ---
genotype_file = 'chr10_hybrid42K.gt.hmp'
sample_list_file = 'samples_to_keep.txt'
output_file = 'chr10_formatted.hmp.txt'
# -------------------

print("Starting the script to format the HapMap file...")

try:
    # Step 1: Read unique sample IDs.
    print(f"Reading unique sample IDs from '{sample_list_file}'...")
    with open(sample_list_file, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines() if line.strip()]
    num_samples = len(sample_ids)
    print(f"Found {num_samples} unique sample IDs.")

    # Step 2: Construct the header.
    standard_header = "rs#\talleles\tchrom\tpos\tstrand\tassembly#\tcenter\tprotLSID\tassayLSID\tpanelLSID\tQCcode"
    sample_header = "\t".join(sample_ids)
    full_header = f"{standard_header}\t{sample_header}\n"
    
    # Step 3: Create the new file and process line by line.
    print(f"Creating new formatted file: '{output_file}'...")
    with open(output_file, 'w') as out_f:
        # Write the header.
        out_f.write(full_header)
        
        # Open the genotype file to read and re-format.
        print(f"Processing and standardizing genotype data from '{genotype_file}'...")
        line_num = 0
        with open(genotype_file, 'r') as in_f:
            for data_line in in_f:
                if not data_line.strip():
                    continue
                line_num += 1
                
                # THIS IS THE NEW, ROBUST PART
                # Split the line by any whitespace and join it back with single tabs.
                genotypes = data_line.strip().split()
                
                # Check if the number of genotypes matches the number of samples
                if len(genotypes) != num_samples:
                    print(f"!! WARNING: Line {line_num} in the data has {len(genotypes)} genotypes, but you have {num_samples} samples in your header. This line will be skipped.")
                    continue # Skip this malformed line
                
                standardized_data = "\t".join(genotypes)
                # ------------------------------------

                # Create placeholder marker info.
                placeholder_marker_info = f"SNP_{line_num}\tNA\t10\t{line_num}\tNA\tNA\tNA\tNA\tNA\tNA\tNA"
                
                # Combine placeholders with the standardized data.
                full_data_line = f"{placeholder_marker_info}\t{standardized_data}\n"
                
                out_f.write(full_data_line)
    
    print(f"✅ Success! The new file '{output_file}' has been created with standardized delimiters.")

except FileNotFoundError as e:
    print(f"❌ Error: A required file was not found: {e.filename}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")