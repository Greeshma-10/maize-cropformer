# A Python script to filter a HapMap file and convert it to VCF format.

# --- Configuration ---
# The original, large, unzipped HapMap file with a header
large_hapmap_file = 'chr10_hybrid42K.gt.hmp'

# The file containing the list of sample IDs to keep (one per line)
samples_to_keep_file = 'samples_to_keep.txt'

# The name for our new, filtered, and converted VCF output file
output_vcf_file = 'chr10_filtered_6k.vcf'
# -------------------

print("Starting the HapMap to VCF conversion and filtering script...")

try:
    # Read the list of samples we want to keep into a fast-lookup set
    print(f"Reading sample list from '{samples_to_keep_file}'...")
    with open(samples_to_keep_file, 'r') as f:
        samples_to_keep = {line.strip() for line in f if line.strip()}
    print(f"Found {len(samples_to_keep)} unique samples to keep.")

    # Open the input and output files
    with open(large_hapmap_file, 'r') as in_f, open(output_vcf_file, 'w') as out_f:
        
        # --- Write VCF Header ---
        out_f.write("##fileformat=VCFv4.2\n")
        out_f.write("##source=HapMapConversionScript\n")
        out_f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")

        # --- Process the HapMap Header ---
        print("Processing the header of the large HapMap file...")
        header_line = in_f.readline()
        header_columns = header_line.strip().split('\t')
        
        indices_to_keep = []
        new_header_samples = []
        
        for i, column_name in enumerate(header_columns):
            if i >= 11 and column_name in samples_to_keep:
                indices_to_keep.append(i)
                new_header_samples.append(column_name)
        
        vcf_header = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'] + new_header_samples
        out_f.write('\t'.join(vcf_header) + '\n')
        print(f"VCF header created with {len(new_header_samples)} samples.")

        # --- Process Data Rows ---
        print("Filtering and converting data rows... This may take several minutes.")
        line_count = 0
        for line in in_f:
            line_count += 1
            if line_count % 1000 == 0:
                print(f"  ...processed {line_count} markers...")
            
            parts = line.strip().split('\t')
            
            # Extract HapMap columns
            rs_id, alleles, chrom, pos = parts[0], parts[1], parts[2], parts[3]
            
            # Parse REF and ALT alleles
            if '/' not in alleles or len(alleles) != 3:
                continue # Skip markers that aren't simple bi-allelic SNPs
            ref, alt = alleles.split('/')

            vcf_row = [chrom, pos, rs_id, ref, alt, '.', '.', '.', 'GT']
            
            # Convert genotypes
            for i in indices_to_keep:
                hmp_geno = parts[i]
                vcf_geno = './.' # Default to missing
                if hmp_geno == ref + ref:
                    vcf_geno = '0/0'
                elif hmp_geno == alt + alt:
                    vcf_geno = '1/1'
                elif hmp_geno == ref + alt or hmp_geno == alt + ref:
                    vcf_geno = '0/1'
                
                vcf_row.append(vcf_geno)
            
            out_f.write('\t'.join(vcf_row) + '\n')

    print(f"✅ Success! Filtered VCF data has been saved to '{output_vcf_file}'.")

except FileNotFoundError as e:
    print(f"❌ Error: A required file was not found: {e.filename}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")