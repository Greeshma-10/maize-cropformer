# A script to remove duplicate lines from a text file.

# --- Configuration ---
# This is the file we will read from and write back to.
sample_file = 'samples_to_keep.txt'
# -------------------

print(f"Reading '{sample_file}' to remove duplicates...")

try:
    # Read all lines from the file
    with open(sample_file, 'r') as f:
        all_ids = [line.strip() for line in f.readlines() if line.strip()]
    original_count = len(all_ids)

    # Use dict.fromkeys to get unique IDs while preserving order
    unique_ids = list(dict.fromkeys(all_ids))
    unique_count = len(unique_ids)

    if original_count == unique_count:
        print("No duplicates found. The file is already clean.")
    else:
        # Overwrite the original file with only the unique IDs
        with open(sample_file, 'w') as f:
            for sample_id in unique_ids:
                f.write(f"{sample_id}\n")
        print(f"Removed {original_count - unique_count} duplicates. {unique_count} unique IDs remain.")
        print(f"✅ Success! '{sample_file}' has been updated.")

except FileNotFoundError:
    print(f"❌ Error: The file '{sample_file}' was not found.")