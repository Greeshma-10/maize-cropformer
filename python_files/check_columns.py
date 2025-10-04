# A simple script to check the number of columns in the first line of a file.

# --- Configuration ---
file_to_check = 'chr10_text_genotypes.raw'
delimiter = ' '
# -------------------

try:
    with open(file_to_check, 'r') as f:
        # Read only the first line to be fast and memory-efficient
        first_line = f.readline()
        
        # Split the line by the space delimiter to get a list of columns
        columns = first_line.strip().split(delimiter)
        
        # Count how many columns were found
        num_columns = len(columns)
        
        print(f"\n✅ The first line of '{file_to_check}' has {num_columns} columns.\n")
        
        # Provide context based on what we expect
        if num_columns > 1000:
            print("This is the correct, large number of columns we expect. The file is ready.")
        else:
            print("WARNING: This number seems low. There might be an issue with the file.")

except FileNotFoundError:
    print(f"❌ Error: The file '{file_to_check}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")