import pandas as pd
import numpy as np

# --- IMPORTANT: UPDATE THESE FILE PATHS ---
# Path to the file with your model's predictions
predicted_file = '/content/drive/MyDrive/crop_former_model/predicted_result.csv'

# Path to the file with the true, original test labels
true_labels_file = '/content/drive/MyDrive/crop_former_model/y_test.csv'
# -----------------------------------------

# Load the datasets
predicted_df = pd.read_csv(predicted_file)
true_df = pd.read_csv(true_labels_file)

# Extract the numerical values
predicted_values = predicted_df['predicted_value'].values
true_values = true_df.iloc[:, 1].values # Select the second column which contains the phenotype values

# Calculate the Pearson Correlation Coefficient (PCC)
# np.corrcoef returns a 2x2 matrix, the value at [0, 1] is the correlation
accuracy = np.corrcoef(predicted_values, true_values)[0, 1]

print(f"✅ Model Accuracy (Pearson Correlation Coefficient): {accuracy:.4f}")