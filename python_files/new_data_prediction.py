# ===================================================================
# Master Prediction and Evaluation Script for Cropformer
# ===================================================================
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from lightning.pytorch import LightningModule
import os

# ========================
# 1. MODEL DEFINITION
# (This must be included so the script knows the model's structure)
# ========================
class SelfAttention(LightningModule):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.query = torch.nn.Linear(input_size, self.all_head_size)
        self.key = torch.nn.Linear(input_size, self.all_head_size)
        self.value = torch.nn.Linear(input_size, self.all_head_size)
        self.attn_dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, input_size)
        self.LayerNorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(input_size, output_dim)
        self.cnn = torch.nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

    def forward(self, input_tensor):
        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))
        input_tensor_after_cnn = cnn_hidden
        mixed_query_layer = self.query(input_tensor_after_cnn)
        mixed_key_layer = self.key(input_tensor_after_cnn)
        mixed_value_layer = self.value(input_tensor_after_cnn)
        query_layer, key_layer, value_layer = mixed_query_layer, mixed_key_layer, mixed_value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor_after_cnn)
        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))
        return output

# ========================
# 2. CONFIGURATION
# (Update these paths to match your Google Drive)
# ========================
DRIVE_FOLDER = '/content/drive/MyDrive/crop_former_model/'
MODEL_FOLDER = DRIVE_FOLDER

# Input files
TRAIN_DATA_PATH = os.path.join(DRIVE_FOLDER, 'X_train.csv')
TEST_DATA_PATH = os.path.join(DRIVE_FOLDER, 'X_test.csv')
TEST_LABELS_PATH = os.path.join(DRIVE_FOLDER, 'y_test.csv')
NEW_DATA_PATH = os.path.join(DRIVE_FOLDER, 'new_simulated_maize.csv')

# --- Hyperparameters (must match the models you trained) ---
INPUT_SIZE = 10000
HIDDEN_SIZE = 64

# ========================
# 3. MAIN SCRIPT LOGIC
# ========================
if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}\n')

    # --- Part 1: Evaluate the accuracy of each saved model ---
    print("--- Evaluating Accuracy of Saved Models ---")
    
    # Load data for evaluation
    X_train_df = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    X_test_df = pd.read_csv(TEST_DATA_PATH, index_col=0)
    y_test_df = pd.read_csv(TEST_LABELS_PATH, index_col=0)

    # Fit a scaler on the training data
    scaler = StandardScaler()
    scaler.fit(X_train_df.values)
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test_df.values)
    X_test_tensor = torch.from_numpy(X_test_scaled).to(torch.float32).to(DEVICE)
    true_values = y_test_df.values.flatten()
    
    model_accuracies = {}

    for i in range(1, 6):
        model_name = f'best_model_fold_{i}.pth'
        model_path = os.path.join(MODEL_FOLDER, model_name)
        
        # NOTE: This assumes best params are similar across folds. For highest accuracy,
        # you'd need to load the specific params for each fold. We use the best overall.
        best_params = {'num_attention_heads': 4, 'attention_probs_dropout_prob': 0.5}

        model = SelfAttention(**best_params, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            predictions_tensor = model(X_test_tensor)
        
        predicted_values = predictions_tensor.cpu().numpy().flatten()
        
        # Calculate Pearson Correlation
        accuracy = np.corrcoef(predicted_values, true_values)[0, 1]
        model_accuracies[model_name] = accuracy
        print(f"  - {model_name}: Accuracy = {accuracy:.4f}")

    print(f"\nAverage Accuracy: {np.mean(list(model_accuracies.values())):.4f}\n")

    # --- Part 2: Predict DTT for new data ---
    print(f"--- Predicting DTT for New Data from '{os.path.basename(NEW_DATA_PATH)}' ---")
    
    # Load the new data
    new_data_df = pd.read_csv(NEW_DATA_PATH, index_col=0)
    
    # Scale the new data using the SAME scaler fitted on the training data
    new_data_scaled = scaler.transform(new_data_df.values)
    new_data_tensor = torch.from_numpy(new_data_scaled).to(torch.float32).to(DEVICE)

    all_predictions = {}
    
    for i in range(1, 6):
        model_name = f'best_model_fold_{i}.pth'
        model_path = os.path.join(MODEL_FOLDER, model_name)
        
        best_params = {'num_attention_heads': 4, 'attention_probs_dropout_prob': 0.5}

        model = SelfAttention(**best_params, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        with torch.no_grad():
            predictions_tensor = model(new_data_tensor)
        
        all_predictions[f'Model_Fold_{i}'] = predictions_tensor.cpu().numpy().flatten()

    # Create a final results DataFrame
    results_df = pd.DataFrame(all_predictions, index=new_data_df.index)
    results_df['Ensemble_Average_DTT'] = results_df.mean(axis=1)

    print("\n✅ Prediction Complete. Results:\n")
    print(results_df.round(2))