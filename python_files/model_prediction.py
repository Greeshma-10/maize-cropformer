import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from lightning.pytorch import LightningModule

# (The model definition is copied here for the script to be self-contained)
class SelfAttention(LightningModule):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, learning_rate=0.001):
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

# --- Main Prediction Logic ---
if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    # --- UPDATE THESE FILE PATHS ---
    # Use the model from the fold with the highest score (e.g., Fold 5)
    model_path = '/content/drive/MyDrive/crop_former_model/best_model_fold_5.pth'
    
    # We need the training data to fit the scaler correctly
    train_data_path = '/content/drive/MyDrive/crop_former_model/X_train.csv'
    
    # This is the "new" data we want to predict on
    test_data_path = '/content/drive/MyDrive/crop_former_model/X_test.csv'
    
    # This is where the final predictions will be saved
    output_path = '/content/drive/MyDrive/crop_former_model/predicted_result.csv'
    # --------------------------------

    # --- Hyperparameters (must match the trained model) ---
    input_size = 10000 
    # Use the best params found for your best fold.
    # From your log, Fold 5 used: {'num_attention_heads': 4, 'attention_probs_dropout_prob': 0.5}
    num_attention_heads = 4
    attention_probs_dropout_prob = 0.5
    
    # Initialize the model structure
    model = SelfAttention(num_attention_heads=num_attention_heads, input_size=input_size, 
                          hidden_size=64, output_dim=1, kernel_size=3,
                          attention_probs_dropout_prob=attention_probs_dropout_prob).to(DEVICE)

    # Load the saved weights
    print(f"Loading trained model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # --- CRITICAL FIX: SCALING THE DATA ---
    print("Loading training data to fit the scaler...")
    X_train = pd.read_csv(train_data_path, index_col=0)
    
    scaler = StandardScaler()
    scaler.fit(X_train.values) # Fit the scaler ONLY on the training data
    
    print(f"Loading and scaling test data from: {test_data_path}")
    X_test = pd.read_csv(test_data_path, index_col=0)
    X_test_scaled = scaler.transform(X_test.values) # Transform the test data
    # -----------------------------------------
    
    X_test_tensor = torch.from_numpy(X_test_scaled).to(torch.float32).to(DEVICE)
    
    # Make predictions
    print("Making predictions on the test data...")
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
    
    # Save predictions
    pd.DataFrame(output.cpu().numpy(), columns=['predicted_value'], index=X_test.index).to_csv(output_path)
    
    print(f"✅ Success! Predictions saved to '{output_path}'.")