"""
Movie recommender built on Two-Tower model
Program is written for MovieLens 100K movie ratings dataset published in 1998, but you can replace it with your own dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Get the MovieLens 100K dataset. You can input your own dataset here. 
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('../ml-100k-dataset/u.data', sep='\t', names=columns)

# Define the number of epochs of the model training (`epochs`) and the threshold of early stopping to prevent overfitting (`patience`).
epochs = 30
patience = 3 # Stop training if validation loss doesn't improve for 3 consecutive epochs

# Remove rows with missing values in-place
df.dropna(inplace=True)

# Map user and item IDs to contiguous indices
user_to_idx = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}
item_to_idx = {iid: idx for idx, iid in enumerate(df['item_id'].unique())}

idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

num_users = len(user_to_idx)
num_items = len(item_to_idx)

df['user_idx'] = df['user_id'].map(user_to_idx)
df['item_idx'] = df['item_id'].map(item_to_idx)

# Training/validation/test sets (80-10-10 split)
train, temp = train_test_split(df, test_size=0.2)
val, test = train_test_split(temp, test_size=0.5)

# Targets are the positive samples, which we define as ratings >= 3.
train_pos = train[train['rating'] >= 3].copy()
val_pos = val[val['rating'] >= 3].copy()
test_pos = test[test['rating'] >= 3].copy()

# Track all interactions of the training set for negative sampling 
user_rated_items = train.groupby('user_idx')['item_idx'].apply(set).to_dict()

# Set up PyTorch tensors
class torch_parse(Dataset):
    def __init__(self, df_pos, num_items, user_rated_items, num_negatives=4, is_training=True):
        self.df_pos = df_pos.reset_index(drop=True)
        self.num_items = num_items
        self.user_rated_items = user_rated_items
        self.num_negatives = num_negatives
        self.is_training = is_training
        
        self.users = self.df_pos['user_idx'].values
        self.items = self.df_pos['item_idx'].values
        
    def __len__(self):
        if self.is_training:
            # Generate 1 positive slot + `num_negatives=4` negative slots for every positive interaction. 
            # The positive/negative slots will store positive/negative samples later.
            return len(self.df_pos) * (1 + self.num_negatives)
        return len(self.df_pos)
    
    def __getitem__(self, idx):
        if self.is_training:
            # `idx=0`is positive slot, `idx=1` to `idx=4` are negative slots. and repeat every 5 slots.
            pos_idx = idx // (1 + self.num_negatives)
            is_neg = (idx % (1 + self.num_negatives)) > 0
            
            user = self.users[pos_idx] # Get the user who made the positive interaction.
            
            if not is_neg: # If this slot is positive
                item = self.items[pos_idx] # Use the movie that this user rated
                label = 1.0 # 1 means liked
            else: # This is a negative slot
                # Use a random movie that the user has not rated
                rated_set = self.user_rated_items.get(user, set())
                neg_item = np.random.randint(0, self.num_items)
                while neg_item in rated_set: # Keep retrying the random picking of a negative sample until you get an unrated movie
                    neg_item = np.random.randint(0, self.num_items)
                item = neg_item
                label = 0.0 # 0 means disliked
                
            return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
        else:
            user = self.users[idx]
            item = self.items[idx]
            return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(1.0, dtype=torch.float32)

# Create DataLoaders
train_dataset = torch_parse(train_pos, num_items, user_rated_items, num_negatives=4, is_training=True)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        # User Tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # Item (Movie) Tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with small random values, sampled from normal distribution, to help convergence
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, user_indices, item_indices):
        user_rep = self.user_embedding(user_indices)
        item_rep = self.item_embedding(item_indices)
        # Dot product similarity. For each sample, sum the similarity score over all 64 embeddings.
        scores = torch.sum(user_rep * item_rep, dim=-1)
        return scores
    
    def get_user_embedding(self, user_indices):
        return self.user_embedding(user_indices)
        
    def get_item_embedding(self, item_indices):
        return self.item_embedding(item_indices)

@torch.no_grad()
def evaluate_retrieval(model, train_df, test_pos_df, num_users, num_items, K=10, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    
    # Pre-compute all item embeddings
    item_indices = torch.arange(num_items, dtype=torch.long).to(device)
    item_reps = []
    for i in range(0, num_items, batch_size):
        batch_items = item_indices[i:i+batch_size]
        item_reps.append(model.get_item_embedding(batch_items))
    item_reps = torch.cat(item_reps, dim=0)
    
    # Pre-compute all user embeddings
    user_indices = torch.arange(num_users, dtype=torch.long).to(device)
    user_reps = []
    for i in range(0, num_users, batch_size):
        batch_users = user_indices[i:i+batch_size]
        user_reps.append(model.get_user_embedding(batch_users))
    user_reps = torch.cat(user_reps, dim=0)
    
    # Compute similarity scores
    all_scores = torch.matmul(user_reps, item_reps.T).cpu().numpy()
    
    # Mask out training set items so they aren't recommended
    train_user_items = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    for user_idx, item_idxs in train_user_items.items():
        all_scores[user_idx, item_idxs] = -np.inf
        
    # Get ground truth positive items from test set
    test_user_pos_items = test_pos_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for user_idx, gt_items in test_user_pos_items.items():
        if len(gt_items) == 0:
            continue
            
        user_scores = all_scores[user_idx]
        top_k_indices = np.argsort(user_scores)[::-1][:K]
        
        hits = [1 if item in gt_items else 0 for item in top_k_indices]
        
        # Precision@K
        precision = sum(hits) / K
        precisions.append(precision)
        
        # Recall@K
        recall = sum(hits) / len(gt_items)
        recalls.append(recall)
        
        # NDCG@K
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(K, len(gt_items)))]) # Ideal discounted cumulative gain, normalization constant for dcg
        dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)]) # Discounted cumulative gain
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
        
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TwoTowerModel(num_users, num_items, embedding_dim=64).to(device)
criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss is more numerically stable than Sigmoid (converts inputs to probabilities) followed by BCELoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

# Initial evaluation before training
p, r, n = evaluate_retrieval(model, train, test_pos, num_users, num_items, K=10)
print(f"Pre-train metrics - Precision@10: {p:.4f}, Recall@10: {r:.4f}, NDCG@10: {n:.4f}")

train_losses = []
val_losses = []
best_val_loss = float('inf')
print(f"Starting training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for users, items, labels in train_loader:
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * users.size(0)
    
    avg_loss = total_loss / len(train_dataset)
    train_losses.append(avg_loss)
    
    # Compute validation loss. No gradients needed.
    model.eval()
    val_total_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for users, items, labels in DataLoader(
            torch_parse(val_pos, num_items, user_rated_items, num_negatives=4, is_training=True),
            batch_size=256
        ):
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            preds = model(users, items)
            v_loss = criterion(preds, labels)
            val_total_loss += v_loss.item() * users.size(0)
            val_count += users.size(0)
    avg_val_loss = val_total_loss / val_count
    val_losses.append(avg_val_loss)

    # Run retrieval evaluation on validation set
    val_p, val_r, val_n = evaluate_retrieval(model, train, val_pos, num_users, num_items, K=10)
    print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Precision@10: {val_p:.4f}, Recall@10: {val_r:.4f}, NDCG@10: {val_n:.4f}")
    
    # Stop model training the moment model stops improving (ie. validation loss stops decreasing) for `patience` epochs in a row. (known as Early stopping)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pt')  # Save best weights
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}.")
        break

# Restore the best model before evaluation
model.load_state_dict(torch.load('best_model.pt'))
print("Trained model saved in best_model.pt.")

# Final Evaluation on test set
test_p, test_r, test_n = evaluate_retrieval(model, train, test_pos, num_users, num_items, K=10)
print(f"Final Test Metrics - Precision@10: {test_p:.4f}, Recall@10: {test_r:.4f}, NDCG@10: {test_n:.4f}")
