import torch.nn as nn

epochs = 30
patience = 3 # Stop training if validation loss doesn't improve for 3 consecutive epochs

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

# Final Evaluation on test set
test_p, test_r, test_n = evaluate_retrieval(model, train, test_pos, num_users, num_items, K=10)
print(f"\nFinal Test Metrics - Precision@10: {test_p:.4f}, Recall@10: {test_r:.4f}, NDCG@10: {test_n:.4f}")
