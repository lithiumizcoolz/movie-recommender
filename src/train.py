"""
Movie recommender built on Two-Tower model.
Program supports MovieLens 100K and MovieLens Latest datasets, up to ~100K movie ratings.
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Handle import when run from root or from src/
try:
    from src.classes import torch_parse, TwoTowerModel
except ImportError:
    from classes import torch_parse, TwoTowerModel

def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower Movie Recommender Model")
    parser.add_argument("--dataset", type=str, default="ml-100k", choices=["ml-100k", "ml-latest-small", "ml-latest"], help="Dataset to train on. Default = ml-100k")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs. Default = 100")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience. Default = 3")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training. Default = 256")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Fraction of the dataset to use for training (useful for large datasets). Default = 1.0")
    args = parser.parse_args()

    # Setup paths relative to the script location
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "Dataset"))
    ARTIFACTS_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "artifacts"))
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"Loading {args.dataset} dataset...")
    
    if args.dataset == "ml-100k":
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        data_path = os.path.join(DATASET_DIR, "ml-100k", "u.data")
        if not os.path.exists(data_path):
            print(f"Error: Dataset file not found at {data_path}")
            sys.exit(1)
        df = pd.read_csv(data_path, sep='\t', names=columns)
        
        # Load Movie Titles
        item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
                     'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                     'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        movies_path = os.path.join(DATASET_DIR, "ml-100k", "u.item")
        if not os.path.exists(movies_path):
            print(f"Error: Movie titles file not found at {movies_path}")
            sys.exit(1)
        movies_df = pd.read_csv(movies_path, sep='|', names=item_cols, encoding='latin-1')
        movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['movie_title']))
    else:
        # ml-latest-small or ml-latest
        if args.dataset == "ml-latest-small":
            data_path = os.path.join(SRC_DIR, "Dataset", "ml-latest-small", "ratings.csv")
            movies_path = os.path.join(SRC_DIR, "Dataset", "ml-latest-small", "movies.csv")
        else:
            data_path = os.path.join(DATASET_DIR, "ml-latest", "ratings.csv")
            movies_path = os.path.join(DATASET_DIR, "ml-latest", "movies.csv")

        if not os.path.exists(data_path):
            print(f"Error: Dataset file not found at {data_path}")
            if args.dataset == "ml-latest-small":
                print("Please run update_data.py --small first to download the dataset.")
            else:
                print("Please run update_data.py first to download the dataset.")
            sys.exit(1)
        df = pd.read_csv(data_path)
        # Rename columns to allow compatibility of code with ml-latest and ml-latest-small dataset
        df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'}, inplace=True)
        
        # Load Movie Titles
        if not os.path.exists(movies_path):
            print(f"Error: Movie titles file not found at {movies_path}")
            sys.exit(1)
        movies_df = pd.read_csv(movies_path)
        movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

    # Randomly sample dataset if sample_fraction is less than 1.0
    if 0.0 < args.sample_fraction < 1.0:
        print(f"Sampling {args.sample_fraction * 100:.2f}% of the dataset for faster training...")
        df = df.sample(frac=args.sample_fraction).reset_index(drop=True)

    # Define training parameters
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size

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

    print(f"Dataset has {num_users} users, {num_items} movies, and {len(df)} ratings.")

    # Training/validation/test sets (80-10-10 split)
    train, temp = train_test_split(df, test_size=0.2)
    val, test = train_test_split(temp, test_size=0.5)

    # Targets are the positive samples, which we define as ratings >= 3.
    train_pos = train[train['rating'] >= 3].copy()
    val_pos = val[val['rating'] >= 3].copy()
    test_pos = test[test['rating'] >= 3].copy()

    # Track all interactions of the training set for negative sampling 
    user_rated_items = train.groupby('user_idx')['item_idx'].apply(set).to_dict()

    # Create DataLoaders
    train_dataset = torch_parse(train_pos, num_items, user_rated_items, num_negatives=4, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = TwoTowerModel(num_users, num_items, embedding_dim=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    # Initial evaluation before training
    p, r, n = evaluate_retrieval(model, train, test_pos, num_users, num_items, K=10, batch_size=batch_size)
    print(f"Pre-train metrics - Precision@10: {p:.4f}, Recall@10: {r:.4f}, NDCG@10: {n:.4f}")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(SRC_DIR, 'best_model.pt')
    
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
                batch_size=batch_size
            ):
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                preds = model(users, items)
                v_loss = criterion(preds, labels)
                val_total_loss += v_loss.item() * users.size(0)
                val_count += users.size(0)
        avg_val_loss = val_total_loss / val_count
        val_losses.append(avg_val_loss)

        # Run retrieval evaluation on validation set
        val_p, val_r, val_n = evaluate_retrieval(model, train, val_pos, num_users, num_items, K=10, batch_size=batch_size)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Precision@10: {val_p:.4f}, Recall@10: {val_r:.4f}, NDCG@10: {val_n:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # Restore the best model before evaluation
    model.load_state_dict(torch.load(best_model_path))
    print(f"Trained model saved in {best_model_path}.")

    # Final evaluation on test set
    test_p, test_r, test_n = evaluate_retrieval(model, train, test_pos, num_users, num_items, K=10, batch_size=batch_size)
    print(f"Final Test Metrics - Precision@10: {test_p:.4f}, Recall@10: {test_r:.4f}, NDCG@10: {test_n:.4f}")

    # From the trained model, save item embeddings only.
    with torch.no_grad():
        all_item_indices = torch.arange(num_items, dtype=torch.long).to(device)
        item_embeddings = model.get_item_embeddings(all_item_indices).cpu()  # (num_items, 64)

    # Save all artifacts
    print("Saving artifacts...")
    with open(os.path.join(ARTIFACTS_DIR, "item_to_idx.pkl"), "wb") as f:
        pickle.dump(item_to_idx, f)

    with open(os.path.join(ARTIFACTS_DIR, "idx_to_item.pkl"), "wb") as f:
        pickle.dump(idx_to_item, f)

    with open(os.path.join(ARTIFACTS_DIR, "movie_id_to_title.pkl"), "wb") as f:
        pickle.dump(movie_id_to_title, f)

    torch.save(item_embeddings, os.path.join(ARTIFACTS_DIR, "item_embeddings.pt"))

    print(f"All artifacts saved to {ARTIFACTS_DIR}:")
    for fname in os.listdir(ARTIFACTS_DIR):
        size = os.path.getsize(os.path.join(ARTIFACTS_DIR, fname))
        print(f"  {fname}  ({size:,} bytes)")
    print("Done! You can now run: uvicorn api:app --reload")

@torch.no_grad()
def evaluate_retrieval(model, train_df, test_pos_df, num_users, num_items, K=10, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    
    # Pre-compute all item embeddings
    item_indices = torch.arange(num_items, dtype=torch.long).to(device)
    item_reps = []
    for i in range(0, num_items, batch_size):
        batch_items = item_indices[i:i+batch_size]
        item_reps.append(model.get_item_embeddings(batch_items))
    item_reps = torch.cat(item_reps, dim=0) # (num_items, 64)
    
    # Get ground truth positive items from test set
    test_user_pos_items = test_pos_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    # Mask out training set items so they aren't recommended
    train_user_items = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    test_users = list(test_user_pos_items.keys())
    if len(test_users) == 0:
        return 0.0, 0.0, 0.0
        
    precisions = []
    recalls = []
    ndcgs = []
    
    # Process test users in batches to avoid OOM
    # We can use a larger batch size compared to training because gradient computations are not needed.
    eval_batch_size = 1000
    for start_idx in range(0, len(test_users), eval_batch_size):
        batch_users = test_users[start_idx:start_idx + eval_batch_size]
        batch_users_tensor = torch.tensor(batch_users, dtype=torch.long).to(device)
        
        # Get embeddings for this batch of users
        user_reps_batch = model.get_user_embeddings(batch_users_tensor) # (eval_batch_size, 64)
        
        # Compute scores for this batch of users against all items
        # shape: (eval_batch_size, num_items)
        scores_batch = torch.matmul(user_reps_batch, item_reps.T).cpu().numpy()
        
        for i, user_idx in enumerate(batch_users):
            gt_items = test_user_pos_items[user_idx]
            if len(gt_items) == 0:
                continue
                
            user_scores = scores_batch[i]
            
            # Mask out training set items
            if user_idx in train_user_items:
                user_scores[train_user_items[user_idx]] = -np.inf
                
            top_k_indices = np.argsort(user_scores)[::-1][:K]
            hits = [1 if item in gt_items else 0 for item in top_k_indices]
            
            # Precision@K
            precision = sum(hits) / K
            precisions.append(precision)
            
            # Recall@K
            recall = sum(hits) / len(gt_items)
            recalls.append(recall)
            
            # NDCG@K
            idcg = sum([1 / np.log2(idx + 2) for idx in range(min(K, len(gt_items)))])
            dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
            
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

if __name__ == "__main__":
    main()