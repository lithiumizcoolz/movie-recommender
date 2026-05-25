"""
save_model.py
Run this once after training your model in the notebook to export all
artifacts the API needs. Make sure best_model.pt already exists.
"""
import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Rebuild the ID mappings from the dataset ──────────────────────────────
print("Loading dataset and rebuilding ID mappings...")
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
df.dropna(inplace=True)

user_to_idx = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}
item_to_idx = {iid: idx for idx, iid in enumerate(df['item_id'].unique())}
idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

num_users = len(user_to_idx)
num_items = len(item_to_idx)
print(f"  {num_users} users, {num_items} movies")

# ── 2. Load movie titles ──────────────────────────────────────────────────────
print("Loading movie titles...")
item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
             'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
             'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
             'Thriller', 'War', 'Western']
movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['movie_title']))

# ── 3. Re-define model architecture (must match training) ────────────────────
class TwoTowerRetrievalModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        user_rep = self.user_embedding(user_indices)
        item_rep = self.item_embedding(item_indices)
        return torch.sum(user_rep * item_rep, dim=-1)

    def get_item_embedding(self, item_indices):
        return self.item_embedding(item_indices)

# ── 4. Load trained weights ───────────────────────────────────────────────────
print("Loading trained model weights from best_model.pt...")
model = TwoTowerRetrievalModel(num_users, num_items, embedding_dim=64)
model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
model.eval()

# ── 5. Pre-compute all item embeddings ───────────────────────────────────────
print("Pre-computing item embeddings...")
with torch.no_grad():
    all_item_indices = torch.arange(num_items, dtype=torch.long)
    item_embeddings = model.get_item_embedding(all_item_indices)  # (num_items, 64)

# ── 6. Save all artifacts ────────────────────────────────────────────────────
print("Saving artifacts...")
with open(f"{ARTIFACTS_DIR}/item_to_idx.pkl", "wb") as f:
    pickle.dump(item_to_idx, f)

with open(f"{ARTIFACTS_DIR}/idx_to_item.pkl", "wb") as f:
    pickle.dump(idx_to_item, f)

with open(f"{ARTIFACTS_DIR}/movie_id_to_title.pkl", "wb") as f:
    pickle.dump(movie_id_to_title, f)

torch.save(item_embeddings, f"{ARTIFACTS_DIR}/item_embeddings.pt")

print("\nAll artifacts saved to ./artifacts/:")
for fname in os.listdir(ARTIFACTS_DIR):
    size = os.path.getsize(f"{ARTIFACTS_DIR}/{fname}")
    print(f"  {fname}  ({size:,} bytes)")
print("\nDone! You can now run: uvicorn api:app --reload")
