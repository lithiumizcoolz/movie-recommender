import pandas as pd

columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)

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
train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

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
