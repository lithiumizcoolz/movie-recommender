import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class torch_parse(Dataset):
    """
    This class parses the dataset into PyTorch tensors and creates `num_negatives` negative samples for every positive sample.
    """
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
                # Get a random movie that the user has not rated
                rated_set = self.user_rated_items.get(user, set())
                neg_item = np.random.randint(0, self.num_items)

                # Keep retrying the random picking of a negative sample until you get an unrated movie
                # If a certain user has rated almost all movies in the dataset and there are < `num_negatives` unrated movies left, the while loop would run infinitely.
                # To prevent infinite loops, we set the limit `whilelimit` to the while loop.
                whilelimit = 100
                whilecount = 0
                while neg_item in rated_set and whilecount < whilelimit:
                    neg_item = np.random.randint(0, self.num_items)
                    whilecount += 1
                item = neg_item
                label = 0.0 # 0 means disliked
                
            return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
        else:
            user = self.users[idx]
            item = self.items[idx]
            return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(1.0, dtype=torch.float32)

class TwoTowerModel(nn.Module):
    """
    This class implements the two-tower model.
    """

    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()

        # User Tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # Item (Movie) Tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with small random weights, sampled from normal distribution, to help convergence
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, user_indices, item_indices):
        # We use a shallow Two-Tower neural network due to the low dimension of the dataset and small sample size. 
        # A deeper neural network introduces more weights, and would risk overfitting on such a dataset. 
        # user_rep and item_rep are each a single linear layer of embeddings.
        user_rep = self.user_embedding(user_indices)
        item_rep = self.item_embedding(item_indices)

        # Dot product similarity. For each sample, sum the similarity score over all embedding-pairs.
        scores = torch.sum(user_rep * item_rep, dim=-1)
        return scores
    
    def get_user_embeddings(self, user_indices):
        return self.user_embedding(user_indices)
        
    def get_item_embeddings(self, item_indices):
        return self.item_embedding(item_indices)