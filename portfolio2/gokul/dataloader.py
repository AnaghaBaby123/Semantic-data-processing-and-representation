import torch
from torch.utils.data import Dataset
import random

class OnionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, token_dropout_prob=0.0):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_dropout_prob = token_dropout_prob
    
    def __len__(self):
        return len(self.texts)
    
    def apply_token_dropout(self, input_ids, attention_mask):
        """
        Randomly drops tokens by replacing them with the tokenizer's pad token
        and updating the attention mask accordingly.
        """
        if self.token_dropout_prob <= 0:
            return input_ids, attention_mask
            
        # Create a dropout mask (1 = keep, 0 = drop)
        # Don't drop special tokens (usually at the start and end)
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        }
        
        dropout_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for idx, token_id in enumerate(input_ids):
            if (token_id.item() not in special_tokens and 
                attention_mask[idx] == 1 and  # Only drop tokens that aren't padding
                random.random() < self.token_dropout_prob):
                dropout_mask[idx] = False
        
        # Apply the dropout mask
        dropped_input_ids = input_ids.clone()
        dropped_attention_mask = attention_mask.clone()
        
        # Replace dropped tokens with pad token
        dropped_input_ids[~dropout_mask] = self.tokenizer.pad_token_id
        # Update attention mask
        dropped_attention_mask[~dropout_mask] = 0
        
        return dropped_input_ids, dropped_attention_mask
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # Apply token dropout if probability > 0
        if self.token_dropout_prob > 0:
            input_ids, attention_mask = self.apply_token_dropout(input_ids, attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }