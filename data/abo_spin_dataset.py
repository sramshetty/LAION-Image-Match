import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoProcessor, AutoModel
from PIL import Image


processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


def collate_fn(batch):
    features = []
    labels = []
    for x, label in batch:
        imgs = [Image.open(path) for path in x]
        inputs = processor(images=imgs, return_tensors='pt')
        
        features.append(model.get_image_features(**inputs).unsqueeze(0))
        labels += [1] if label else [0]

    return torch.cat(features), labels


class AboSpinsDataset(Dataset):

    def __init__(self, df, mode="train", id_name="spin_id", image_dir="", seq_len=10, negative_ratio=0.5, seed=None):
        self.df = df
        self.mode = mode
        self.id_col = id_name
        self.image_dir = image_dir
        self.unique_ids = self.df[self.id_col].unique()
        self.seq_len = seq_len
        self.neg_ratio = negative_ratio
        self.n = len(self.unique_ids)
        
        if seed is not None:
            torch.manual_seed(seed)
    
    def __len__(self):
        return self.n

    def __getitem__(self, index):
        spin_id = self.unique_ids[index]

        # Determine if this sample is a negative sample or not
        negative = True if torch.rand(1)[0] < self.neg_ratio else False 

        id_rows = self.df[self.df[self.id_col] == spin_id]
        seq_len = min(len(id_rows), self.seq_len)
        
        samples = id_rows.sample(seq_len)

        # If negative sample, randomly find another image that is not share the same id
        if negative:
            x_matches = samples.iloc[:-1]
            y_match = self.df[self.df[self.id_col] != spin_id].sample(1)
        else:
            x_matches = samples.iloc[:-1]
            y_match = samples.iloc[-1:]
        
        paths = pd.concat([x_matches, y_match], ignore_index=True)['path'].tolist()
        return [self.image_dir + p for p in paths], not negative