# 导入相关的包
import os
import gc
import cv2
import copy
import time
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

MODEL_PATHS = [
    '../input/deberta_base-v3/Loss-Fold-0.bin',
    '../input/deberta_base-v3/Loss-Fold-1.bin',
    '../input/deberta_base-v3/Loss-Fold-2.bin'
]

# 设置CFG参数

CONFIG = dict(
    seed = 666,
    model_name = '../input/deberta-v3-base',
    test_batch_size = 8,
    max_length = 512,
    num_classes = 3,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])

test_path = "../input/feedback-prize-effectiveness/test"

def get_essay(essay_id):
    essay_path = os.path.join(test_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

df = pd.read_csv("../input/feedback-prize-effectiveness/test.csv")
df['essay_text'] = df['essay_id'].apply(get_essay)
df.head()

with open("../input/deberta_base-v3/le.pkl", "rb") as fp:
    encoder = joblib.load(fp)

print(encoder.classes_)


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse = df['discourse_text'].values
        self.essay = df['essay_text'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        discourse = self.discourse[index]
        essay = self.essay[index]
        text = discourse + " " + self.tokenizer.sep_token + " " + essay
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }

test_dataset = FeedBackDataset(df, CONFIG['tokenizer'], max_length=CONFIG['max_length'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['test_batch_size'],
                         num_workers=2, shuffle=False, pin_memory=True)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, CONFIG['num_classes'])

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs


@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    preds = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        outputs = F.softmax(outputs, dim=1)
        preds.append(outputs.cpu().detach().numpy())

    preds = np.concatenate(preds)
    gc.collect()

    return preds


def inference(model_paths, dataloader, device):
    final_preds = []
    for i, path in enumerate(model_paths):
        model = FeedBackModel(CONFIG['model_name'])
        model.to(CONFIG['device'])
        model.load_state_dict(torch.load(path))

        print(f"Load model and predictions {i + 1}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds

model_preds = inference(MODEL_PATHS, test_loader, CONFIG['device'])

sample_submission = pd.read_csv("../input/feedback-prize-effectiveness/sample_submission.csv")
print(sample_submission.head())

sample_submission['Adequate'] = model_preds[:, 0]
sample_submission['Effective'] = model_preds[:, 1]
sample_submission['Ineffective'] = model_preds[:, 2]

print(sample_submission.head())
