import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
import pdb
import os
from model import BertEnsemble, tokenize_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('data/train.csv')

model_checkpoint = 'bert-base-uncased' 
input_ids_train, attention_masks_train =  tokenize_function(
  train_df['q1'], train_df['q2'], train_df['previous_correct'])
input_ids_train, attention_masks_train = (
  input_ids_train.to(device), attention_masks_train.to(device))
labels_train = torch.tensor(train_df['belief_change']).to(device).long()

batch_size = 32
dataloader_train = DataLoader(
  TensorDataset(
    input_ids_train, 
    attention_masks_train, 
    labels_train), batch_size=batch_size)

num_ensembles = 5
bert_ensemble = BertEnsemble(num_ensembles, model_checkpoint).to(device)

epochs = 2  
for ensemble in range(num_ensembles):
  bert_model = bert_ensemble.models[ensemble]
  optimizer = AdamW(bert_model.parameters(), lr=5e-5)
  loss_fn = torch.nn.CrossEntropyLoss()
  total_steps = len(dataloader_train) * epochs
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps)
  for epoch in range(epochs):
    bert_model.train()
    total_train_loss = 0
    for step, batch in enumerate(tqdm(
        dataloader_train, desc=f"Ensemble {ensemble} Epoch {epoch}")):
      batch = tuple(t.to(device) for t in batch)  
      b_input_ids, b_input_mask, b_labels = batch
      bert_model.zero_grad() 
      outputs = bert_model(b_input_ids, attention_mask=b_input_mask)
      loss = loss_fn(
        outputs.logits, torch.nn.functional.one_hot(b_labels, 2).float()) 
      total_train_loss += loss.item()
      loss.backward()
      _ = torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0) 
      optimizer.step()
      scheduler.step()
    avg_train_loss = total_train_loss / len(dataloader_train)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")


pdb.set_trace()

save_dir = 'ckpts'
os.makedirs(save_dir, exist_ok=True)
torch.save(bert_ensemble.state_dict(), os.path.join(save_dir, 'bert.pth'))

print("Training complete!")