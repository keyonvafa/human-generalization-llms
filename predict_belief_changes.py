import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import pdb
from model import BertEnsemble, tokenize_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_ensembles = 5
num_samples = 10000 
rs = np.random.RandomState(42)
model_checkpoint = 'bert-base-uncased'
model = BertEnsemble(num_ensembles, model_checkpoint)
model.load_state_dict(torch.load('ckpts/bert.pth'))
model.to(device)
model.eval()

questions = pd.read_csv('analysis-data/all_questions.csv')
# Create a new dataframe where we randomly sample pairs of rows in questions
num_rows = len(questions)
indices = rs.choice(num_rows, size=(num_samples, 2), replace=True)
sampled_pairs = pd.DataFrame({
    'q1': questions['question'].iloc[indices[:, 0]].reset_index(drop=True),
    'q2': questions['question'].iloc[indices[:, 1]].reset_index(drop=True),
    'answer1': questions['answer'].iloc[indices[:, 0]].reset_index(drop=True),
    'answer2': questions['answer'].iloc[indices[:, 1]].reset_index(drop=True),
    'source1': questions['source'].iloc[indices[:, 0]].reset_index(drop=True),
    'source2': questions['source'].iloc[indices[:, 1]].reset_index(drop=True)
})

input_ids_prev_incorrect, attention_masks_prev_incorrect = tokenize_function(
  sampled_pairs['q1'], sampled_pairs['q2'], np.zeros(len(sampled_pairs)))
input_ids_prev_correct, attention_masks_prev_correct = tokenize_function(
  sampled_pairs['q1'], sampled_pairs['q2'], np.ones(len(sampled_pairs)))
input_ids_prev_incorrect, attention_masks_prev_incorrect = (
  input_ids_prev_incorrect.to(device), attention_masks_prev_incorrect.to(device))
input_ids_prev_correct, attention_masks_prev_correct = (
  input_ids_prev_correct.to(device), attention_masks_prev_correct.to(device))

batch_size = 1024
dataloader_test = DataLoader(
  TensorDataset(input_ids_prev_incorrect, attention_masks_prev_incorrect), 
  batch_size=batch_size, shuffle=False)
preds_prev_incorrect = []
for batch in tqdm(dataloader_test):
  input_ids, attention_masks = batch
  with torch.no_grad():
    probs = model(input_ids, attention_masks)
    preds_prev_incorrect.append(probs.cpu().numpy())

# Do the same with previous correct
dataloader_test = DataLoader(
  TensorDataset(input_ids_prev_correct, attention_masks_prev_correct), 
  batch_size=batch_size, shuffle=False)
preds_prev_correct = []
for batch in tqdm(dataloader_test):
  input_ids, attention_masks = batch
  with torch.no_grad():
    probs = model(input_ids, attention_masks)
    preds_prev_correct.append(probs.cpu().numpy())

sampled_pairs_correct = sampled_pairs.copy()
sampled_pairs_correct['pred'] = np.concatenate(preds_prev_correct)[:, 1]
sampled_pairs_incorrect = sampled_pairs.copy()
sampled_pairs_incorrect['pred'] = np.concatenate(preds_prev_incorrect)[:, 1]

sampled_pairs_correct.to_csv("analysis-data/large_correct_sample_with_pred.csv", index=False)
sampled_pairs_incorrect.to_csv("analysis-data/large_incorrect_sample_with_pred.csv", index=False)

print("Saved data!")