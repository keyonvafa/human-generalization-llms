import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from model import BertEnsemble, tokenize_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv('data/test.csv')

# Load model
num_ensembles = 5
model_checkpoint = 'bert-base-uncased'
model = BertEnsemble(num_ensembles, model_checkpoint)
model.load_state_dict(torch.load('ckpts/bert.pth'))
model.to(device)
model.eval()

input_ids_test, attention_masks_test = tokenize_function(
   test_df['q1'], test_df['q2'], test_df['previous_correct'])
input_ids_test, attention_masks_test = (
  input_ids_test.to(device), attention_masks_test.to(device))
labels_test = torch.tensor(test_df['belief_change']).to(device).long()

test_batch_size = 128
dataloader_test = DataLoader(
   TensorDataset(input_ids_test, attention_masks_test, labels_test), 
   batch_size=test_batch_size, shuffle=False)
preds = []
for batch in tqdm(dataloader_test):
  input_ids, attention_masks, labels = batch
  with torch.no_grad():
    probs = model(input_ids, attention_masks)
    preds.append(probs.cpu().numpy())

preds = np.concatenate(preds)[:, 1]

# Get NLL
nll_overall = -np.mean(
   test_df['belief_change'] * np.log(preds) + 
   (1 - test_df['belief_change']) * np.log(1 - preds))
nll_prev_correct = -np.mean(
   test_df[test_df['previous_correct'] == 1]['belief_change'] * 
   np.log(preds[test_df['previous_correct'] == 1]) + 
   (1 - test_df[test_df['previous_correct'] == 1]['belief_change']) * 
   np.log(1 - preds[test_df['previous_correct'] == 1]))
nll_prev_incorrect = -np.mean(
   test_df[test_df['previous_correct'] == 0]['belief_change'] * 
   np.log(preds[test_df['previous_correct'] == 0]) + 
   (1 - test_df[test_df['previous_correct'] == 0]['belief_change']) * 
   np.log(1 - preds[test_df['previous_correct'] == 0]))
# Get AUC
auc_overall = roc_auc_score(test_df['belief_change'], preds)
auc_prev_correct = roc_auc_score(
   test_df[test_df['previous_correct'] == 1]['belief_change'], 
   preds[test_df['previous_correct'] == 1])
auc_prev_incorrect = roc_auc_score(
   test_df[test_df['previous_correct'] == 0]['belief_change'], 
   preds[test_df['previous_correct'] == 0])

print("BERT")
print("=====================================")
print(f'  NLL Overall: {nll_overall:.3f}')
print(f'  NLL Previous Correct: {nll_prev_correct:.3f}')
print(f'  NLL Previous Incorrect: {nll_prev_incorrect:.3f}')
print(f'  AUC Overall: {auc_overall:.3f}')
print(f'  AUC Previous Correct: {auc_prev_correct:.3f}')
print(f'  AUC Previous Incorrect: {auc_prev_incorrect:.3f}')
print("=====================================")
print("")


# Now baseline 1: previous correct
train_df = pd.read_csv('data/train.csv')
change_prob_given_correct = train_df[train_df['previous_correct'] == 1]['belief_change'].mean()
change_prob_given_incorrect = train_df[train_df['previous_correct'] == 0]['belief_change'].mean()
preds_baseline_1 = np.where(test_df['previous_correct'] == 1, 
                            change_prob_given_correct, 
                            change_prob_given_incorrect)
nll_baseline_1 = -np.mean(test_df['belief_change'] * np.log(preds_baseline_1) + 
                          (1 - test_df['belief_change']) * np.log(1 - preds_baseline_1))
nll_baseline_1_prev_correct = -np.mean(
   test_df[test_df['previous_correct'] == 1]['belief_change'] * 
   np.log(preds_baseline_1[test_df['previous_correct'] == 1]) + 
   (1 - test_df[test_df['previous_correct'] == 1]['belief_change']) * 
   np.log(1 - preds_baseline_1[test_df['previous_correct'] == 1]))
nll_baseline_1_prev_incorrect = -np.mean(
   test_df[test_df['previous_correct'] == 0]['belief_change'] * 
   np.log(preds_baseline_1[test_df['previous_correct'] == 0]) + 
   (1 - test_df[test_df['previous_correct'] == 0]['belief_change']) * 
   np.log(1 - preds_baseline_1[test_df['previous_correct'] == 0]))
auc_baseline_1 = roc_auc_score(test_df['belief_change'], preds_baseline_1)
auc_baseline_1_prev_correct = roc_auc_score(
   test_df[test_df['previous_correct'] == 1]['belief_change'], 
   preds_baseline_1[test_df['previous_correct'] == 1])
auc_baseline_1_prev_incorrect = roc_auc_score(
   test_df[test_df['previous_correct'] == 0]['belief_change'], 
   preds_baseline_1[test_df['previous_correct'] == 0])

print("Baseline 1: Previous correct")
print("=====================================")
print(f'  NLL Overall: {nll_baseline_1:.3f}')
print(f'  NLL Previous Correct: {nll_baseline_1_prev_correct:.3f}')
print(f'  NLL Previous Incorrect: {nll_baseline_1_prev_incorrect:.3f}')
print(f'  AUC Overall: {auc_baseline_1:.3f}')
print(f'  AUC Previous Correct: {auc_baseline_1_prev_correct:.3f}')
print(f'  AUC Previous Incorrect: {auc_baseline_1_prev_incorrect:.3f}')
print("=====================================")
print("")

# Now baseline 2: Logistic regression on previous correct and same source
same_source_train = (train_df['source1'] == train_df['source2']).astype(int)
same_source_test = (test_df['source1'] == test_df['source2']).astype(int)
X_train = np.stack([train_df['previous_correct'], same_source_train], axis=1)
X_test = np.stack([test_df['previous_correct'], same_source_test], axis=1)
logreg = LogisticRegression()
logreg.fit(X_train, train_df['belief_change'])
preds_baseline_2 = logreg.predict_proba(X_test)[:, 1]
nll_baseline_2 = -np.mean(
   test_df['belief_change'] * np.log(preds_baseline_2) + 
   (1 - test_df['belief_change']) * np.log(1 - preds_baseline_2))
nll_baseline_2_prev_correct = -np.mean(
   test_df[test_df['previous_correct'] == 1]['belief_change'] * 
   np.log(preds_baseline_2[test_df['previous_correct'] == 1]) + 
   (1 - test_df[test_df['previous_correct'] == 1]['belief_change']) * 
   np.log(1 - preds_baseline_2[test_df['previous_correct'] == 1]))
nll_baseline_2_prev_incorrect = -np.mean(
   test_df[test_df['previous_correct'] == 0]['belief_change'] * 
   np.log(preds_baseline_2[test_df['previous_correct'] == 0]) + 
   (1 - test_df[test_df['previous_correct'] == 0]['belief_change']) * 
   np.log(1 - preds_baseline_2[test_df['previous_correct'] == 0]))
auc_baseline_2 = roc_auc_score(test_df['belief_change'], preds_baseline_2)
auc_baseline_2_prev_correct = roc_auc_score(
   test_df[test_df['previous_correct'] == 1]['belief_change'], 
   preds_baseline_2[test_df['previous_correct'] == 1])
auc_baseline_2_prev_incorrect = roc_auc_score(
   test_df[test_df['previous_correct'] == 0]['belief_change'], 
   preds_baseline_2[test_df['previous_correct'] == 0])

print("Baseline 2: Logistic regression on previous correct and same source")
print("=====================================")
print(f'  NLL Overall: {nll_baseline_2:.3f}')
print(f'  NLL Previous Correct: {nll_baseline_2_prev_correct:.3f}')
print(f'  NLL Previous Incorrect: {nll_baseline_2_prev_incorrect:.3f}')
print(f'  AUC Overall: {auc_baseline_2:.3f}')
print(f'  AUC Previous Correct: {auc_baseline_2_prev_correct:.3f}')
print(f'  AUC Previous Incorrect: {auc_baseline_2_prev_incorrect:.3f}')
print("=====================================")
print("")

### Calculate AUC per source.
def calculate_auc(group):
    if len(group['belief_change'].unique()) == 1:
        return None
    if len(group) < 5:
        return None
    return roc_auc_score(group['belief_change'], preds[group.index])

auc_per_source1 = test_df.groupby('source1').apply(calculate_auc)
auc_per_source1.dropna(inplace=True)
auc_per_source1.sort_values(ascending=False)

auc_per_source2 = test_df.groupby('source2').apply(calculate_auc)
auc_per_source2.dropna(inplace=True)
auc_per_source2.sort_values(ascending=False)

auc_per_source_pair = test_df.groupby(['source1', 'source2']).apply(calculate_auc)
auc_per_source_pair.dropna(inplace=True)
auc_per_source_pair.sort_values(ascending=False)

## Get predictions for a random sample of correct and incorrect questions.
correct_sample = pd.read_csv('analysis-data/correct_sample.csv')
incorrect_sample = pd.read_csv('analysis-data/incorrect_sample.csv')
correct_input_ids, correct_attention_masks =  tokenize_function(
   correct_sample['q1'], correct_sample['q2'], np.ones(len(correct_sample)))
correct_input_ids, correct_attention_masks = (
   correct_input_ids.to(device), correct_attention_masks.to(device))
incorrect_input_ids, incorrect_attention_masks =  tokenize_function(
   incorrect_sample['q1'], incorrect_sample['q2'], np.zeros(len(incorrect_sample)))
incorrect_input_ids, incorrect_attention_masks = (
   incorrect_input_ids.to(device), incorrect_attention_masks.to(device))

correct_sample_dataloader = DataLoader(
   TensorDataset(correct_input_ids, correct_attention_masks), 
   batch_size=128, shuffle=False)
incorrect_sample_dataloader = DataLoader(
   TensorDataset(incorrect_input_ids, incorrect_attention_masks), 
   batch_size=128, shuffle=False)

prev_correct_preds = []
for batch in tqdm(correct_sample_dataloader):
  input_ids, attention_masks = batch
  with torch.no_grad():
    probs = model(input_ids, attention_masks)
    prev_correct_preds.append(probs.cpu().numpy())

prev_incorrect_preds = []
for batch in tqdm(incorrect_sample_dataloader):
  input_ids, attention_masks = batch
  with torch.no_grad():
    probs = model(input_ids, attention_masks)
    prev_incorrect_preds.append(probs.cpu().numpy())

prev_correct_preds = np.concatenate(prev_correct_preds)[:, 1]
prev_incorrect_preds = np.concatenate(prev_incorrect_preds)[:, 1]

correct_sample['pred'] = prev_correct_preds
incorrect_sample['pred'] = prev_incorrect_preds

correct_sample.to_csv('analysis-data/correct_sample_with_pred.csv', index=False)
incorrect_sample.to_csv('analysis-data/incorrect_sample_with_pred.csv', index=False)

