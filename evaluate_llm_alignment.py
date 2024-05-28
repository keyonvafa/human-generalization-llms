import numpy as np
import pandas as pd
import os
import pdb
import argparse
import pickle
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--llm',
                    default='gpt-4',
                    help="The LLM used for evaluation",
                    type=str,
                    choices=['alpaca-7b', 'llama-2-7b-chat', 'llama-2-13b-chat', 
                             'llama-2-70b-chat', 'StripedHyena-Nous-7B', 
                             'mistralai/Mistral-7B-Instruct-v0.2', 
                             'gpt-3.5-turbo', 'gpt-4'])
args = parser.parse_args()
llm = args.llm

# Load LLM responses.
pickle_dir = f'llm-responses/{llm}/'
with open(os.path.join(pickle_dir, 'responses.pkl'), 'rb') as f:
  responses = pickle.load(f)

# Create prompt from each question
df_correct_sample = pd.read_csv('analysis-data/correct_sample_with_pred.csv')
df_incorrect_sample = pd.read_csv('analysis-data/incorrect_sample_with_pred.csv')
assert all(df_correct_sample['q1'] == df_incorrect_sample['q1'])
assert all(df_correct_sample['q2'] == df_incorrect_sample['q2'])

actual_previous_corrects = []
actual_current_corrects = []

for i in range(len(df_correct_sample)):
  previous_q = df_correct_sample.iloc[i]['q1']
  current_q = df_correct_sample.iloc[i]['q2']
  previous_subject = df_correct_sample.iloc[i]['source1']
  current_subject = df_correct_sample.iloc[i]['source2']
  previous_target = df_correct_sample.iloc[i]['answer1']
  current_target = df_correct_sample.iloc[i]['answer2']
  #
  previous_task = utils.get_task(previous_subject)
  previous_response_type = utils.get_response_type(previous_task)
  previous_valid_responses, previous_prompt_end = utils.get_valid_responses(
    previous_response_type, previous_task)
  previous_prompt = (previous_q.replace("<br>", "\n").replace('<b>', '').replace('</b>', '') 
                     + "\n" + previous_prompt_end)
  # Generate a unique key for each question
  if previous_prompt in responses:
    previous_correct = responses[previous_prompt]
  else:
    raise ValueError("LLM response not found")
    # Uncomment code below to recompute the response
    # previous_correct = utils.check_answer(utils.get_answer(previous_prompt, llm), previous_target, previous_response_type, previous_valid_responses)
    # responses[previous_prompt] = previous_correct
  current_task = utils.get_task(current_subject)
  current_response_type = utils.get_response_type(current_task)
  current_valid_responses, current_prompt_end = utils.get_valid_responses(current_response_type, current_task)
  current_prompt = (current_q.replace("<br>", "\n").replace('<b>', '').replace('</b>', '') 
                    + "\n" + current_prompt_end)
  # Generate a unique key for each question
  if current_prompt in responses:
    current_correct = responses[current_prompt]
  else:
    raise ValueError("LLM response not found")
    # Uncomment code below to recompute the response
    # current_correct = utils.check_answer(utils.get_answer(current_prompt, llm), current_target, current_response_type, current_valid_responses)
    # responses[current_prompt] = current_correct
  #
  actual_previous_corrects.append(previous_correct)
  actual_current_corrects.append(current_correct)  

actual_previous_corrects = np.array(actual_previous_corrects)
actual_current_corrects = np.array(actual_current_corrects)

# From survey
average_prior = 0.629
average_posterior_given_correct = 0.663
average_posterior_given_incorrect = 0.551

change_probs = np.where(actual_previous_corrects == 1, 
                        df_correct_sample['pred'], 
                        df_incorrect_sample['pred'])
# What posteriors would be if everyone changed.
full_posteriors = np.where(actual_previous_corrects == 1, 
                           average_posterior_given_correct, 
                           average_posterior_given_incorrect)
# Posterior as mixture model:
posterior_probs = change_probs * full_posteriors + (1-change_probs) * average_prior

print(f"Model: {llm}")
print("-----------------")
posterior_prob_acc = np.where(actual_current_corrects == 1, 
                              posterior_probs, 
                              1 - posterior_probs)
weighted_alphas = [1, 19, 50, 99]
print("Weighted generalized accuracy:")
for alpha in weighted_alphas:
  acc = np.average(posterior_prob_acc, 
                   weights=np.where(actual_current_corrects == 0, alpha, 1))
  print(f"  Implied threshold={(alpha / (alpha + 1)):.2f}, accuracy={acc:.3f}")

# Now do the same thing for nll
posterior_nll = np.where(actual_current_corrects == 1, 
                         -np.log(posterior_probs), 
                         -np.log(1 - posterior_probs))
print("Weighted generalized NLL")
for alpha in weighted_alphas:
  nll = np.average(posterior_nll, 
                   weights=np.where(actual_current_corrects == 0, alpha, 1))
  print(f"  Implied threshold={(alpha / (alpha + 1)):.2f}, NLL={nll:.3f}")
print()
