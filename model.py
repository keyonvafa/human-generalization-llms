import torch
from transformers import BertForSequenceClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

class BertEnsemble(torch.nn.Module):
  def __init__(self, num_ensembles, model_checkpoint):
    super(BertEnsemble, self).__init__()
    self.num_ensembles = num_ensembles
    self.models = torch.nn.ModuleList([
      BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
      for _ in range(num_ensembles)
    ])
    #
  def forward(self, input_ids, attention_mask):
    outputs = []
    for model in self.models:
      output = model(input_ids, attention_mask=attention_mask)
      outputs.append(output.logits.softmax(-1))
    ensemble_output = torch.mean(torch.stack(outputs), dim=0)
    return ensemble_output

def tokenize_function(q1, q2, previous_correct):
  previous_correct_token = ["correct" if x == 1 else "incorrect" for x in previous_correct]
  prefix = [f'{previous_correct_token[i]}: {q1[i]}' for i in range(len(q1))]
  tokenized_inputs = tokenizer(prefix, list(q2), padding=True, truncation=True)
  input_ids = torch.tensor(tokenized_inputs['input_ids'])
  attention_masks = torch.tensor(tokenized_inputs['attention_mask'])
  return input_ids, attention_masks