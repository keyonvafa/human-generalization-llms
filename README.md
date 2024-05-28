# Measuring the Human Generalization Function
Source code for the paper "Do Large Language Models Generalize the way People Expect? Measuring the Human Generalization Function" by Keyon Vafa, Ashesh Rambachan, and Sendhil Mullainathan.

## Data

The folder `data/` contains the data we collect about human generalizations. There are two folders, `train.csv` and `test.csv`. `train.csv` contains the 18480 examples used to train the model, and `test.csv` contains the 492 examples used for evaluation. See our paper for more information for how these datasets are constructed.

Each CSV contains the following columns:
- `q1`: The question humans generalize from.
- `q2`: The question humans generalize to.
- `source1`: The dataset source of the question humans generalize from.
- `source2`: The dataset source of the question humans generalize to.
- `previous_correct`: Whether the LLM answers `q1` correctly (1) or incorrectly (0).
- `belief_change`: Whether humans update their belief about whether the LLM will respond to `q2` correctly after seeing the LLM's response to `q1` (1 if they change their belief, 0 otherwise).

For example, human survey respondents first see `q2`, and are asked for their belief that an LLM will respond to the question correctly. They are then shown how the LLM responded to `q1` (given by `previous_correct`), and are given the option to change their belief on how the model would respond to `q2` (recorded in `belief_change`). Humans do not see `source1` or `source2`. The human generalization function models `belief_change` as a function of `q1`, `q2`, and `previous_correct`. Our goal is to model this function. 


## Analyzing the human generalization function
In our paper, we model the human generalization function using BERT. 

## Install dependencies
To install the dependencies, run:
```bash
pip install -r requirements.txt
```


### Training/loading the BERT model
To train the model from scratch, run:
```bash
python train.py
```
This will train a model and save it to `ckpts/bert.pth`.

Alternatively, if you don't want to train the model, you can load our model from the [Huggingface Hub](https://huggingface.co/keyonvafa/human-generalization-bert/blob/main/bert.pth). First create a `ckpts` directory, then visit [the link](https://huggingface.co/keyonvafa/human-generalization-bert/blob/main/bert.pth) to download the model, saving `bert.pth` in `ckpts`. (Note that you'll need to manually download the model from the website rather than use Git, which stores a pointer to the model rather than the model itself.)

### Evaluating predictions
To evaluate the model on the test set, run:
```bash
python evaluate_belief_change_models.py
``` 
This will:
- Record the NLL and AUC for the BERT model
- Record the NLL and AUC for two baselines
- Use the BERT model to form predictions of belief changes on a data sample used to evaluate LLM alignment. 

### Evaluating LLM alignment
To evaluate the alignment of the LLM with human generalizations, make sure you've used the BERT model to form predictions on the data sample by running `python evaluate_belief_change_models.py`. The results will be saved to `analysis-data/correct_sample_with_pred.csv` and `analysis-data/incorrect_sample_with_pred.csv`. 

The folder `llm-responses` contains the responses of the LLM to the questions in the data sample. To re-generate these or to create responses to new questions, you can uncomment code in `evaluate_llm_alignment.py` and provide your Together AI/OpenAI API keys at the top of `utils.py`.

To evaluate the alignment of GPT-4 with human generalizations, run:
```bash
python evaluate_llm_alignment.py --llm gpt-4
```
The other possible models to evaluate are `alpaca-7b`, `llama-2-7b-chat`, `llama-2-13b-chat`, `llama-2-70b-chat`, `StripedHyena-Nous-7B`, `mistralai/Mistral-7B-Instruct-v0.2`, `gpt-3.5-turbo`, and `gpt-4`.

### Analyzing qualitative examples
To analyze qualitative examples of human generalizations, first run:
```bash
python predict_belief_changes.py
```
This will generate 10000 random samples of pairs of questions and evaluate the BERT model's predictions. 

Then, the notebook `qualitative_analysis.ipynb` can be used to analyze the qualitative examples.