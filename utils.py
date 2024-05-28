import together
from openai import OpenAI
import pdb

# COMPLETE THE FOLLOWING
# together.api_key = XXX
# client = OpenAI(api_key=XXX)
# client = OpenAI()

def get_response_type(task):
  if task in ['object_counting', 'multistep_arithmetic_two']:
    response_type = "numeric"  
  elif task in ['causal_judgement', 'navigate', 'sports_understanding', 'web_of_lies']:
    response_type = "yes_no"
  elif task in ['formal_fallacies']:
    response_type = 'valid_invalid'
  else:
    response_type = "multiple_choice"
  return response_type

def get_valid_responses(response_type, task):
  if response_type == "multiple_choice":
    if task in ['hyperbaton', 'snarks']:
      valid_responses = ['A', 'B']
    elif task in ['logical_deduction_three_objects', 'disambiguation_qa', 
                  'tracking_shuffled_objects_three_objects']:
      valid_responses = ['A', 'B', 'C']
    elif task in ['mmlu', 'temporal_sequences']:
      valid_responses = ['A', 'B', 'C', 'D']
    elif task in ['logical_deduction_five_objects', 'penguins_in_a_table', 
                  'movie_recommendation', 
                  'tracking_shuffled_objects_five_objects']:
      valid_responses = ['A', 'B', 'C', 'D', 'E']
    elif task in ['logical_deduction_seven_objects', 
                  'tracking_shuffled_objects_seven_objects']:
      valid_responses = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    elif task in ['date_understanding', 'salient_translation_error_detection']:
      valid_responses = ['A', 'B', 'C', 'D', 'E', 'F']
    elif task in ['geometric_shapes', 'ruin_names']:
      valid_responses = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    elif task in ['reasoning_about_colored_objects']:
      valid_responses = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                         'L', 'M', 'N', 'O', 'P', 'Q', 'R']
    prompt_end = " Answer with a single multiple choice response and no words, i.e. {}. Answer:".format(" ".join([f'({x})' for x in valid_responses]))
  elif response_type == "yes_no":
    prompt_end = " Answer with a single word, either 'Yes' or 'No'. The answer is: '"
    valid_responses = ['yes', 'no']
  elif response_type == "valid_invalid":
    prompt_end = " Answer with a single word, either 'valid' or 'invalid'. The answer is: '"
    valid_responses = ['valid', 'invalid']
  elif response_type == "numeric":
    prompt_end = " Answer with a single number, no words. Answer:"
    valid_responses = None
  return valid_responses, prompt_end

def get_task(subject):
  if subject.split("_")[0] == 'mmlu':
    return 'mmlu'
  elif subject.split("_")[0] == 'bbh':
    return "_".join(subject.split("_")[1:])
  else:
    raise ValueError(f"Unknown subject: {subject}")
  
def get_answer(prompt, llm):
  if 'gpt' in llm:
    completion = client.chat.completions.create(
      model=llm,
      temperature=0.0,
      max_tokens=5,
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content.strip()
  else:
    if '/' not in llm:
      llm = f"togethercomputer/{llm}"
    output = together.Complete.create(
            prompt=prompt, 
            model=llm,
            max_tokens=5,  # Adjust as needed
            temperature=0.0,
            top_k=60,
            repetition_penalty=1,
            stop=[],
            top_p=1.0,
            # logprobs=3, # NOTE: stopped working for some reason
    )
    return output['output']['choices'][0]['text'].strip()

def check_answer(response, true_target, response_type, valid_responses):
  if response_type == 'multiple_choice':
    clean_response = response.replace("$\boxed{\textbf ", "").replace(
      "This ", "").replace("information", "").replace(
        "$\boxed{\textbf", "").replace('$\\boxed{\\textbf', '').replace(
          '{', "").replace('}', "").replace('\n', ' ').replace(
            '(', '').replace(')', '').replace(",", "").replace(".", "").strip()
    try:
      clean_response = clean_response.split(" ")[0].replace(
        '(', '').replace(')', '').replace(",", "").strip()
    except:
      pdb.set_trace()
    print(clean_response, true_target, 1 if clean_response == true_target else 0)
    if clean_response in valid_responses or 'cannot be answered' in response or 'does not' in response or 'can vary':
      return 1 if clean_response == true_target else 0
    else:
      return 0
  elif response_type == 'numeric':
    response = ''.join([x for x in response if x.isnumeric()])
    try:
      response = int(float(response))
      true_target = int(float(true_target))
    except:
      response = None
    print(response, true_target, 1 if response == true_target else 0)
    return 1 if response == true_target else 0
  elif response_type == 'yes_no':
    edited_response = response.split(" ")[0].replace('\n', '').replace(
      '(', '').replace(')', '').replace("'", "").replace(".", "").strip().lower()
    if not edited_response in valid_responses:
      if edited_response[:2] == 'no':
        edited_response = 'no'
      elif edited_response[:3] == 'yes':
        edited_response = 'yes'
      else:
        raise ValueError(f"Invalid response: {edited_response}")
    print(edited_response, 
          true_target.lower(), 
          1 if edited_response == true_target.lower() else 0)
    return 1 if edited_response == true_target.lower() else 0
  elif response_type == 'valid_invalid':
    response = response.split(" ")[0].replace('\n', '').replace(
      '(', '').replace(')', '').replace("'", "").replace(".", "").strip().lower()
    response = 'invalid' if response.startswith('invalid') else response
    response = 'valid' if response.startswith('valid') else response
    print(response, true_target.lower(), 1 if response == true_target.lower() else 0)
    if not response in valid_responses:
      raise ValueError(f"Invalid response: {response}")
    return 1 if response == true_target.lower() else 0