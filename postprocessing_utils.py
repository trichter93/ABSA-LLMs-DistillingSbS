from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, ClassVar, List, IO, Tuple, Union
import json
import pandas as pd

def escape_single_quotes(input_string):
    return input_string.replace("'", "\\'")

def inference_on_gpu(model, tokenized_data, device, generated_dicts={}, batch_start=0):
    batch_size = 64  # You can adjust this batch size based on your GPU memory
    num_batches = (len(tokenized_data) + batch_size - 1) // batch_size
    for batch_index in tqdm(range(batch_start, num_batches)):
        print(f"Processing batch {batch_index + 1}/{num_batches}")
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(tokenized_data))
        input_ids_batch = {'input_ids': torch.tensor(tokenized_data[start_index:end_index]['input_ids']).to(device), 'attention_mask': torch.tensor(tokenized_data[start_index:end_index]['attention_mask']).to(device)}
        with torch.no_grad():
            outputs = model.generate(**input_ids_batch, max_length=MAX_LENGTH, exponential_decay_length_penalty = (100,2.0))
        for i, out in enumerate(outputs):
            generated_dicts[tokenized_data[start_index + i]['userReviewId']] = out.tolist()
    return generated_dicts

class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
            (re.compile(r',(\s*])'), r'\1'),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)

def generated_tensors_to_list(generated_dicts):
  list_dict = {k : v.tolist() for k,v in generated_dicts.items()}
  return list_dict # redundant

def clean_label(label_string):
    clean = label_string.replace("<pad>", "").replace("</s>", "").strip()
    return clean

def add_curly_braces(label_string):
  clean = label_string
  if not clean.startswith("{"):
      clean = "{" + clean
  if not clean.endswith("}"):
      clean = clean + "}"
  return clean

def generated_strings_to_json(generated_dicts : Dict[str, str], label_dicts = {}, failed=[]):
  for user_review_id in tqdm(generated_dicts):
    try:
      temp = tokenizer.decode(generated_dicts[user_review_id])
      cleaned = clean_label(temp)
      if not cleaned.endswith('"'):
            cleaned = cleaned[0:temp.rindex(',')]
      cleaned_with_curly = add_curly_braces(cleaned)
      label_dicts[user_review_id] = json.loads(cleaned_with_curly, cls=LazyDecoder)
    except ValueError as e:
      if str(e).startswith("Expecting ':' delimiter") or str(e).startswith("Unterminated string starting at"):
          cleaned = cleaned[0:cleaned.rindex(',')]
          cleaned_with_curly = add_curly_braces(cleaned)
          try:
            label_dicts[user_review_id] = json.loads(cleaned_with_curly, cls=LazyDecoder)
          except ValueError as e1:
            failed.append(user_review_id)
  return label_dicts, failed

def reduce_failures(failed, label_dicts, generated_dicts_with_lists): #old
  new_failed=[]
  for user_review_id in tqdm(failed):
    try:
      temp = tokenizer.decode(generated_dicts_with_lists[user_review_id])
      cleaned = clean_label(temp)
      if not cleaned.endswith('"'):
          cleaned = cleaned[0:cleaned.rindex(',')]
      cleaned_with_curly = add_curly_braces(cleaned)
      label_dicts[user_review_id] = json.loads(cleaned_with_curly, cls=LazyDecoder)
    except ValueError as e:
      new_failed.append(user_review_id)

  newer_failed=[]
  for user_review_id in new_failed:
    try:
      temp = tokenizer.decode(generated_dicts_with_lists[user_review_id])
      cleaned = clean_label(temp)
      if not cleaned.endswith('"'):
          cleaned = cleaned[0:cleaned.rindex(',')]
      cleaned_with_curly = add_curly_braces(cleaned)
      label_dicts[user_review_id] = json.loads(cleaned_with_curly, cls=LazyDecoder)
    except ValueError as e:
        newer_failed.append(user_review_id)
    return newer_failed