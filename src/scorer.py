#The purpose of this script is to act as the Observer. It iterates through a dataset of human and AI-generated text, uses a Large Language Model (GPT-2) to measure how "surprising" each word choice is, and outputs a sequence of numerical scores.
#calculating the log probablities of the next token in the sequence and using that as a measure of surprise for the model.
import torch 
import json
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id="google/gemma-2-2b"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id).to(device)
dataset= load_dataset("ibomohsin/gagle", streaming=True)
data_stream = dataset['train']

output_file = "fractal_scores.jsonl"
processed_ids = set()

# Read existing progress to prevent data loss on restart
if os.path.exists(output_file):
    print("Reading existing progress...")
    with open(output_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_ids.add(data['id'])
            except:
                pass
    print(f"Found {len(processed_ids)} articles already processed. Resuming...")

count = 0
#above line is used to split the dataset for easy loading and scoring

with open(output_file, "a") as f:
    for entry in data_stream:
        print(f"Available keys in this entry: {entry.keys()}")
        
        entry_id = entry.get('_id')
        if entry_id in processed_ids:
            continue
            
        text = entry.get('llm_text') 

        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        #Text is converted into integers (input_ids) that the model understands.
        input_ids = inputs["input_ids"]
        #Fractal analysis requires a consistent window of data. The script ignores any article shorter than 464 tokens to ensure there is enough data for the "Warmup" and "Analysis" windows.
        if input_ids.shape[1] < 464:
            continue
# till here i have created the inital pipeline and the engine for scoring
# next we will take on the scoring logic 
        with torch.no_grad():
            outputs = model(input_ids)
            logits=outputs.logits
            # Shift logits and labels to align prediction with the next token
            shift_logits=logits[..., :-1, :].contiguous()
            shift_labels=input_ids[..., 1:].contiguous()
            # calculating the log probs
            log_probs=F.log_softmax(shift_logits, dim=-1)
            target_log_probs=torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(2)
            surprise_scores = target_log_probs[0].tolist()
            # basic parsing and refining of input dataset ie ignore first 64 tokens as a warmup and clip size to 400
            #Language models often exhibit "unstable" or outlier probability behavior at the very beginning of a sequence because they lack context. By skipping the first 64 tokens, we allow the model's internal "memory" to stabilize.
            refined_scores = surprise_scores[64:464]
            # saving the scores in xlsx file
            result = {
                    "id": entry_id,
                    "domain": entry.get('dataset_name'), # Maps to 'dataset_name' in your output
                    "model_used": entry.get('model') if entry.get('model') else "Human",    # Maps to 'model' in your output
                    "scores": refined_scores,
                    "prompt type": entry.get('prompt_type'),
                    "temp": entry.get('temperature')
            }
            f.write(json.dumps(result) + "\n")
            f.flush()
            
            count +=1
            print(f"processed {count}")

print(f"finished with total count: {count}")
