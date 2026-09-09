import json
from datasets import load_dataset

print("Tapping into the Hugging Face data stream...")
dataset = load_dataset("ibomohsin/gagle", split="train", streaming=True)

output_file = "fractal_scores.jsonl" 
count = 0

with open(output_file, "w") as f:
    for entry in dataset:
        domain = entry.get('dataset_name')
        try:
            # Grab the raw list of scores
            raw_llm = entry['scores']['gemma-2b']['llm_text']
            raw_human = entry['scores']['gemma-2b']['original_text']
            
            # Safety check: if it randomly hands us a nested list, flatten it
            llm_scores = raw_llm[0] if isinstance(raw_llm[0], list) else raw_llm
            human_scores = raw_human[0] if isinstance(raw_human[0], list) else raw_human

            # 1. Pluck the AI scores
            llm_result = {
                "id": str(entry.get('_id')),
                "domain": domain,
                "model_used": entry.get('model', 'Unknown_LLM'),
                "scores": llm_scores[64:464] if len(llm_scores) >= 464 else llm_scores,
                "prompt type": entry.get('prompt_type'),
                "temp": entry.get('temperature')
            }
            f.write(json.dumps(llm_result) + "\n")
            
            # 2. Pluck the Human scores
            human_result = {
                "id": str(entry.get('dataset_id')) + "_human",
                "domain": domain,
                "model_used": "Human",
                "scores": human_scores[64:464] if len(human_scores) >= 464 else human_scores,
                "prompt type": "continue",
                "temp": "Human"
            }
            f.write(json.dumps(human_result) + "\n")
            
            count += 2
            if count % 2000 == 0:
                print(f"Extracted {count} rows...")
                
            if count >= 30000: 
                break
        except (KeyError, IndexError, TypeError):
            continue

print(f"Done! Saved {count} rows directly to {output_file}.")