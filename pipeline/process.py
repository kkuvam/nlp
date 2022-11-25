import os
import json
from pathlib import Path
from collections import defaultdict
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Initialize spaCy pipeline
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("spacytextblob")  # for sentiment

def process_text(data: dict) -> dict:
        try:
            data['entities'] = {
                'loc': [],
                'org': [],
                'person': []
            }
            doc = nlp(data['text'])
            for ent in doc.ents:
                entity_type = ent.label_.lower()
                entity_name = ent.text.lower().replace('\n', ' ').strip()
                if entity_type in ['loc', 'org', 'person']:
                    # Check we add this entity in data
                    if entity_name in data['entities'][entity_type]:
                        continue
                    else:
                        data['entities'][entity_type].append(entity_name)
                # Sentiment Analysis
                data['score'] = score = round(doc._.blob.polarity, 3)
                data['label'] = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at line {line_number}: {e}")
            return None


def process_file(input_path: str) -> None:
    """
    Process a JSON Lines file, extracting entities and sentiment from each line.
    Writes the processed data to a new file with '_processed' appended before the extension.
    """
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_processed.jsonl")
    with open(file_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line_number, line in enumerate(fin, start=1):
            try:
                data = json.loads(line)
                processed_data = process_text(data)
                if processed_data:
                    fout.write(json.dumps(processed_data) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {line_number}: {e}")
            except Exception as e:
                print(f"[{input_path.name}] Unexpected error at line {lineno}: {e}")
    

def process_directory(dir_path: str):
    p = Path(dir_path)
    jsonl_files = list(p.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {dir_path}")
        return
    for f in jsonl_files:
        print(f"Processing {f.name} ...")
        process_file(f)
    print("Done processing files.")


if __name__ == "__main__":
    # Input file path
    file_path = "28Jun22.jsonl"
    process_file(file_path)
