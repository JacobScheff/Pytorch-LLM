import pandas as pd
import json
from tqdm import tqdm

df = pd.read_parquet('wiki_data.parquet')
# Write the text column to a json file. the output should look like this: ["this is a sentence", "this is another sentence", ...]
text = df['text']
output = []
for i in tqdm(range(len(text))):
    out = text[i].split('\n')
    for j in range(len(out)):
        if len(out[j].split(" ")) > 10: # Only include entries with more than 10 words
            output.append(out[j])
            break # Only include one entry per wikipedia page to not train too much on one subject

with open('wiki_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)