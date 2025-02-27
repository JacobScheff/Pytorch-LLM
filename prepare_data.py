import pandas as pd

df = pd.read_parquet('wiki_data.parquet')
# Write the text column to a json file. the output should look like this: ["this is a sentence", "this is another sentence", ...]
text = df['text']
output = []
for i in range(min(len(text), 20)):
    out = text[i].split('\n')
    for j in range(len(out)):
        if len(out[j].split(" ")) > 10: # Only include entries with more than 10 words
            output.append(out[j])
            break # Only include one entry per wikipedia page to not train too much on one subject

print(output)