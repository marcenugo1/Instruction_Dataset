import pandas as pd
import random
import math
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")

input_file = '/workspace/storage/experiment_datasets_finetune/MVD_35K_prompt.csv'
output_file = '/workspace/storage/experiment_datasets_finetune/MVD_alpaca.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Load 15K records from the dataset and add them to the DataFrame
num_records_to_add = 15000
additional_records = []

for i in range(num_records_to_add):
    random_index = random.randint(0, len(dataset['train']) - 1)
    record = dataset['train'][random_index]
    additional_records.append({'instruction': record['instruction'], 'comments': record['output'], 'prompt': record['text']})

df_additional = pd.DataFrame(additional_records)

df = pd.concat([df, df_additional], ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True)

# Write the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)
