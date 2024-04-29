from datasets import load_dataset
from datasets import DatasetDict

# Load the original dataset
dataset = load_dataset("csv", data_files="/workspace/storage/experiment_datasets_finetune/MVD_alpaca.csv", split = "train").train_test_split(test_size=.1, train_size=.9)

# Create a DatasetDict to organize our new splits
new_splits = DatasetDict({
    'train': dataset["train"],
    'test': dataset["test"]
})

# Save the new splits to disk for later use
new_splits.save_to_disk('/workspace/storage/finetune/MVD_alpaca_split')
