#Lightweight Fine-Tuning Project
#TODO: In this cell, describe your choices for each of the following

#PEFT technique: LoRA
#Model:gpt2
#Evaluation approach:
#Fine-tuning dataset:The evaluation approach involves an assessment of the fine-tuned model's performance in comparison to the original pre-trained model. The primary metrics for this evaluation will be accuracy.
#Loading and Evaluating a Foundation Model
#TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.



# Import necessary libraries
import os
import torch
from transformers import AutoTokenizer, GPT2ForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset without caching and limit to 1000 samples
print("Loading dataset...")
dataset = load_dataset('yelp_polarity', split='train[:1000]', cache_dir=None)

# Split the limited dataset into train and test sets
print("Splitting dataset...")
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load the pre-trained model and tokenizer
print("Loading model and tokenizer...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Convert tokenized datasets to PyTorch tensors
train_input_ids = torch.tensor(tokenized_train_dataset["input_ids"])
train_attention_masks = torch.tensor(tokenized_train_dataset["attention_mask"])
train_labels = torch.tensor(tokenized_train_dataset["label"])

test_input_ids = torch.tensor(tokenized_test_dataset["input_ids"])
test_attention_masks = torch.tensor(tokenized_test_dataset["attention_mask"])
test_labels = torch.tensor(tokenized_test_dataset["label"])

# Create TensorDatasets
train_tensor_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_tensor_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# DataLoader to handle batching
print("Creating dataloaders...")
test_dataloader = DataLoader(test_tensor_dataset, batch_size=1, shuffle=False)

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    predictions, labels = [], []

    for i, batch in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}")
        input_ids, attention_mask, batch_labels = [x.to(model.device) for x in batch]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    return accuracy

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the pre-trained model
print("Evaluating the pre-trained model...")
pretrain_accuracy = evaluate_model(model, test_dataloader)
print(f"Pre-trained model accuracy: {pretrain_accuracy:.4f}")

# Save the pre-trained model accuracy to a file
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "pretrain_accuracy.txt")

with open(output_file, "w") as f:
    f.write(f"Pre-trained model accuracy: {pretrain_accuracy:.4f}\n")

print(f"Pre-trained accuracy result saved to {output_file}")

