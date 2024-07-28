# train and save the PEFT model to local files


# Import necessary libraries
import os
import torch
from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments, GPT2Tokenizer, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load the dataset without caching
print("Loading dataset...")
dataset = load_dataset('yelp_polarity', split='train[:1000]', cache_dir=None)

# Load the pre-trained model and tokenizer
print("Loading model and tokenizer...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # Adjust the embedding size to the next multiple of 8
    new_embedding_size = ((len(tokenizer) + 7) // 8) * 8
    model.resize_token_embeddings(new_embedding_size)

# Ensure the padding token is set in the model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and test
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Create a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

print("Creating custom datasets...")
train_encodings = {key: train_dataset[key] for key in ['input_ids', 'attention_mask']}
train_labels = train_dataset['label']
train_dataset = CustomDataset(train_encodings, train_labels)

eval_encodings = {key: eval_dataset[key] for key in ['input_ids', 'attention_mask']}
eval_labels = eval_dataset['label']
eval_dataset = CustomDataset(eval_encodings, eval_labels)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Define training arguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="logs",
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="epoch",
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train the model
print("Training the model...")
trainer.train()

# Save the fine-tuned model
model_dir = "fine_tuned_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Save the fine-tuned model accuracy to a file
eval_results = trainer.evaluate()
train_accuracy = eval_results['eval_accuracy']
output_file = os.path.join("results", "train_accuracy.txt")

with open(output_file, "w") as f:
    f.write(f"Fine-tuned model accuracy: {train_accuracy:.4f}\n")

print(f"Fine-tuned accuracy result saved to {output_file}")
