# Performing Inference with a PEFT Model
# This code might take around 40 mins, 20 min each for the pre-train and train model


# Import necessary libraries
import os
import torch
from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments, GPT2Tokenizer, default_data_collator
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# Load the evaluation dataset
print("Loading evaluation dataset...")
eval_dataset = load_dataset('yelp_polarity', split='test[:1000]', cache_dir=None)

# Load the tokenizer and model, and set the padding token
print("Loading tokenizer and setting pad token...")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Ensure the padding token is set in the model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize the evaluation dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Create DataLoader for evaluation dataset
print("Creating dataloader...")
eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=8, collate_fn=default_data_collator)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Define evaluation arguments
training_args = TrainingArguments(
    output_dir="results",
    per_device_eval_batch_size=8,
    logging_dir="logs",
    logging_steps=10,
)

# Evaluate the pre-trained model
print("Loading pre-trained model...")
pretrained_model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
pretrained_model.config.pad_token_id = tokenizer.pad_token_id

trainer = Trainer(
    model=pretrained_model,
    args=training_args,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator
)

print("Evaluating the pre-trained model...")
pretrained_eval_results = trainer.evaluate()
pretrained_eval_accuracy = pretrained_eval_results['eval_accuracy']

# Save the evaluation results of the pre-trained model to a file
output_file_pretrained = os.path.join("results", "pretrained_eval_accuracy.txt")
with open(output_file_pretrained, "w") as f:
    f.write(f"Pre-trained model evaluation accuracy: {pretrained_eval_accuracy:.4f}\n")

print(f"Pre-trained model evaluation accuracy result saved to {output_file_pretrained}")

# Load the fine-tuned model and tokenizer
print("Loading fine-tuned model and tokenizer...")
fine_tuned_model_name = "fine_tuned_model"  # Ensure this matches the saved model directory name
fine_tuned_model = GPT2ForSequenceClassification.from_pretrained(fine_tuned_model_name)
fine_tuned_model.config.pad_token_id = tokenizer.pad_token_id

trainer = Trainer(
    model=fine_tuned_model,
    args=training_args,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator
)

# Evaluate the fine-tuned model
print("Evaluating the fine-tuned model...")
fine_tuned_eval_results = trainer.evaluate()
fine_tuned_eval_accuracy = fine_tuned_eval_results['eval_accuracy']

# Save the evaluation results of the fine-tuned model to a file
output_file_fine_tuned = os.path.join("results", "fine_tuned_eval_accuracy.txt")
with open(output_file_fine_tuned, "w") as f:
    f.write(f"Fine-tuned model evaluation accuracy: {fine_tuned_eval_accuracy:.4f}\n")

print(f"Fine-tuned model evaluation accuracy result saved to {output_file_fine_tuned}")

# Print a comparison of the results
print(f"Pre-trained model evaluation accuracy: {pretrained_eval_accuracy:.4f}")
print(f"Fine-tuned model evaluation accuracy: {fine_tuned_eval_accuracy:.4f}")
