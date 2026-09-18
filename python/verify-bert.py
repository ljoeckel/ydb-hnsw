from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer and model
# 'bert-base-uncased' is a standard English BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example text
text = "Hello, my dog is cute"

# Tokenize input and convert to PyTorch tensors
inputs = tokenizer(text, return_tensors="pt")

# Run the model (no_grad is used for inference to save memory)
with torch.no_grad():
    outputs = model(**inputs)

# The output contains the hidden states of the last layer
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
