import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class MemeDataset(Dataset):
    def __init__(self, data_file, tokenizer, mlb=None, max_len=512, include_labels=True):
        self.include_labels = include_labels
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlb = mlb
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()
        
        if self.include_labels:
            if self.mlb is not None:
                labels = self.mlb.transform([item['labels']])[0]
                labels = torch.tensor(labels, dtype=torch.float)
            else:
                raise ValueError("MultiLabelBinarizer is not initialized but include_labels is set to True.")
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load and prepare the training data
with open("D://Documents//MS//NLP//Project//Dataset//annotations//subtask1//train.json", 'r', encoding='utf-8') as f:
    train_data = json.load(f)
mlb = MultiLabelBinarizer()
mlb.fit([item['labels'] for item in train_data])
train_dataset = MemeDataset("D://Documents//MS//NLP//Project//Dataset//annotations//subtask1//train.json", tokenizer, mlb)

# Load and prepare the validation data
with open("D://Documents//MS//NLP//Project//Dataset//annotations//subtask1//validation.json", 'r', encoding='utf-8') as f:
    validation_data = json.load(f)
validation_dataset = MemeDataset("D://Documents//MS//NLP//Project//Dataset//annotations//subtask1//validation.json", tokenizer, mlb)

# Define the model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    problem_type="multi_label_classification",
    num_labels=len(mlb.classes_)
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',  # Changed to 'steps' for more frequent evaluations
    eval_steps=500,               # Evaluate the model every 500 steps
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,  # Can be larger if you have enough memory
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

model.to(device)

# Train the model
trainer.train()

# Save the model and MultiLabelBinarizer for later use
model.save_pretrained('./trained_model')
torch.save(mlb, './mlb_transformer.pkl')

# Predict on the unlabeled dev set
dev_dataset = MemeDataset('path/to/dev_unlabeled.json', tokenizer, mlb, include_labels=False)
predictions = trainer.predict(dev_dataset)
predicted_labels = mlb.inverse_transform(torch.sigmoid(torch.tensor(predictions.predictions)).numpy() > 0.5)

# Generate submission file
submission_data = [{'id': item['id'], 'labels': list(labels)} for item, labels in zip(dev_dataset.data, predicted_labels)]
with open('submission.json', 'w', encoding='utf-8') as f:
    json.dump(submission_data, f, ensure_ascii=False, indent=4)
