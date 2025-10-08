import nltk
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem import WordNetLemmatizer
import pickle

# --- NLTK Downloads and Setup (Robust) ---
# Note: Using generic Exception to avoid DownloadError import issues
try:
    nltk.data.find('tokenizers/punkt')
except Exception: 
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except Exception: 
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except Exception: 
    nltk.download('omw-1.4')
# Additional check for the notorious punkt_tab if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except Exception:
    nltk.download('punkt_tab') 

# --- Helper Functions (Ensures consistency) ---
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, words):
    """Creates the Bag of Words vector for the given sentence."""
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

# --- 1. Data Loading and Preparation ---
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

try:
    with open('intents.json') as data_file:
        intents = json.loads(data_file.read())
except FileNotFoundError:
    print("Error: 'intents.json' file not found.")
    exit()

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Create training arrays (X_train and y_train)
X_train = []
y_train = []
for pattern_sentence, tag in documents:
    bag = bag_of_words(pattern_sentence, words)
    X_train.append(bag)
    label = classes.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# --- 2. PyTorch Model Definition and Training ---

# Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.001
INPUT_SIZE = len(X_train[0]) # FINAL INPUT SIZE
HIDDEN_SIZE = 128
OUTPUT_SIZE = len(classes)

print(f"DEBUG: FINAL VOCABULARY SIZE USED FOR TRAINING: {INPUT_SIZE}")

class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.x_data = torch.from_numpy(X_data)
        self.y_data = torch.from_numpy(y_data).long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

train_loader = DataLoader(dataset=ChatDataset(X_train, y_train), 
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
print("Starting PyTorch model training...")
for epoch in range(NUM_EPOCHS):
    for (words_batch, labels_batch) in train_loader:
        outputs = model(words_batch)
        loss = criterion(outputs, labels_batch)
        optimizer.zero_grad() 
        loss.backward()       
        optimizer.step()      
        
    if (epoch+1) % 50 == 0:
        print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# --- 3. Save Assets ---
# 3.1 Save PyTorch Model State (.pth)
MODEL_FILE = "chatbot_model_pytorch.pth"
data_torch = {
"model_state": model.state_dict(),
"input_size": INPUT_SIZE, # Explicitly saved input size
"hidden_size": HIDDEN_SIZE,
"output_size": OUTPUT_SIZE,
}
torch.save(data_torch, MODEL_FILE)

# 3.2 Save essential non-model data (words and classes) using pickle
DATA_FILE = "training_data_pytorch.pkl"
pickle.dump({'words': words, 'classes': classes}, open(DATA_FILE, 'wb'))

print(f"Model saved to {MODEL_FILE} and data saved to {DATA_FILE}")