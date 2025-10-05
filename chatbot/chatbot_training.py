import nltk
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem import WordNetLemmatizer
from nltk.downloader import DownloadError 
import pickle # Used to save words and classes

# --- NLTK Downloads (Initial Setup) ---
try:
    nltk.data.find('tokenizers/punkt')
except DownloadError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/wordnet')
except DownloadError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('corpora/omw-1.4')
except DownloadError:
    nltk.download('omw-1.4')

# --- 1. Data Loading and Preparation (BoW Vectorization) ---
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Load intents (Assumes intents.json is in the same directory)
try:
    with open('intents.json') as data_file:
        intents = json.loads(data_file.read())
except FileNotFoundError:
    print("Error: 'intents.json' file not found. Please create it first.")
    exit()

# Tokenize and build vocabulary/classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Function to create BoW for a sentence
def bag_of_words(tokenized_sentence, words):
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

# Create training arrays
X_train = [] # Features: Bag of Words vectors
y_train = [] # Labels: Intent Indices

for pattern_sentence, tag in documents:
    bag = bag_of_words(pattern_sentence, words)
    X_train.append(bag)

    label = classes.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# ... (Part 1: Data Loading and Preparation)

# Load intents (Assumes intents.json is in the same directory)
try:
    with open('intents.json') as data_file:
        intents = json.loads(data_file.read())
except FileNotFoundError:
    print("Error: 'intents.json' file not found. Please create it first.")
    exit()

# Tokenize and build vocabulary/classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag'])) # <--- This is where line 169 might be
        
    # Ensure this check is SEPARATE from the above line
    if intent['tag'] not in classes: 
        classes.append(intent['tag'])
# --- 2. PyTorch Setup and Model Definition ---

# Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.001
INPUT_SIZE = len(X_train[0])
HIDDEN_SIZE = 128
OUTPUT_SIZE = len(classes)

# Custom PyTorch Dataset
class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        # Convert numpy arrays to torch tensors
        self.x_data = torch.from_numpy(X_data)
        self.y_data = torch.from_numpy(y_data).long() # Labels must be Long type

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True, 
                          num_workers=0) 

# Define the Neural Network Model
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

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. Training Loop ---
print("Starting PyTorch model training...")
for epoch in range(NUM_EPOCHS):
    for (words_batch, labels_batch) in train_loader:
        # Forward pass
        outputs = model(words_batch)
        loss = criterion(outputs, labels_batch)
        
        # Backward and optimize
        optimizer.zero_grad() 
        loss.backward()       
        optimizer.step()      
        
    if (epoch+1) % 50 == 0:
        print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')
print("PyTorch training complete.")

# --- 4. Save Assets ---

# 4.1 Save PyTorch Model State (.pth)
MODEL_FILE = "chatbot_model_pytorch.pth"
data_torch = {
"model_state": model.state_dict(),
"input_size": INPUT_SIZE,
"hidden_size": HIDDEN_SIZE,
"output_size": OUTPUT_SIZE,
}
torch.save(data_torch, MODEL_FILE)

# 4.2 Save essential non-model data (words and classes) using pickle
DATA_FILE = "training_data_pytorch.pkl"
pickle.dump({'words': words, 'classes': classes}, open(DATA_FILE, 'wb'))

print(f"Model saved to {MODEL_FILE} and data saved to {DATA_FILE}")