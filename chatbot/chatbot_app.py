import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import json
import random
import torch
import torch.nn as nn

# --- Model Definition (Must be identical to the trainer's model) ---
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

# --- Load Assets ---
try:
    # Load model and metadata
    FILE = "chatbot_model_pytorch.pth"
    data_torch = torch.load(FILE)

    INPUT_SIZE = data_torch["input_size"]
    HIDDEN_SIZE = data_torch["hidden_size"]
    OUTPUT_SIZE = data_torch["output_size"]
    MODEL_STATE = data_torch["model_state"]
    
    # Load words and classes (vocabulary and intent names)
    DATA_FILE = "training_data_pytorch.pkl"
    data_pkl = pickle.load(open(DATA_FILE, 'rb'))
    words = data_pkl['words']
    classes = data_pkl['classes']

    # Load intents (for fetching responses)
    with open('intents.json') as f:
        intents = json.loads(f.read())

except FileNotFoundError:
    print("Error: Required files (model/data) not found. Please run chatbot_trainer.py first.")
    exit()

# Instantiate model and load state
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(MODEL_STATE)
model.eval() # Set model to evaluation mode

# Initialize Lemmatizer (needed for cleaning user input)
lemmatizer = WordNetLemmatizer()

# --- Prediction Functions (Same logic as trainer) ---

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, words):
    # This function must be IDENTICAL to the one used for training
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1.0
    return bag

def chatbot_response(msg):
    # 1. Create BoW for user message
    sentence = tokenize(msg)
    X = bag_of_words(sentence, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # 2. Predict intent using the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]

    # 3. Check probability (optional, but good for confidence)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() < 0.75: # Confidence threshold
        return "I'm not sure I understand. Can you rephrase that?"

    # 4. Retrieve a random response
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])
    
    # Fallback
    return "I am unable to process that request right now."

# --- Main Chat Loop ---
print("--- PyTorch Chatbot is ready! (Type 'quit' to exit) ---")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        # Use the model to give a goodbye response
        print(f"Bot: {chatbot_response('bye')}")
        break
    
    response = chatbot_response(user_input)
    print("Bot:", response)