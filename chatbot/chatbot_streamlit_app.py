import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# --- Helper Functions (From the trainer/app) ---
# Note: These helpers must be available globally to the Streamlit script
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, words):
    """Creates the BoW vector using the *loaded* vocabulary list."""
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1.0
    return bag

# --- Asset Loading (Uses Streamlit's cache for efficiency) ---

@st.cache_resource
def load_assets():
    """Loads model and data only once."""
    try:
        # 1. Load words and classes (vocabulary and intent names)
        DATA_FILE = "training_data_pytorch.pkl"
        data_pkl = pickle.load(open(DATA_FILE, 'rb'))
        words = data_pkl['words']
        classes = data_pkl['classes']

        # 2. Load model and metadata
        FILE = "chatbot_model_pytorch.pth"
        data_torch = torch.load(FILE)

        INPUT_SIZE = data_torch["input_size"]
        HIDDEN_SIZE = data_torch["hidden_size"]
        OUTPUT_SIZE = data_torch["output_size"]
        MODEL_STATE = data_torch["model_state"]
        
        # 3. Load intents (for fetching responses)
        with open('intents.json') as f:
            intents = json.loads(f.read())

        # Instantiate model and load state
        model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        model.load_state_dict(MODEL_STATE)
        model.eval() # Set model to evaluation mode
        
        return model, words, classes, intents

    except FileNotFoundError:
        st.error("Error: Required files (model/data) not found. Please run chatbot_trainer.py first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        st.stop()
        
# Load assets into the app
model, words, classes, intents = load_assets()

# --- Prediction Logic ---

def get_chatbot_response(msg):
    """The core prediction function."""
    
    # Check for quit/exit command
    if msg.lower() in ["quit", "exit", "bye"]:
        return random.choice(["Goodbye!", "Talk to you later."])

    # 1. Create BoW for user message
    sentence = tokenize(msg)
    X = bag_of_words(sentence, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float() 

    # 2. Predict intent using the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]

    # 3. Check probability (Confidence)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    # Low confidence threshold
    if prob.item() < 0.75: 
        return "I'm not sure I understand that. Can you rephrase?"

    # 4. Retrieve a random response
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])
    
    return "I am unable to process that request right now."


# --- 3. Streamlit Frontend UI ---

st.title("PyTorch Intent-Based Chatbot 🤖")
st.markdown("Ask questions about business hours, pricing, or contact information.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Say something to the bot..."):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.spinner('Thinking...'):
        response = get_chatbot_response(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})