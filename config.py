# backend/config.py

import os

# Base directory for the project (assuming this file is in 'backend/')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path where trained classical models are saved
# This points to backend/saved_models/ (or wherever you want them)
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True) # Ensure directory exists

CLASSICAL_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'classical_model.pkl')
QUANTUM_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'quantum_model.pkl')

# UPLOAD_FOLDER for temporary image storage
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)