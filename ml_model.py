# backend/ml_model.py

import os
import sys
import time
import random
# import pickle # Uncomment if you save/load models with pickle
# from PIL import Image # Uncomment if you process images with Pillow
# import numpy as np # Uncomment if you use numpy for image processing or model inputs

# This import works when ml_model.py is imported as a module (e.g., by app.py)
# If running ml_model.py directly, the sys.path modification in __main__ will handle it.
try:
    from . import config
except ImportError:
    # Fallback for when ml_model.py is run directly and not as part of a package
    # This block is essential for direct execution.
    # We need to add the parent directory (Project) to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    import backend.config as config # Now this absolute import should work


class MLModel:
    def __init__(self, model_path, model_type="classical"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the ML model from the specified path.
        In a real application, you'd load your actual model here.
        e.g., self.model = pickle.load(open(self.model_path, 'rb'))
        or for TensorFlow/Keras: self.model = tf.keras.models.load_model(self.model_path)
        """
        # Check if the model file actually exists
        if not os.path.exists(self.model_path):
            print(f"WARNING: {self.model_type.capitalize()} model file not found at {self.model_path}. Using a dummy model.")
            self.model = True # Still set to True for simulation purposes
            return

        print(f"Loading {self.model_type} model from {self.model_path}...")
        try:
            # Simulate model loading time
            time.sleep(0.5)
            # --- REPLACE THIS WITH YOUR ACTUAL MODEL LOADING LOGIC ---
            # For example:
            # with open(self.model_path, 'rb') as f:
            #     self.model = pickle.load(f)
            self.model = True # Placeholder for a loaded model object
            # --- END OF REPLACE SECTION ---
            print(f"{self.model_type.capitalize()} model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load {self.model_type} model from {self.model_path}: {e}")
            self.model = False # Indicate failure
            print(f"WARNING: Using dummy model for {self.model_type} due to loading error.")


    def preprocess_image(self, image_file_stream):
        """
        Preprocesses the uploaded image for model input.
        Replace with your actual image preprocessing steps.
        This function will take the file stream from Flask's request.files['image'].
        """
        # IMPORTANT: Uncomment and implement your actual image processing here.
        # This will depend on what your ML/QML models expect as input.
        # Example using Pillow (ensure Pillow is installed: pip install Pillow):
        # try:
        #     img = Image.open(image_file_stream)
        #     img = img.convert('RGB') # Or 'L' for grayscale
        #     img = img.resize((224, 224)) # Example target size for a CNN
        #     img_array = np.array(img).astype('float32')
        #     # Normalize pixel values (e.g., to [0, 1] or mean/std normalization)
        #     img_array = img_array / 255.0
        #     # Add batch dimension if your model expects it (e.g., (1, H, W, C))
        #     img_array = np.expand_dims(img_array, axis=0)
        #     return img_array
        # except Exception as e:
        #     print(f"Error during image preprocessing: {e}")
        #     raise # Re-raise to be caught by the calling function

        # For now, just return a placeholder for simulation
        print(f"Simulating image preprocessing for {self.model_type} model.")
        return "Preprocessed Image Data Placeholder"

    def predict(self, preprocessed_data):
        """
        Makes a prediction using the loaded model.
        Replace with your actual model prediction logic using self.model.
        """
        if not self.model:
            print(f"ERROR: {self.model_type.capitalize()} model not loaded. Returning dummy prediction.")

        start_time = time.time()

        # --- REPLACE THIS WITH YOUR ACTUAL MODEL PREDICTION LOGIC ---
        # For example:
        # if self.model:
        #     prediction_result = self.model.predict(preprocessed_data)
        #     # Process prediction_result to get predicted_label and raw_output
        #     # Example: If prediction_result is a softmax output array
        #     # predicted_index = np.argmax(prediction_result)
        #     # possible_labels = ["No Tumor Detected", "Benign Tumor", "Malignant Tumor"]
        #     # predicted_label = possible_labels[predicted_index]
        #     # raw_output = prediction_result[0].tolist() # Assuming batch size 1
        # else:
        #     # Fallback for when model isn't loaded
        #     # ... (simulate as below)
        # --- END OF REPLACE SECTION ---

        # Simulate prediction time and result
        if self.model_type == "classical":
            time.sleep(random.uniform(1.0, 3.0))
            possible_labels = ["No Tumor Detected", "Benign Tumor", "Malignant Tumor"]
        else: # Quantum ML
            time.sleep(random.uniform(2.0, 5.0)) # QML often slower
            possible_labels = ["No Tumor Detected (QML)", "Benign Tumor (QML)", "Malignant Tumor (QML)"]

        predicted_label = random.choice(possible_labels)
        raw_output = [round(random.random(), 4) for _ in range(len(possible_labels))]
        sum_raw = sum(raw_output)
        if sum_raw > 0: # Normalize to sum to 1 if representing probabilities
            raw_output = [round(p / sum_raw, 4) for p in raw_output]
        else: # Handle case where all random numbers are 0, prevent division by zero
            raw_output = [1.0/len(possible_labels)] * len(possible_labels) # Assign equal probability

        timing = round(time.time() - start_time, 2)

        return {
            'predicted_label': predicted_label,
            'raw_output': raw_output,
            'timing': timing
        }

# --- Global model instances ---
classical_ml_model = None
quantum_ml_model = None

def load_all_models(config_obj):
    """
    Loads both classical and quantum models using the provided config.
    This function should be called once when the Flask app starts.
    """
    global classical_ml_model, quantum_ml_model
    if classical_ml_model is None:
        classical_ml_model = MLModel(config_obj.CLASSICAL_MODEL_PATH, "classical")
    if quantum_ml_model is None:
        quantum_ml_model = MLModel(config_obj.QUANTUM_MODEL_PATH, "quantum")
    print("All ML and QML models initialized.")

def get_classical_model():
    """Returns the loaded classical ML model instance."""
    return classical_ml_model

def get_quantum_model():
    """Returns the loaded quantum ML model instance."""
    return quantum_ml_model

# This block runs only when ml_model.py is executed directly for testing
if __name__ == '__main__':
    print("This is ml_model.py being executed directly for testing.")
    print("It defines model loading and prediction logic.")

    # The ImportError block above *should* have already added the project root.
    # We can re-check, but usually, it's done once.
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(current_dir, '..'))
    # if project_root not in sys.path:
    #     sys.path.append(project_root)

    print("Attempting to load models for standalone test...")
    load_all_models(config) # Use the new combined loader

    # Example of how to use the models if loaded
    if get_classical_model() and get_quantum_model():
        print("\nTesting classical model prediction:")
        # For actual use, you'd pass a file stream here
        dummy_preprocessed_data = "Dummy preprocessed image data"
        classical_result = get_classical_model().predict(dummy_preprocessed_data)
        print(f"Classical Prediction: {classical_result}")

        print("\nTesting quantum model prediction:")
        quantum_result = get_quantum_model().predict(dummy_preprocessed_data)
        print(f"Quantum Prediction: {quantum_result}")
    else:
        print("Models could not be loaded for standalone test. Ensure config and model paths are correct.")