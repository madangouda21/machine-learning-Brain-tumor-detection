# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import time
import sys

# Import your config and ml_model modules using robust import logic
try:
    from . import config
    from . import ml_model
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    try:
        import backend.config as config
        import backend.ml_model as ml_model
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import backend.config or backend.ml_model. Please check your project structure and Python path. Error: {e}")
        sys.exit(1)

# Initialize the Flask application
app = Flask(__name__)

# --- IMPORTANT: Configure CORS ---
CORS(app)

# Configure upload folder from config
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Global model instances (loaded once on app startup)
classical_ml_model = None
quantum_ml_model = None

# This function initializes models. It will be called explicitly.
def initialize_models():
    """
    Loads both classical and quantum ML models when the Flask app first starts.
    This function is now called explicitly in the main block or via a different Flask hook.
    """
    global classical_ml_model, quantum_ml_model
    try:
        print("Initializing models...") # Removed "on app startup" as hook changed
        ml_model.load_all_models(config)
        classical_ml_model = ml_model.get_classical_model()
        quantum_ml_model = ml_model.get_quantum_model()
        if classical_ml_model and quantum_ml_model:
            print("All models successfully initialized and ready for use.")
        else:
            print("WARNING: Some models failed to load properly. Predictions might be simulated.")
    except Exception as e:
        app.logger.error(f"Failed to initialize models: {e}")
        print(f"ERROR: Model initialization failed: {e}")


@app.route('/predict_classical', methods=['POST'])
def predict_classical():
    app.logger.info(f"Received POST request at /predict_classical from {request.remote_addr}")

    if 'image' not in request.files:
        app.logger.warning("No 'image' file part in the request.")
        return jsonify({'error': 'No image file provided in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        app.logger.warning("No selected file name received.")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        if not classical_ml_model or not classical_ml_model.model:
            app.logger.error("Classical ML model not loaded or initialized.")
            return jsonify({'error': 'Classical ML model not available.'}), 503 # Service Unavailable

        start_request_time = time.time()

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            app.logger.info(f"Classical ML: File saved temporarily to: {filepath}")

            with open(filepath, 'rb') as f_stream:
                preprocessed_img = classical_ml_model.preprocess_image(f_stream)

            prediction_result = classical_ml_model.predict(preprocessed_img)

            os.remove(filepath)

            total_timing = round(time.time() - start_request_time, 2)

            app.logger.info(f"Classical ML prediction complete. Label: {prediction_result['predicted_label']}, "
                            f"Inference Time: {prediction_result['timing']}s, Total Request Time: {total_timing}s")

            return jsonify({
                'predicted_label': prediction_result['predicted_label'],
                'raw_prediction_encoded': prediction_result['raw_output'],
                'message': 'Classical ML prediction successful',
                'timing': total_timing,
                'inference_timing': prediction_result['timing']
            }), 200

        except Exception as e:
            app.logger.error(f"Error during classical ML prediction: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Classical ML prediction failed: {str(e)}'}), 500
    
    app.logger.error("Unexpected error in /predict_classical route.")
    return jsonify({'error': 'An unexpected error occurred processing the file.'}), 500


@app.route('/predict_quantum', methods=['POST'])
def predict_quantum():
    app.logger.info(f"Received POST request at /predict_quantum from {request.remote_addr}")

    if 'image' not in request.files:
        app.logger.warning("No 'image' file part in the request for QML.")
        return jsonify({'error': 'No image file provided in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        app.logger.warning("No selected file name received for QML.")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        if not quantum_ml_model or not quantum_ml_model.model:
            app.logger.error("Quantum ML model not loaded or initialized.")
            return jsonify({'error': 'Quantum ML model not available.'}), 503

        start_request_time = time.time()

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            app.logger.info(f"QML: File saved temporarily to: {filepath}")

            with open(filepath, 'rb') as f_stream:
                preprocessed_img = quantum_ml_model.preprocess_image(f_stream)

            prediction_result = quantum_ml_model.predict(preprocessed_img)

            os.remove(filepath)

            total_timing = round(time.time() - start_request_time, 2)

            app.logger.info(f"Quantum ML prediction complete. Label: {prediction_result['predicted_label']}, "
                            f"Inference Time: {prediction_result['timing']}s, Total Request Time: {total_timing}s")

            return jsonify({
                'predicted_label': prediction_result['predicted_label'],
                'raw_qml_output': prediction_result['raw_output'],
                'message': 'Quantum ML prediction successful',
                'timing': total_timing,
                'inference_timing': prediction_result['timing']
            }), 200

        except Exception as e:
            app.logger.error(f"Error during quantum ML prediction: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Quantum ML prediction failed: {str(e)}'}), 500

    app.logger.error("Unexpected error in /predict_quantum route.")
    return jsonify({'error': 'An unexpected error occurred processing the file.'}), 500

if __name__ == '__main__':
    # This is how you run Flask applications in modern Flask versions.
    # We directly call initialize_models within the app context.
    with app.app_context():
        initialize_models()
    app.run(debug=True, port=5005, host='127.0.0.1')