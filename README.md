# machine-learning-Brain-tumor-detection


Machine Learning Model for Brain Tumor Detection

This project uses classical machine learning algorithms to detect brain tumors from MRI images. It includes feature extraction, model training, and an API (app.py) to connect the model with a frontend interface.


Setup Instructions

1. Create Python Environment

python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate

2. Install Dependencies

pip install -r requirements.txt




Dataset

You need to download the Brain Tumor MRI dataset from Kaggle:

Dataset Link: Brain Tumor Classification (MRI)

After downloading, extract and place the dataset inside the project root folder:

ML-model-for-Brain-Tumor-detection/Dataset/



Run the Project

Step 1: Feature Extraction

Extract features from the MRI images and save them into a CSV file:

python feature_extraction.py

This will generate a CSV file with all the extracted features used for training.

Step 2: Train the Model

python train_model.py

You can change the ML algorithm used (e.g., SVM, Random Forest, etc.) in the script.

Step 3: Connect to Frontend

Start the API server to allow frontend integration:

python app.py

This will launch a REST API that listens for image input from the frontend and returns the tumor prediction result.


Technologies Used
	•	Python
	•	Scikit-learn
	•	NumPy, Pandas
	•	Flask or FastAPI (for app.py)
	•	OpenCV (for image preprocessing)
	•	Matplotlib, Seaborn (for visualization)

⸻

Notes
	•	Ensure all required packages are installed, especially for image processing and ML.
	•	The project is modular: feature extraction, model training, and API are all in separate scripts.
	•	This is a backend-only ML pipeline; use with a compatible frontend for full interaction.

