# Audio_DeepFake_Detection
This ia a ML project to detect the difference between Real and Fake Audio using ML algorithm
# Project Overview

Audio deepfakes pose a significant threat to digital security by mimicking human speech with AI-generated voices. This project aims to develop a robust detection system using Support Vector Machine (SVM) and Convolutional Neural Networks (CNN) to distinguish between real and fake audio.

📌 Features

✅ Deepfake Detection: Classifies audio as FAKE or REAL

✅ Machine Learning Models: Implemented SVM & CNN for comparison

✅ Feature Engineering: Used MFCC (Mel-Frequency Cepstral Coefficients) for audio processing

✅ High Accuracy: Achieved 99.62% (SVM) and 99.19% (CNN)

✅ Dataset Used: Balanced dataset for training & evaluation

📂 Dataset

The dataset consists of extracted MFCC features from real and fake speech samples.

Source: Custom dataset based on deepfake audio samples.

Preprocessing: Extracted key features (MFCCs) for efficient classification.

🏗️ Model Implementation

🔹 Support Vector Machine (SVM)

Uses Radial Basis Function (RBF) Kernel for better classification.

Performs well on small datasets with lower computational cost.

Achieved 99.62% accuracy.

🔹 Convolutional Neural Network (CNN)

Converts MFCC features into a spectrogram-like structure for deep learning.

Uses 1D convolutional layers to detect unique patterns in fake vs. real speech.

Achieved 99.19% accuracy.

----------------------------------------------------Documentation & Analysis ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1️⃣ Implementation Process
🛠️ Challenges Encountered & Solutions
🔹 Challenge 1: Feature Selection & Preprocessing
Issue: Extracting relevant features from audio data is critical for accurate classification.

Solution: Used MFCC features, as they capture important speech characteristics while reducing dimensionality.

🔹 Challenge 2: Model Selection
Issue: Choosing between SVM and CNN for the task.

Solution: Implemented both models and compared their performance.

SVM performed slightly better (99.62%) and was computationally lighter.

CNN had similar accuracy (99.19%) but required more resources.

🔹 Challenge 3: Model Overfitting
Issue: Both models achieved very high accuracy (>99%), which could mean overfitting.

Solution: Used train-test split with stratification and cross-validation to ensure robustness.

🔹 Challenge 4: Real-World Generalization
Issue: Deepfake techniques evolve rapidly; models must generalize to unseen fake voices.

Solution: Used a balanced dataset, but further testing with real-world audio would be necessary.

2️⃣ Analysis
🔹 Why Choose SVM?
SVM worked exceptionally well on the extracted MFCC features.

Less computational cost than CNN, making it ideal for quick inference.

Higher accuracy (99.62%) compared to CNN (99.19%).

🔹 How the Model Works?
🔹 Support Vector Machine (SVM)
Creates a decision boundary (hyperplane) that separates fake and real audio.

Uses MFCC features as input and applies the Radial Basis Function (RBF) kernel to capture complex patterns.

🔹 CNN
Converts audio features (MFCC spectrograms) into image-like structures.

Uses 1D convolutional layers to detect unique deepfake patterns.

🔹 Performance Results
Metric	SVM	CNN
Accuracy	99.62%	99.19%
Precision	1.00	0.99
Recall	0.99-1.00	0.99-1.00
F1-Score	1.00	0.99
📌 Both models performed extremely well, but SVM was slightly more efficient.

🔹 Strengths & Weaknesses
Aspect	SVM	CNN
Strengths	Fast, works well on small datasets, high accuracy, interpretable	Good for large datasets, extracts deep feature patterns
Weaknesses	May struggle with raw waveforms, might not generalize to unseen fakes	Requires more data and GPU, complex to fine-tune
🔹 Suggestions for Future Improvements
Test on Real-World Deepfakes

Apply model to unseen AI-generated voices (e.g., TTS systems, cloned speech).

Use More Advanced Features

Instead of only MFCC, add Mel spectrograms, pitch variations, formants.

Try Hybrid Models

Combine SVM for feature extraction with a lightweight CNN for better generalization.

Deploy & Optimize for Speed

Convert to ONNX/TensorFlow Lite for real-time mobile/web applications.

3️⃣ Reflection Questions
1️⃣ What were the most significant challenges in implementing this model?
Feature Engineering: Choosing the right features (MFCC, spectrograms).

Model Overfitting: Achieving 99%+ accuracy raised concerns about real-world performance.

Dataset Balance: Ensuring an equal distribution of REAL vs. FAKE samples.

2️⃣ How might this approach perform in real-world conditions vs. research datasets?
✅ SVM may struggle with unseen deepfake methods.
✅ CNN could work better with more training data.
✅ Research datasets are often cleaner than real-world audio (background noise, accents, distortions).
✅ The model should be tested on live conversations, podcasts, and low-quality speech.

3️⃣ What additional data or resources would improve performance?
✔ More deepfake audio samples from different AI generators (e.g., ElevenLabs, Resemble AI).
✔ Noisy real-world recordings to test robustness.
✔ Larger, more diverse datasets (e.g., multilingual, different accents).
✔ Fine-tuning on additional deepfake datasets (ASVspoof, FakeAVCeleb, WaveFake).

4️⃣ How would you approach deploying this model in a production environment?
🚀 Steps to Deploy SVM Model:

Save the Model & Scaler

python
Copy
Edit
import joblib
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
Deploy as a Flask or FastAPI API

Send audio input → Extract MFCC features → Predict FAKE/REAL.

Optimize for Speed

Convert to ONNX or TensorFlow Lite for lightweight deployment.

Integrate with Real-time Applications

Add to web apps, mobile apps, or security software.

Monitor & Improve

Collect feedback, retrain on new deepfake samples.

--------------------------------------------------------------------------------END---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🤝 Contributing

Pull requests are welcome! Feel free to fork this repository and submit improvements.

📝 Contact

For any inquiries, feel free to reach out:📧 ayushswaroop84424@gmail.com

