#  Emotion Detection App

A smart AI-powered application that detects emotions from **text**, **facial expressions**, and **voice**, and alerts HR in case of negative emotions like sadness, anger, or fear. Designed as part of the Zidio AI-Powered Task Optimizer project.

---

##  Features

- Text-based emotion detection using NLP and machine learning.
- Facial emotion detection using webcam and DeepFace.
- Voice emotion detection using MFCC and machine learning.
- Automatic HR email alert for negative emotions.
- Friendly UI built with Streamlit.
- Wellness suggestions and motivational quotes.

---

##  Technologies Used

- Python
- Streamlit
- OpenCV, DeepFace
- Librosa, sounddevice
- scikit-learn
- Joblib, NLTK
- Gmail SMTP for email alerts

---

##  Project Structure
emotion-detection-app/
│
├── app.py                        # Streamlit main app
├── models/
│   └── train_voice_model.py     # Voice emotion model training
├── scripts/
│   └── email_alerts.py          # HR email alert system
├── voice_data/                  # Voice dataset (excluded in GitHub)
├── screenshots/                 # Project screenshots for documentation
├── model.pkl                    # Trained model for text emotion detection
├── vectorizer.pkl               # TF-IDF vectorizer for text
├── voice_emotion_model.pkl      # Trained voice emotion model
├── requirements.txt             # All dependencies
├── README.md                    # Project documentation
└── .streamlit/                  # Streamlit config
##  Screenshots

![Text Emotion Detection](screenshots/text_emotion.png)  
![Voice Emotion Detection](screenshots/voice_emotion.png)  
![Facial Emotion Detection](screenshots/face_emotion.png)
---

##  How to Run

1. Clone the repo:https://github.com/Jyothsnaguthula01/emotion-detection.git  
2. Navigate to the project folder:cd emotion-detection
3. Create virtual environment and activate:python -m venv venv venv\Scripts\activate
4. Install dependencies:pip install -r requirements.txt
5. Run the app:streamlit run app.py

## Author

**Jyothsna Guthula**  
 jyothsnaguthula01@gmail.com  
[LinkedIn](https://www.linkedin.com/in/jyothsnaguthula)  
[GitHub](https://github.com/Jyothsnaguthula01)


