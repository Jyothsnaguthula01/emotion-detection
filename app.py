import sounddevice as sd
import librosa
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from deepface import DeepFace
import streamlit as st
import cv2
import joblib
import re
import nltk
import csv
from datetime import datetime
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="Emotion Detector", layout="wide")

# Download stopwords for text cleaning
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Correct dummy user data structure
users = {
    "Jyothsna": {"name": "Jyothsna", "password": "pass123"},
    "Priyanka": {"name": "Priyanka", "password": "jyo123"},
    "meghana": {"name": "meaghana", "password": "sai123"}
}

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.name = ""

# Sidebar login
with st.sidebar:
    st.subheader("Employee Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Debugging information
        st.write(f"Entered Username: {username}, Entered Password: {password}")  # Debugging line
        
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.name = users[username]["name"]
            st.success(f"Logged in as {st.session_state.name}")
        else:
            st.error("Invalid credentials")

# Email HR on negative emotions
def send_email_to_hr(emotion, user_input):
    sender_email = "zidio.hr.alerts@gmail.com"
    sender_password = "echh ofar bbcx uvsm"  # Use environment variables for secure password handling
    receiver_email = "jyothsnaguthula@gmail.com"

    employee_name = st.session_state.get("name", "Unknown")

    subject = f"Alert: Negative Emotion Detected - {emotion}"
    body = f"""Dear HR Team,

A negative emotion ({emotion}) has been detected from employee: {employee_name}

Input Source:
"{user_input}"

Please consider reaching out to check on them.

Best regards,
Emotion Detection App"""

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()
        print("Email sent to HR")
    except Exception as e:
        print(f"Error sending email: {e}")

# Log mood to CSV
def log_mood(emotion, source):
    with open("mood_logs.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         st.session_state.get("name", "Unknown"),
                         source,
                         emotion])

# Load models
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Motivation and task suggestions
emotion_tasks = {
    "sadness": "Take a 10-minute mindfulness break or write in a journal.",
    "anger": "Go for a walk, do cardio, or try a breathing exercise.",
    "fear": "Watch a motivational video or talk to someone you trust.",
    "happy": "Start a new idea, help a teammate, or celebrate progress.",
    "neutral": "Organize your workspace or plan the rest of your day."
}

emotion_suggestions = {
    "sadness": {
        "quote": "Tough times never last, but tough people do.",
        "video": "https://www.youtube.com/watch?v=mgmVOuLgFB0"
    },
    "anger": {
        "quote": "For every minute you're angry, you lose 60 seconds of happiness.",
        "video": "https://www.youtube.com/watch?v=3bKuoH8CkFc"
    },
    "fear": {
        "quote": "Feel the fear and do it anyway.",
        "video": "https://www.youtube.com/watch?v=VEB8YzPZ9X8"
    },
    "happy": {
        "quote": "The purpose of our lives is to be happy!",
        "video": "https://www.youtube.com/watch?v=ZbZSe6N_BXs"
    },
    "neutral": {
        "quote": "Stay calm, stay focused, stay strong.",
        "video": "https://www.youtube.com/watch?v=2Lz0VOltZKA"
    }
}

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #f0932b;'> Emotion Detection App </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type how you feel or try the camera! We'll handle it with care </p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Text Emotion", "Face Emotion"])

if not st.session_state.logged_in:
    st.warning("Please log in from the sidebar to access emotion detection features.")
    st.stop()

# Text Detection
with tab1:
    user_input = st.text_input("Enter your sentence:")
    if st.button("Detect Text Emotion") and user_input.strip():
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"**Predicted Emotion:** {prediction}")
        log_mood(prediction, "text")

        if suggestion := emotion_suggestions.get(prediction):
            st.markdown("###  Motivation Corner")
            st.success(f"**Quote:** {suggestion['quote']}")
            st.markdown(f"[ Watch a Motivational Video]({suggestion['video']})")
        st.markdown(f"** Recommended Task:** {emotion_tasks.get(prediction)}")

        if prediction in ["sadness", "anger", "fear"]:
            with open("hr_notifications.txt", "a") as file:
                file.write(f"[TEXT] ALERT: {prediction.upper()} | Message: {user_input}\n")
                st.info("HR Email: zidio.hr.alerts@gmail.com")
                send_email_to_hr(prediction, user_input)

# Face Detection
with tab2:
    if st.button("Start Camera Emotion Detection"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected.")
                break
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                log_mood(emotion, "face")

                cv2.putText(frame, f"Emotion: {emotion}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                stframe.image(frame, channels="BGR", use_column_width=True)

                if suggestion := emotion_suggestions.get(emotion):
                    st.markdown("### 💬 Motivation Corner")
                    st.success(f"**Quote:** {suggestion['quote']}")
                    st.markdown(f"[📺 Watch a Motivational Video]({suggestion['video']})")
                st.markdown(f"**📝 Recommended Task:** {emotion_tasks.get(emotion)}")

                if emotion in ["sad", "angry", "fear"]:
                    with open("hr_notifications.txt", "a") as file:
                        file.write(f"[FACE] ALERT: {emotion.upper()} detected from camera\n")
            except Exception as e:
                stframe.warning("No face detected...")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Voice Emotion Detection
st.markdown("### Voice Emotion Detection")
if st.button("Record Voice"):
    duration = 5
    fs = 22050
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("Recording complete!")

    mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=fs, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)

    try:
        voice_model = joblib.load("voice_emotion_model.pkl")
        voice_emotion = voice_model.predict(mfccs_scaled)[0]
        st.success(f"Voice Emotion: {voice_emotion}")
        log_mood(voice_emotion, "voice")

        if voice_emotion in ["sadness", "anger", "fear"]:
            with open("hr_notifications.txt", "a") as file:
                file.write(f"[VOICE] ALERT: {voice_emotion.upper()} detected from voice\n")
            send_email_to_hr(voice_emotion, "Voice input detected")
    except Exception as e:
        st.error(f"Error in voice emotion detection: {e}")

# Team Mood Analysis
st.markdown("## 🧑‍🤝‍🧑 Team Mood Analysis")

try:
    df = pd.read_csv("mood_logs.csv", header=None, names=["Timestamp", "Name", "Source", "Emotion"])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter last 7 days or any logic you want
    recent_df = df[df['Timestamp'] > (datetime.now() - pd.Timedelta(days=7))]

    st.markdown("### Recent Team Emotions (Last 7 Days)")
    st.dataframe(recent_df)

    # Emotion distribution
    emotion_counts = Counter(recent_df["Emotion"])
    emotion_df = pd.DataFrame(emotion_counts.items(), columns=["Emotion", "Count"])
    
    st.bar_chart(emotion_df.set_index("Emotion"))

    # Most common emotion
    if not emotion_df.empty:
        top_emotion = emotion_df.loc[emotion_df["Count"].idxmax()]["Emotion"]
        st.success(f" Most common team emotion this week: **{top_emotion}**")

except FileNotFoundError:
    st.warning("No mood logs found. Team analysis will appear once emotions are recorded.")

