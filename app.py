import sounddevice as sd
import librosa
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart 
def send_email_to_hr(emotion, user_input):
    sender_email = "zidio.hr.alerts@gmail.com"              # Your Gmail
    sender_password = "echh ofar bbcx uvsm"             # App password from Gmail
    receiver_email = "jyothsnaguthula@gmail.com"             # HR's email

    subject = f"🔔 Alert: Negative Emotion Detected - {emotion}"
    body = f"""Dear HR Team,

A negative emotion ({emotion}) has been detected from a user input:

"{user_input}"

Please consider reaching out to check on the person.

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
        print("✅ Email sent to HR")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
from deepface import DeepFace
import streamlit as st
import cv2
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load ML model for text
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Emotion Detector", layout="wide")
st.markdown("<h1 style='text-align: center; color: #f0932b;'>🌟 Emotion Detection App 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type how you feel or try the camera! We'll handle it with care 🤗</p>", unsafe_allow_html=True)

# Tabs for text and face
tab1, tab2 = st.tabs(["📝 Text Emotion", "📸 Face Emotion"])

# ---------------- TEXT TAB ---------------- #
with tab1:
    user_input = st.text_input("Enter your sentence:")

    if st.button("Detect Text Emotion"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            st.success(f"**Predicted Emotion:** {prediction}")

            # Alert HR if emotion is negative
            if prediction in ["sadness", "anger", "fear"]:
                with open("hr_notifications.txt", "a") as file:
                    file.write(f"[TEXT] ALERT: {prediction.upper()} | Message: {user_input}\n")
                    st.info("📩 HR Email: zidio.hr.alerts@gmail.com")  # Replace with the actual HR email
                    send_email_to_hr(prediction, user_input)
# ---------------- FACE TAB ---------------- #
with tab2:
    run_cam = st.button("Start Camera Emotion Detection")

    if run_cam:
        st.info("Opening webcam... (press 'q' in camera window to stop)")

        # Face emotion detection logic
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected.")
                break

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']

                # Display on the frame
                cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show frame in Streamlit
                stframe.image(frame, channels="BGR", use_column_width=True)

                # Save alert if needed
                if emotion in ["sad", "angry", "fear"]:
                    with open("hr_notifications.txt", "a") as file:
                        file.write(f"[FACE] ALERT: {emotion.upper()} detected from camera\n")

            except Exception as e:
                stframe.warning("No face detected...")

            # Break if user presses "q"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

st.markdown("### 🎙️ Voice Emotion Detection")
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
        voice_prediction = voice_model.predict(mfccs_scaled)[0]
        st.write(f"🎧 **Detected Voice Emotion:** {voice_prediction}")
        if voice_prediction in ["sad", "angry", "fear"]:
            send_email_to_hr(voice_prediction, "Detected via voice recording")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
