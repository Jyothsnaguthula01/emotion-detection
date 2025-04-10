# 🔔 Email Alert Function
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_to_hr(emotion):
    sender_email = "zidio.hr.alerts@gmail.com"          # 🔁 your Gmail
    sender_password = "echh ofar bbcx uvsm"          # 🔁 your Gmail App Password
    receiver_email = "jyothsnaguthula@gmail.com"   # ✅ HR's email

    subject = f"📸 Face Emotion Detected: {emotion}"
    body = f"""Dear HR Team,

A negative facial emotion was detected via webcam:

Detected Emotion: {emotion}

Please check in with the concerned individual.

Regards,
Emotion Detection App"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✅ HR has been notified via email.")
    except Exception as e:
        print(f"❌ Email failed to send: {e}")


import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Trigger HR email for negative emotions
        if dominant_emotion in ["sad", "angry", "fear"]:
            send_email_to_hr(dominant_emotion)

    except Exception as e:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




