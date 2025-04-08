import smtplib
from email.message import EmailMessage

def send_hr_email(emotion, message):
    hr_email = "zidio.hr.alerts@gmail.com"  # 🔁 replace with the email you created for HR alerts
    app_password = "echh ofar bbcx uvsm"  # 🔁 paste your app password here

    msg = EmailMessage()
    msg.set_content(
    f"🚨 ALERT: Negative emotion detected\n\n"
    f"Emotion: {emotion}\n"
    f"Message: {message}\n\n"
    "Please take appropriate action."
)


    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(hr_email, app_password)
            smtp.send_message(msg)
        print("HR Email sent successfully!")
    except Exception as e:
        print("Failed to send HR email:", e)
