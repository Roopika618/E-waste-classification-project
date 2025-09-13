import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("e_waste_classifier.h5")

# ✅ Update this with the exact number of classes from your training
class_labels = {
    0: "Plastic Bottle → Recyclable Waste",
    1: "Food Scraps → Organic Waste",
    2: "Mobile Charger → E-Waste",
    3: "Battery → Hazardous Waste"
}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (128, 128))   # match input size of model
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    # ✅ Safety check to avoid IndexError
    if class_index < len(class_labels):
        label = class_labels[class_index]
        confidence = np.max(prediction) * 100
        text = f"{label}: {confidence:.2f}%"
    else:
        text = "Unknown"

    # Show result
    cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Prediction", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()