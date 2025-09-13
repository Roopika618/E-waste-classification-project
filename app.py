import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Load trained CNN/Transfer Model
# -----------------------------
model = load_model("e_waste_classifier.keras")

# Define your class labels (update according to your dataset)
class_labels = [
    "Plastic Bottle → Recyclable Waste",
    "Food Scraps → Organic Waste",
    "Mobile Charger → E-Waste",
    "Battery → Hazardous Waste",
    "Other Waste",
    "Glass → Recyclable Waste",
    "Paper → Recyclable Waste",
    "Metal → Recyclable Waste",
    "Clothes → Textile Waste",
    "Wood → Organic Waste"
]

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))   # Resize to model input size
    img = img / 255.0                   # Normalize
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("♻ Real-Time Waste Classification App")
st.write("Upload an image or use live camera to identify the type of waste.")

# -----------------------------
# Option 1: Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    img_array = np.array(image)

    # Preprocess
    processed_img = preprocess_image(img_array)

    # Predict
    prediction = model.predict(processed_img)
    class_index = int(np.argmax(prediction))   # ensure int
    confidence = float(np.max(prediction) * 100)

    # Show result
    st.success(f"Prediction: *{class_labels[class_index]}* ({confidence:.2f}%)")

# -----------------------------
# Option 2: Live Camera (local only)
# -----------------------------
if st.button("Use Camera"):
    st.warning("⚠ Camera works only on local system (not on Streamlit Cloud).")

    cap = cv2.VideoCapture(0)
    st_frame = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_img = preprocess_image(frame)
        prediction = model.predict(processed_img)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        label = f"{class_labels[class_index]} ({confidence:.2f}%)"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        st_frame.image(frame, channels="BGR")

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()