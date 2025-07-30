import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import cv2

st.title("🔍 Face Recognition App")

# Load known encodings and names
known_encodings = np.load("known_encodings.npy", allow_pickle=True)
known_names = np.load("known_names.npy", allow_pickle=True)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect faces and encodings
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    if len(face_encodings) == 0:
        st.warning("No face detected!")
    else:
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            # Draw bounding box
            cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw name label with larger font
            cv2.putText(image_np, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Convert back to PIL and display result
        result_image = Image.fromarray(image_np)
        st.success("Recognition complete!")
        st.image(result_image, caption="Result", use_column_width=True)
