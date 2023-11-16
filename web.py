import cv2
import streamlit as st
import numpy as np
import tempfile

cap = cv2.VideoCapture(1)
st.title("Video Capture with OpenCV")
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")

script = """
<script>
  // Example: Send a message to the parent window
  window.parent.postMessage({ message: 'Hello from Streamlit!' }, '*');
</script>
"""

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()
    if not ret:
        st.write("The video capture has ended.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        st.markdown(script, unsafe_allow_html=True)
        break

cap.release()
cv2.destroyAllWindows()
