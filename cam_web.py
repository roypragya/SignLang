import cv2
import cvzone
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, Response
import tensorflow as tf

app=Flask(__name__)

def generate_frames():
    
    CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

    def prepare(filepath):
        IMG_SIZE = 256
        new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
        arr1 = np.zeros((3,256,256))
        arr1[0,:,:]=new_array
        new_array=arr1.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        return new_array
    
    model = tf.keras.models.load_model("SignLang.h5")

    # url="http://192.168.0.104:8080/video"

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (200, 200))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        prediction = model.predict([prepare(gray)])
        final = (CATEGORIES[int(np.argmax(prediction[0]))])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,final,(200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow('Input', frame)
        
        c = cv2.waitKey(1)
        if c == 27: # hit esc key to stop
            break
    
        # translate()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index1.html')            

    
# @app.route('/sign_lang')
# def translate():
#     return render_template('sign_lang.html')

# Stream video

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Driver code
if __name__ == "__main__":
    app.run(debug=True)
