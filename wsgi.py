from flask_sqlalchemy import SQLAlchemy
import torch
from pymongo import MongoClient

from flask import request, jsonify
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import torchvision
from crypt import methods
from flask import Flask, request, render_template, make_response, redirect, url_for, jsonify
from flask import Response
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
import cv2
import flash
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import base64
from PIL import Image
import time
import pyautogui
import keyboard
import io

app = Flask(__name__)
app.secret_key = 'secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    number_plate = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    roll_number = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f"Student(number_plate='{self.number_plate}', name='{self.name}', roll_number='{self.roll_number}')"


@app.route('/save', methods=['POST'])
def save_data():
    number_plate = request.form['number_plate']
    name = request.form['name']
    roll_number = request.form['roll_number']
    student = Student(number_plate=number_plate,
                      name=name, roll_number=roll_number)
    try:
        db.session.add(student)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Data saved successfully.'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Failed to save data.', 'error': str(e)})


login_manager = LoginManager()
login_manager.init_app(app)

users = {"john": "pass123", "jane": "pass456"}


@app.route("/check", methods=['POST'])
def check():
    client = MongoClient('mongodb://localhost:27017/')


class User(UserMixin):
    def __init__(self, username):
        self.id = username


@login_manager.user_loader
def load_user(username):
    if username in users:
        return User(username)
    return None


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/home')
@login_required
def index():
    return render_template('index.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username not in users:
            error = "Invalid credentials. Please try again."
        elif users[username] != password:
            error = "Invalid credentials. Please try again."
        else:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
    return render_template("home.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


# ...


@app.route('/opencamera')
@login_required
def opencamera():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            # Capture the video frames
            success, frame = cap.read()
            if not success:
                break
            else:
                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

            # Yield the frame in bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Check for keyboard input
            if keyboard.is_pressed("q"):
                # Save the current frame to disk
                print("Stopped/Started clicking!")
                cv2.imwrite('frame.jpg', frame)
                flash('Image saved!')
                return redirect(url_for('detect'))

            time.sleep(0.03)

    # Create a streaming response object
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/open')
def open():
    return render_template("camera.html")


@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the uploaded image
        dataURL = request.json['image']
        # Remove the data URL prefix and decode the base64-encoded image data
        imgData = base64.b64decode(dataURL.split(',')[1])
        image = Image.open(io.BytesIO(imgData)).convert('RGB')
        gray = np.array(image.convert('L'))

        # Apply Filter and Edge Detection
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edge = cv2.Canny(bfilter, 30, 200)

        # Find Contours and Apply Mask
        keypoints = cv2.findContours(
            edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(gray, gray, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # Use EasyOcr
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        print(result)

        # Return the detected text as a string
        text = ' '.join([x[1] for x in result])
        print(text)
        response = {
            'cropped_image': cropped_image.tolist(),
            'str': text
        }
        return jsonify(response)

    except Exception as e:
        error = f"An error occurred: {str(e)}"
        print(error)
        return jsonify({'error': error})


if __name__ == '__main__':
    app.run(debug=True)
