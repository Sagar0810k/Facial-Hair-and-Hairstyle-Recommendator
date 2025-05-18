from flask import Flask, render_template, request, redirect, url_for, Response
import os
import time
import random
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename
from models.detect_face import detect_face_shape
from models.recommendator import get_recommendation

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads/")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"Upload directory created or confirmed at: {UPLOAD_FOLDER}")
except Exception as e:
    print(f"Error creating upload directory: {e}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video source")
        return
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read frame")
                break
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded!", 400
    
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return "Invalid file type!", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        print(f"File saved successfully at: {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return f"Error saving file: {e}", 500
    
    return redirect(url_for("result", filename=filename))

@app.route("/capture", methods=["POST"])
def capture_image():
    try:
        image_data = request.form["image"]
        image_data = image_data.split(",")[1]
        
        image_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        filename = "captured_image.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, image)
        print(f"Captured image saved at: {filepath}")
        
        return redirect(url_for("result", filename=filename))
    except Exception as e:
        print(f"Error capturing image: {e}")
        return f"Error capturing image: {str(e)}", 500

@app.route("/result")
def result():
    filename = request.args.get("filename")
    if not filename:
        return "No file provided!", 400
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return f"File not found: {filepath}", 404

    delay = random.uniform(1.0, 3.0)
    time.sleep(delay)
    
    debug_filename = f"debug_{filename}"
    debug_filepath = os.path.join(app.config["UPLOAD_FOLDER"], debug_filename)
    
    try:
        face_shape = detect_face_shape(filepath, save_debug_image=True)
        recommended_styles = get_recommendation(face_shape)
    except Exception as e:
        print(f"Error in processing: {e}")
        return f"Error processing image: {str(e)}", 500
    
    return render_template(
        "result.html", 
        filename=filename, 
        debug_filename=debug_filename,
        face_shape=face_shape, 
        recommendation=recommended_styles
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)