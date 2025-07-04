import os
import uuid
from datetime import datetime
from unittest import result
import cv2
import numpy as np
import psycopg2
import pyautogui
import pygetwindow as gw
from firebase_admin import credentials, firestore, storage
import firebase_admin
from queue import Empty, Full, Queue
import logging
import multiprocessing as mp
from flask import Flask, request, Response, render_template, send_file, jsonify
from threading import Lock, Thread
from io import BytesIO
import time
import random
import string
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
from PIL import Image, ImageEnhance, ImageOps
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
latest_frame = None
frame_lock = Lock()
frame_queue = mp.Queue(maxsize=10)
frame_interval = 1.0 / 10
last_frame_time = 0
process = None
process_running = False

# Connection parameters
host = 'localhost'
database = 'users'
user = 'postgres'
password = '1234'
port = 5432

# Establish the connection
conn = psycopg2.connect(
    host=host,
    database=database,
    user=user,
    password=password,
    port=port
)

# Create a cursor object
cur = conn.cursor()
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'
app.config['OUTPUT_FOLDER'] = 'extracted_and_augmented_frames'

face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

cred = credentials.Certificate("serviceAccountKey.json")
firebase_app = firebase_admin.initialize_app(cred, {
    'storageBucket': "finalfyp-ae408.appspot.com"
})
db = firestore.client()
bucket = storage.bucket()

# Ensure the upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)

def trainModel():
    dataset_dir = Path('D:/python projects/camifyServer/extracted_and_augmented_frames')
    images = []
    labels = []
    for file_path in dataset_dir.glob('*.jpg'):
        try:
            # Open the image using PIL
            with Image.open(file_path) as img:
                # Convert the image to RGB format and then to a NumPy array
                img_array = np.array(img.convert('RGB'))

                # Pass the NumPy array directly to the FaceAnalysis app
                faces = face_app.get(img_array)
                if faces:
                    print(f"Face detected in {file_path}")
                    embedding = faces[0].normed_embedding  # Extract the embedding
                    images.append(embedding)
                    # Extract user ID from the filename
                    user_id = file_path.stem.split('_')[0]
                    labels.append(user_id)
                else:
                    print(f"No face detected in {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if images and labels:
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        clf = SVC(kernel='linear', probability=True)
        clf.fit(images, labels)
        joblib.dump(clf, 'D:/python projects/camifyServer/arcface_classifier.pkl')
        joblib.dump(le, 'D:/python projects/camifyServer/label_encoder.pkl')
        print("Training complete and models saved.")
    else:
        print("No images could be processed, check the dataset and paths.")


@app.route('/register_user', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    if 'cnic' not in request.form:
        return jsonify({'error': 'No CNIC provided'}), 400

    video = request.files['video']
    cnic = request.form['cnic']

    if video.filename == '':
        return jsonify({'error': 'No video selected for uploading'}), 400

    filename = secure_filename(cnic + '.mp4')
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    try:
        first_frame_url = extract_and_augment_images(video_path, app.config['OUTPUT_FOLDER'], cnic)
        trainModel()
        if first_frame_url:
            return jsonify({
                'message': 'Video successfully uploaded and processed',
                'first_frame_url': first_frame_url
            }), 200
        else:
            return jsonify({'error': 'Failed to process video or upload image'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Server error: {}'.format(str(e))}), 500


def extract_and_augment_images(video_path, output_dir, cnic, num_frames=1000):
    cap = cv2.VideoCapture(video_path)
    video_filename = os.path.basename(video_path).split('.')[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)
    current_frame = 0
    extracted_count = 0
    first_frame_url = None

    while extracted_count < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % step == 0:
            frame_id = str(uuid.uuid4())
            img_path = os.path.join(output_dir, f"{video_filename}_{frame_id}.jpg")
            cv2.imwrite(img_path, frame)
            augment_image(img_path)
            extracted_count += 1
            if extracted_count == 1:  # Save the first frame to Firebase Storage
                first_frame_url = upload_to_firebase_storage(img_path, cnic)
                print(first_frame_url)

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()
    return first_frame_url


def upload_to_firebase_storage(file_path, cnic):
    try:
        blob = bucket.blob('images/' + os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Failed to upload {file_path} to Firebase: {str(e)}")
        return None


def augment_image(image_path):
    img = Image.open(image_path)
    augmentations = [
        ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5)),
        ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5)),
        ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5)),
        img.transpose(Image.FLIP_LEFT_RIGHT),
        ImageOps.solarize(img, threshold=random.randint(0, 256)),
        ImageOps.equalize(img)
    ]

    base, ext = os.path.splitext(image_path)
    for i, aug_img in enumerate(augmentations):
        aug_img.save(f"{base}_aug_{i}{ext}")


def frame_processor(frame_queue, pipe, key):
    print(key)
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    intrusion_recorded = False

    def extract_boundary_coords(image_path):
        boundary_image = cv2.imread(image_path)
        hsv = cv2.cvtColor(boundary_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_contour = max(contours, key=cv2.contourArea)
        boundary_coords = [(point[0][0], point[0][1]) for point in boundary_contour]
        return boundary_coords

    def point_in_polygon(x, y, poly):
        num = len(poly)
        j = num - 1
        odd_nodes = False
        for i in range(num):
            if poly[i][1] < y and poly[j][1] >= y or poly[j][1] < y and poly[i][1] >= y:
                if poly[i][0] + (y - poly[i][1]) / (poly[j][1] - poly[i][1]) * (poly[j][0] - poly[i][0]) < x:
                    odd_nodes = not odd_nodes
            j = i
        return odd_nodes

    def update_firestore(detection_data, key):
        # Get the window by title
        window = gw.getWindowsWithTitle('Video')
        if not window:
            print("Video window not found!")
            return

        # Get the window's position and size
        window = window[0]
        left, top, width, height = window.left, window.top, window.width, window.height

        # Capture the screenshot of the specific window area
        screenshot = pyautogui.screenshot(region=(left, top, width, height))

        # Define the file path
        file_path = f'{key}/alerts/'+ str(uuid.uuid4()) + '.png'
        local_path = 'screenshot.png'

        # Save the screenshot locally
        screenshot.save(local_path)

        # Upload to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(file_path)
        blob.upload_from_filename(local_path)

        # Make the file publicly accessible
        blob.make_public()

        # Get the URL of the uploaded file
        image_url = blob.public_url

        # Get the current date and time
        current_time = datetime.now().isoformat()

        # Add image URL and timestamp to detection data
        detection_data.update({
            'image_url': image_url,
            'timestamp': current_time
        })

        # Initialize Firestore client
        db = firestore.client()

        # Reference the Firestore document
        doc_ref = db.collection('taskCollection').document(key).collection('alerts').document()

        # Set the detection data in Firestore
        doc_ref.set(detection_data)

        # Clean up the local file
        os.remove(local_path)

    boundary_coords = extract_boundary_coords('image_with_boundary.jpg')

    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
        except Empty:
            continue

        def process_frame(frame):
            nonlocal intrusion_recorded  # Access the variable defined in the outer scope
            start_time = time.time()

            height, width, _ = frame.shape

            scaled_boundary_coords = [
                (int(x * width / 720), int(y * height / 480)) for x, y in boundary_coords
            ]

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for point in scaled_boundary_coords:
                cv2.circle(frame, (point[0], point[1]), 5, (0, 255, 0), -1)
            cv2.polylines(frame, [np.array(scaled_boundary_coords)], isClosed=True, color=(0, 255, 0), thickness=2)

            intrusion_detected = False
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                center_x, center_y = x + w // 2, y + h // 2

                if point_in_polygon(center_x, center_y, scaled_boundary_coords):
                    intrusion_detected = True
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = "Intrusion" if intrusion_detected else "No Intrusion"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if intrusion_detected and not intrusion_recorded:
                detection_data = {'detection': 'intrusion'}
                thread = Thread(target=update_firestore, args=(detection_data,key))
                thread.start()
                intrusion_recorded = True
            elif not intrusion_detected:
                intrusion_recorded = False

            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Frame processed in {processing_time:.2f} seconds")

            return frame

        processed_frame = process_frame(frame)

        cv2.imshow('Video', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', methods=['POST'])
def video_feed():
    global latest_frame, last_frame_time
    frame = request.get_data()

    def process_frame(frame_data):
        global last_frame_time  # Ensure it's global to avoid reference issues
        decoded_frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        with frame_lock:
            global latest_frame
            latest_frame = decoded_frame
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            last_frame_time = current_time
            try:
                frame_queue.put_nowait(decoded_frame)
            except Full:
                pass

    # Start a new thread to process the frame
    thread = Thread(target=process_frame, args=(frame,))
    thread.start()

    return "Frame received", 200

@app.route('/user_detector', methods=['POST'])
def start_frame_processor():
    global process, process_running
    data = request.get_json()
    key = data.get('key')
    if process_running:
        return "Frame processor is already running", 400

    process = mp.Process(target=frame_processor, args=(frame_queue, None,key))
    process.start()
    process_running = True
    return "Frame processor started", 200

@app.route('/stop_user_detector', methods=['POST'])
def stop_frame_processor():
    global process, process_running
    if not process_running:
        return "Frame processor is not running", 400

    frame_queue.put(None)
    process.join()
    process_running = False
    return "Frame processor stopped", 200

def generate_key():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

@app.route('/save_user', methods=['POST'])
def save_user():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    homeaddress = data.get('homeaddress')
    phone = data.get('phone')
    key = generate_key()

    def save_to_db():
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO keys (key) VALUES (%s) RETURNING id", (key,))
            key_id = cur.fetchone()[0]
            cur.execute("INSERT INTO userdata (name, email, homeaddress, phone, key_id) VALUES (%s, %s, %s, %s, %s)",
                        (name, email, homeaddress, phone, key_id))
            conn.commit()
            result = {'key': key, 'name': name, 'email': email}
        except psycopg2.Error as e:
            result = {'error': str(e)}
        finally:
            cur.close()

        return result

    thread = Thread(target=save_to_db)
    thread.start()
    thread.join()  # Wait for the thread to complete if you need the result immediately

    return jsonify(result)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.get_json()
    key = data.get('key')
    if not key:
        return jsonify({'error': 'Key not provided'}), 400

    result_queue = Queue()

    def authenticate_user(queue):
        cur = conn.cursor()
        try:
            cur.execute("SELECT * FROM keys WHERE key = %s", (key,))
            row = cur.fetchone()
            if not row:
                queue.put({'message': 'failure'})
                return

            key_id = row[0]
            cur.execute("SELECT * FROM userdata WHERE key_id = %s", (key_id,))
            user_row = cur.fetchone()
            if not user_row:
                queue.put({'message': 'failure'})
                return

            user_data = {
                'name': user_row[0],
                'email': user_row[1],
                'homeaddress': user_row[2],
                'phone': user_row[3]
            }
            queue.put({'message': 'all set', 'userData': user_data})
        except psycopg2.Error as e:
            queue.put({'error': str(e)})
        finally:
            cur.close()

    thread = Thread(target=authenticate_user, args=(result_queue,))
    thread.start()
    thread.join()  # Wait for the thread to complete

    result = result_queue.get()
    return jsonify(result)

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            with frame_lock:
                frame = latest_frame.copy()
            resized_frame = cv2.resize(frame, (720, 480))
            ret, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

@app.route('/video_feed')
def video_feed_stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image')
def capture_image():
    global latest_frame
    logging.info("Capture image endpoint hit")
    with frame_lock:
        if latest_frame is not None:
            logging.info("Frame is available, encoding")
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                logging.error("Could not encode frame")
                return "Could not encode frame", 500
            io_buf = BytesIO(buffer)
            logging.info("Sending file")
            return send_file(io_buf, mimetype='image/jpeg', as_attachment=True, download_name='capture.jpg')
        else:
            logging.warning("No frame available")
            return "No frame available", 404

@app.route('/upload_image', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        logging.error("No image file in the request")
        return "No image file in the request", 400

    file = request.files['image']
    if file.filename == '':
        logging.error("No selected file")
        return "No selected file", 400

    filename = 'image_with_boundary.jpg'
    file.save(filename)
    logging.info("Image saved")
    img = cv2.imread(filename)
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_width = 720
    new_height = int(new_width / aspect_ratio)
    resized_img = cv2.resize(img, (new_width, new_height))
    cv2.imwrite(filename, resized_img)

    logging.info("Image resized and saved")
    return "Image uploaded and resized successfully", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
