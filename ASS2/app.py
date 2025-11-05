from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Use pretrained COCO model (no training needed!)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# COCO dataset animal classes
ANIMAL_CLASSES = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                  'elephant', 'bear', 'zebra', 'giraffe']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Run detection
    results = model(filepath)
    
    # Save result image
    results_img = results.render()[0]
    result_filename = f'result_{filename}'
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, results_img)
    
    # Extract predictions - only animals
    predictions = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        cls_name = model.names[int(cls)]
        # Filter to show only animals
        if cls_name in ANIMAL_CLASSES:
            predictions.append({
                'class': cls_name,
                'confidence': float(conf)
            })
    
    return jsonify({
        'success': True,
        'image_url': f'/static/uploads/{result_filename}',
        'predictions': predictions
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)