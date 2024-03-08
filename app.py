from flask import Flask, render_template, request, jsonify
from compare_faces import compare_faces, extract_features
import os
import shutil
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    # Get the uploaded images
    image1 = request.files['image1']
    image2 = request.files['image2']

    # Create the temporary directory if it does not already exist
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save the images to the temporary directory
    image1_path = os.path.join(temp_dir, 'image1.png')
    image2_path = os.path.join(temp_dir, 'image2.png')
    image1.save(image1_path)
    image2.save(image2_path)

    # Extract features and compare the images
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)
    result = compare_faces(features1, features2)

    # Clean up the temporary directory
    if os.listdir(temp_dir):
        shutil.rmtree(temp_dir)

    # Convert NumPy arrays to Python lists
    features1 = features1.tolist()
    features2 = features2.tolist()
    result = result.tolist()

    # Return the result
    return json.dumps({'result': result, 'features1': features1, 'features2': features2})

if __name__ == '__main__':
    app.run(debug=True)