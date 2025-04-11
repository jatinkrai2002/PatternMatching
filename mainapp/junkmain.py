
"""

 Flask web application that serves as a diagnostic tool for ophthalmologists, enabling them to quickly and accurately detect eye diseases from retinal images. The code includes try, except, else, and finally blocks for error handling, and integrates the trained deep learning model.

Python Code (Flask Application)

"""
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'supersecretkey'

# Load the trained model with error handling
try:
    model = load_model('retinal_cnn_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_disease(image_path):
    if model is None:
        return "Model not loaded. Please check the model file."
    try:
        img = load_img(image_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        result = 'Disease Detected' if prediction > 0.5 else 'No Disease Detected'
    except Exception as e:
        print(f"Error during prediction: {e}")
        result = "Error during prediction. Please try again."
    else:
        print("Prediction completed successfully.")
    finally:
        print("Prediction process finished.")
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_disease(filepath)
        except Exception as e:
            print(f"Error during file upload: {e}")
            flash('Error during file upload. Please try again.')
            return redirect(request.url)
        else:
            return render_template('result.html', result=result, filename=filename)
        finally:
            print("File upload process finished.")
    flash('Invalid file format')
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)


"""

Explanation:
Flask Application:

The Flask app is configured to allow image uploads and save them in the uploads directory.
The predict_disease function loads the image, preprocesses it, and uses the trained CNN model to predict whether a disease is detected. It includes error handling using try, except, else, and finally blocks.
The / route renders the index.html template for the home page.
The /upload route handles the file upload, calls the prediction function, and renders the result.html template with the prediction result. It includes error handling for file upload.
HTML Templates:

index.html provides a form for users to upload retinal images.
result.html displays the prediction result for the uploaded image.
This application-oriented approach demonstrates the real-world utility of the developed image analysis techniques, serving as a diagnostic tool for ophthalmologists. Let me know if you need any further assistance!



HTML Code (index.html)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retinal Image Analysis</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mt-5">Retinal Image Analysis</h1>
        <p class="lead">Upload a retinal image to receive disease detection results.</p>
        <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">Choose an image file</label>
                <input type="file" class="form-control-file" id="file" name="file" required>
            </div>
            <button type="submit" class="btn btn-primary">Upload</button>
        </form>
    </div>
</body>
</html>
HTML Code (result.html)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retinal Image Analysis Result</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mt-5">Analysis Result</h1>
        <p class="lead">The result for the uploaded retinal image is:</p>
        <div class="alert alert-info" role="alert">
            {{ result }}
        </div>
        <a href="{{ url_for('index') }}" class="btn btn-primary">Upload Another Image</a>
    </div>
</body>
</html>
"""