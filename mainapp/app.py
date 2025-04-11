
"""
###
# Jatin K Rai
# DawTag# : 856581905

###
"""
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import joblib
import cv2

# Flask libarires
app = Flask(__name__)   

PredictionResults = []

# Load the trained model with error handling
try:
    cnntrainmodeloutput_dir = 'C:\\JatinkRai\\patternmatch\\projects\\UpdatedCode\\ModelCNN\\CNNModelOutput'
    cnntrainmodeloutput_filename = '\\retinal_cnn_model_SIUC.h5'

    cnnTrainmodel_fileSavePath = cnntrainmodeloutput_dir + cnntrainmodeloutput_filename
    cnnmodel = load_model (cnnTrainmodel_fileSavePath)


    knntrainmodeloutput_dir = 'C:\\JatinkRai\\patternmatch\\projects\\UpdatedCode\\ModelKNN\\KNNModelOutput'
    knntrainmodeloutput_filename = '\\retinal_knn_model_SIUC.pkl'
    knnTrainmodel_fileSavePath = knntrainmodeloutput_dir + knntrainmodeloutput_filename

    knnmodel = joblib.load(knnTrainmodel_fileSavePath)


   # model = load_model('C:/JatinkRai/patternmatch/projects/UpdatedCode/Jatin_retinal_knn_model.h5')
    print("Model, knn and cnn loaded successfully.")
    print(f"CNN Model type: {type(cnnmodel)}")  # Debugging information
    print(f"KNN Model type: {type(knnmodel)}")  # Debugging information
except Exception as e:
    print(f"Error loading model: {e}")
    cnnmodel = None
    knnmodel = None


 # Function to preprocess a single image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize image to 128x128
    return image.flatten().reshape(1, -1)  # Flatten and reshape to 2D array


# Predict Disease
def predict_disease(image_path, filename):
    try:
        img = load_img(image_path, target_size=(128,128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image array shape: {img_array.shape}")  # Debugging information
         # Ensure the model is correctly loaded and used for prediction

        if cnnmodel is None:
            raise AttributeError("CNN Model object does not have looded.")
        if not hasattr(cnnmodel, 'predict'):
            raise AttributeError("CNN Model object does not have 'predict' method.")

        cnnprediction = cnnmodel.predict(img_array)
        print(f"cnn Prediction: {cnnprediction}")  # Debugging information

        if knnmodel is None:
            raise AttributeError(" KNN Model object does not have looded.")
        if not hasattr(knnmodel, 'predict'):
            raise AttributeError(" KNN Model object does not have 'predict' method.")

        image = preprocess_image(image_path)
        knnprediction = knnmodel.predict(image)
        knnprediction_proba = knnmodel.predict_proba(image)
        knnpredicted_class = knnprediction[0]
        knnpredicted_proba = knnprediction_proba[0][knnmodel.classes_.tolist().index(knnpredicted_class)]
        print(f"Prediction: {knnpredicted_class}")
        print(f"Prediction Probability: {knnpredicted_proba * 100:.2f}%")
        

        print(f"knnPrediction: {knnprediction}")  # Debugging information
       
        resultval =""

        if cnnprediction > 0.5 :
            resultval = resultval + 'CNN model has Disease Detected. '
        else:
            resultval = resultval +  ' CNN model has No Disease Detected. '

        knnpredictval = knnpredicted_proba;

        if knnpredictval > 0.5 :
            resultval = resultval +  ' However KNN model has  Disease Detected. '
        else:
            resultval = resultval +  ' But KNN model has  NOT Disease Detected '


        PredictionResults.append({'imageName': filename  , 'CNNPredResult':f"{cnnprediction[0][0]* 100:.2f}%", 'KNNPredResult':f"{knnpredicted_proba * 100:.2f}%" , 'Comments':resultval})

        return resultval
        
    except AttributeError as e:
        print(f"AttributeError during prediction: {e}")
        result = "Error during prediction. Please try again."
    except Exception as e:
        print(f"Error during prediction: {e}")
        result = "Error during prediction. Please try again."
    else:
        print("Prediction completed successfully.")
    finally:
        print("Prediction process finished.")

    return result

#Load webapps with / commands
  
@app.route('/')   
def main():   
    return render_template("siuassindex.html")   


@app.route('/siuassindex', methods=['GET', 'POST'])   
def siumain():
    return render_template("siuassindex.html") 


# makesure mainapp is not two times
getcwd = os.getcwd()
uploadedfolder = getcwd + '\\uploadedbywebapp\\'


@app.route('/uploaded', methods = ['POST'])   
def uploaded():   
    if request.method == 'POST': 
        if 'file' not in request.files: 
            return redirect(url_for('Diagnosis'))
        
        fileobj = request.files['file'] 

        if fileobj.filename == '':
            return redirect(url_for('Diagnosis'))

        filepath = uploadedfolder+fileobj.filename
        fileobj.save(filepath)

        try:
            result = predict_disease(filepath, fileobj.filename)
            return render_template('siuassresult.html', result=result, filename=fileobj.filename, PredictionResults= PredictionResults)
        except Exception as exp:
            return render_template("Acknowledgement.html", name = f"Error in predict disease {exp}") 


       # return render_template("Acknowledgement.html", name = fileobj.filename)   


#Call main method
if __name__ == '__main__':   
    app.run(debug=True)
    #app.run(host=”0.0.0.0", port=5000, debug = True)
