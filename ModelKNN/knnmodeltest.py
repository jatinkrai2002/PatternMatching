
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import cv2



# Load the trained model with error handling
try:
    knntrainmodeloutput_dir = 'C:\\JatinkRai\\patternmatch\\projects\\UpdatedCode\\ModelKNN\\KNNModelOutput'
    
    knntrainmodeloutput_filename = '\\retinal_knn_model_SIUC.pkl'

    knnTrainmodel_fileSavePath = knntrainmodeloutput_dir + knntrainmodeloutput_filename

    knnmodel = joblib.load(knnTrainmodel_fileSavePath)
    

    print("Model knn loaded successfully.")
    print(f"Model type: {type(knnmodel)}")  # Debugging information
except Exception as e:
    print(f"Error loading model: {e}")
    knnmodel = None

# Predict Disease
def predict_disease(image_path):

    try:
        img = load_img(image_path, target_size=(128, 128)) #change to 128,128,
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image array shape: {img_array.shape}")  # Debugging information
         # Ensure the model is correctly loaded and used for prediction

        if knnmodel is None:
            raise AttributeError("Model object does not have looded.")
        if not hasattr(knnmodel, 'predict'):
            raise AttributeError("Model object does not have 'predict' method.")

        knnprediction = knnmodel.predict(img_array)
        print(f"knnPrediction: {knnprediction}")  # Debugging information
        resultval =""

        if knnprediction > 0.5 :
            resultval = resultval +  'KNN model has  Disease Detected'
        else:
            resultval = resultval +  'KNN model has  No Disease Detected'

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

# Function to preprocess a single image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize image to 128x128
    return image.flatten().reshape(1, -1)  # Flatten and reshape to 2D array


try:
    filepath = "C:/JatinkRai/patternmatch/projects/UpdatedCode/mainapp/Test/norm1.jpg"
    image = preprocess_image(filepath)
    prediction = knnmodel.predict(image)
    prediction_proba = knnmodel.predict_proba(image)
    predicted_class = prediction[0]
    predicted_proba = prediction_proba[0][knnmodel.classes_.tolist().index(predicted_class)]
    print(f"Prediction: {predicted_class}")
    print(f"Prediction Probability: {predicted_proba * 100:.2f}%")
    print(f"Prediction: {prediction[0]}")
   # print(f"Prediction Probability: {prediction_proba[0][prediction[0]] * 100:.2f}%")

 #   result = predict_disease(filepath)
   # print(f"Here is KNN predicted result {prediction}")
except Exception as exp:
    print(f"Here is KNN predicted error {exp}") 