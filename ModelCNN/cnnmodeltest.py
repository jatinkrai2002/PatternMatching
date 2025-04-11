import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img



# Load the trained model with error handling
try:
   
    cnntrainmodeloutput_dir = 'C:\\JatinkRai\\patternmatch\\projects\\UpdatedCode\\ModelCNN\\CNNModelOutput'
    cnntrainmodeloutput_filename = '\\retinal_cnn_model_SIUC.h5'

    cnnTrainmodel_fileSavePath = cnntrainmodeloutput_dir + cnntrainmodeloutput_filename
    
    cnnmodel = load_model (cnnTrainmodel_fileSavePath)

    print("Model cnn loaded successfully.")
    print(f"Model type: {type(cnnmodel)}")  # Debugging information
except Exception as e:
    print(f"Error loading model: {e}")
    cnnmodel = None

# Predict Disease
def predict_disease(image_path):

    try:
        # includes resizing, normalization, and augmentation to enhance the robustness of the model.
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image array shape: {img_array.shape}")  # Debugging information
         # Ensure the model is correctly loaded and used for prediction

        if cnnmodel is None:
            raise AttributeError("Model object does not have looded.")
        if not hasattr(cnnmodel, 'predict'):
            raise AttributeError("Model object does not have 'predict' method.")

     
        cnnprediction = cnnmodel.predict(img_array)
      
        print(f"cnnPrediction: {cnnprediction}")  # Debugging information

        #cnnprodval = cnnprediction[0]
        
        print(f"Prediction Probability: {cnnprediction[0][0]* 100:.2f}%")
        resultval =""

        if cnnprediction > 0.5 :
            resultval = resultval + 'CNN model has Disease Detected'
        else:
            resultval = resultval +  'CNN model has No Disease Detected'

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


try:
    filepath = "C:/JatinkRai/patternmatch/projects/UpdatedCode/mainapp/Test/diab1.jpeg"      # "norm1.jpg"
    result = predict_disease(filepath)
    print(f"Here is CNN predicted result {result}")
except Exception as exp:
    print(f"Here is CNN predicted error {exp}") 