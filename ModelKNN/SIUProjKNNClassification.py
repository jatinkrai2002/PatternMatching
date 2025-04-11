"""

Name: Jatin K rai

K-NN implementation: The knn_classification function scales the data and applies the K-NN algorithm.


def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))  # Resize images to 128x128
            images.append(image.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)


    # Function to load images and labels

def load_data(data_dir, target_size=(224, 224), batch_size=32): #target_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    images, labels = [], []
    for i in range(len(generator)):
        img, label = generator[i]
        images.append(img[0])
        labels.append(label[0])



    
    return np.array(images), np.array(labels)

"""
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib


# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(data_dir):
            image_path = os.path.join(data_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128)) # Resize images to 128x128
            images.append(image.flatten()) # Flatten the image to 1D
            labels.append(label)
    return np.array(images), np.array(labels)


# Load data

#path to Train data is local folder :- 

traindata_dir='C:/JatinkRai/patternmatch/projects/UpdatedCode' \
'                /datasets/Images' \
'              /Training_Set/Training_Set/Training'                       #'C:/JatinkRai/patternmatch/projects/UpdatedCode/datasets/images/' #

X, y = load_data(traindata_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN model
knn = KNeighborsClassifier(n_neighbors=3)
try:

    trainmodeloutput_dir = 'C:\\JatinkRai\\patternmatch\\projects\\UpdatedCode\\ModelKNN\\KNNModelOutput'
    trainmodeloutput_filename = '\\retinal_knn_model_SIUC.pkl'
    Trainmodel_fileSavePath = trainmodeloutput_dir + trainmodeloutput_filename

    # Train the model
    knn.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(knn, Trainmodel_fileSavePath)
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("Model trained and saved successfully.")
finally:
    print("Model training process completed.")

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
