import numpy as np
from numpy import loadtxt
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_yaml
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

Batch = 16

yaml_file = open('Models/Inception3.Architecture.0.99.50.16.True.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("Models/Inception3.Weight.0.99.50.16.True.h5")
print("Loaded model from disk")


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = test_datagen.flow_from_directory("Images",
                                                        target_size=(299,299),
                                                        color_mode='rgb',
                                                        batch_size=Batch,
                                                        class_mode='categorical',
                                                        shuffle=False)

# file validation picture
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk("Images"):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

Y_pred = loaded_model.predict_generator(validation_generator, len(files) // Batch +1)

y_pred = np.argmax(Y_pred, axis=1)
y_test = validation_generator.classes

print(y_pred)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['can','glass','plastic']
print(classification_report(y_test, y_pred, target_names=target_names))
modelAcc = accuracy_score(y_test, y_pred)