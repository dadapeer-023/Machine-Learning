from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

train_path = "fruits-360/Training/" 
test_path = "fruits-360/Test/"
batch_size = 32 

# Load the saved model
loaded_model = load_model('fruits-cnn-model.keras')

img = load_img(test_path + "Apple Braeburn/3_100.jpg")
x = img_to_array(img) 

train_datagen = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3, 
                   horizontal_flip=True, 
                   zoom_range = 0.3) 


test_datagen = ImageDataGenerator(rescale= 1./255) 

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size = x.shape[:2], 
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size=x.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")


true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
# print("class_labels", test_generator.class_indices)


# Function to predict class labels for an image
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size= (100, 100, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Rescale pixel values to [0, 1]
    predicted_class = loaded_model.predict(x)
    predicted_label = class_labels[np.argmax(predicted_class)]
    return predicted_label

image_path = test_path + "Apple Golden 1/63_100.jpg"

predicted_label = predict_image_class(image_path)
print("Predicted label:", predicted_label)
