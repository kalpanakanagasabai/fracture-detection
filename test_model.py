from PIL import Image
import numpy as np
import joblib
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image

# Load the model
model_path = "C:/Users/Windows/Desktop/fracture detection/Backend/model/static/uploads/model/xgboost_model_with_mobilenet_features (1).joblib"
model = joblib.load(model_path)

# Load MobileNet for feature extraction
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Preprocess the image function
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Open the image and ensure it's in RGB format
    img = img.resize((224, 224))  # Resize the image to the expected size (224x224 for MobileNet)
    img_array = np.array(img) / 255.0  # Normalize the image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array

# Fracture name mapping
fracture_label_map = {
    0: "Complete_Fracture_Barton_fracture",
    1: "Complete_Fracture_BeFunky_fracture",
    2: "Complete_Fracture_Bilateral_colles_fracture",
    3: "Complete_Fracture_boxer_fracture",
    4: "Complete_Fracture_chauffeur_fracture",
    5: "Complete_Fracture_colles_fracture",
    6: "Complete_Fracture_concurrent_ulnar-head_and_colles_fractures",
    7: "Complete_Fracture_distal_radial_fracture",
    8: "Complete_Fracture_enchondroma-with-pathological-fracture",
    9: "Complete_Fracture_galeazzi_fracture",
    10: "Complete_Fracture_hamate_fracture",
    11: "Complete_Fracture_pseudo_bennett_fracture",
    12: "Complete_Fracture_radial_styloid_fracture",
    13: "Complete_Fracture_salter-harris-type-iii-fracture",
    14: "Complete_Fracture_smith_fracture",
    15: "Complete_Fracture_transverse-distal-radial-fracture",
    16: "Dislocation_Fracture_bennett_fracture",
    17: "Dislocation_Fracture_displaced-radial-shaft-fracture-with-radial-head-dislocation",
    18: "Dislocation_Fracture_galeazzi_fracture",
    19: "Dislocation_Fracture_perilunate-dislocation",
    20: "Dislocation_Fracture_reverse-bennett-fracture-dislocation",
    21: "Dislocation_Fracture_trans-scaphoid-perilunate-dislocation",
    22: "Incomplete_Fracture_distal-quarter-of-radius-fracture",
    23: "Incomplete_Fracture_distal-radial-greenstick-fracture",
    24: "Incomplete_Fracture_distal-radius-torus-fracture",
    25: "Incomplete_Fracture_greenstick-fracture",
    26: "Incomplete_Fracture_torus-fracture",
    27: "Incomplete_Fracture_transverse-distal-radial-fracture",
    28: "Incomplete_Fracture_triangular-fibrocartilage-complex-tfcc-tear-with-avulsion-fracture-of-the-ulnar-styloid-process (Frontal)",
    29: "Normal_no_fracture"
}

# Test with an image
image_path = 'C:/Users/Windows/Desktop/fracture detection/Backend/model/static/uploads/temp/Dislocation Fracture_galeazzi-fracture-dislocation-4-1-_png.rf (1).jpeg'  # Replace with the actual image path you want to test

# Preprocess the image
img_array = preprocess_image(image_path)

# Extract features from the image using MobileNet
features = mobilenet_model.predict(img_array)

# Flatten the features to match the input shape for XGBoost (e.g., 50176 features for MobileNet)
features_flattened = features.flatten().reshape(1, -1)

# Make prediction using the XGBoost model
prediction = model.predict(features_flattened)

# Get the predicted index
predicted_index = np.argmax(prediction, axis=1)[0] if len(prediction.shape) > 1 else prediction[0]
# Modify the fracture name format by replacing underscores with spaces
predicted_fracture_name = fracture_label_map.get(predicted_index, "Unknown").replace("_", " ")

# Print the predicted label and fracture name
print("Predicted Label Index:", predicted_index)
print("Predicted Fracture Name:", predicted_fracture_name)

