from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image
import numpy as np
import joblib
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pickle
import os
import matplotlib.pyplot as plt 
import cv2
import base64
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from flask import send_file
from io import BytesIO
import imghdr
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import cv2
import numpy as np
import matplotlib.pyplot as plt
import io


import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt

# Initialize Flask app with correct static and template folder paths
app = Flask(
    __name__,
    static_folder=r"C:\Users\Windows\Desktop\fracture detection\Backend\model\static",
    template_folder=r"C:\Users\Windows\Desktop\fracture detection\Backend\templates"
)

# Load the XGBoost model
model_path = "C:/Users/Windows/Desktop/fracture detection/Backend/model/static/uploads/model/xgboost_model_with_mobilenet_features (1).joblib"
model = joblib.load(model_path)

# Load the heatmap model
heatmap_model_path = r"C:\Users\Windows\Desktop\fracture detection\Backend\model\static\uploads\model\mobilenet_wrist_model.h5"
heatmap_model = load_model(heatmap_model_path)


# Load MobileNet for feature extraction
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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

# Image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format
    img = img.resize((224, 224))  # Resize the image to 224x224 for MobileNet
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array




# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file:
            # Save the uploaded file temporarily
            image_path = "temp_image.jpg"
            file.save(image_path)

            # Preprocess the image
            img_array = preprocess_image(image_path)

            # Extract features using MobileNet
            features = mobilenet_model.predict(img_array)

            # Flatten features for XGBoost
            features_flattened = features.flatten().reshape(1, -1)

            # Predict with the XGBoost model
            prediction = model.predict(features_flattened)

            # Get predicted index and map to fracture name
            predicted_index = np.argmax(prediction, axis=1)[0] if len(prediction.shape) > 1 else prediction[0]
            predicted_fracture_name = fracture_label_map.get(predicted_index, "Unknown").replace("_", " ")

            return jsonify({"prediction": predicted_fracture_name})

        return jsonify({"error": "No file provided!"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Function to generate Grad-CAM heatmap
# import cv2
# import numpy as np

# def generate_grad_cam_heatmap(heatmap_model, image_path, output_path):
#     """
#     Generate Grad-CAM heatmap for the given image path using the specified model,
#     and save the heatmap to the specified output path.
    
#     Args:
#     - heatmap_model: The model used to generate the Grad-CAM heatmap.
#     - image_path: The file path of the image to generate the heatmap for.
#     - output_path: The file path where the heatmap will be saved.
    
#     Returns:
#     - heatmap: The generated Grad-CAM heatmap.
#     """
    
#     # Load the image
#     img = cv2.imread(image_path)

#     # Check if the image is loaded properly
#     if img is None:
#         raise ValueError(f"Error: Unable to load image from {image_path}")
    
#     print(f"Image loaded successfully from {image_path}")

#     # Ensure the image is not empty
#     if img.size == 0:
#         raise ValueError(f"Error: The image at {image_path} is empty.")

#     try:
#         # Resize the image to the model's input size (224x224 for ResNet or as required)
#         img_resized = cv2.resize(img, (224, 224))  # Resize to model's input size
#         print(f"Image resized successfully to {img_resized.shape}")
#     except cv2.error as e:
#         raise RuntimeError(f"Error resizing image: {e}")

#     # Normalize the image (if required by your model)
#     img_normalized = img_resized / 255.0  # Example: Normalize pixel values to [0, 1]

#     # Reshape the image to match the model's input (e.g., add batch dimension)
#     img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Here, you would pass the img_input to your model and generate the Grad-CAM heatmap
    # For example:
    # grad_cam_heatmap = heatmap_model.predict(img_input)
    # (The actual Grad-CAM generation part depends on your model and setup)

    # Assuming `grad_cam_heatmap` is generated somehow here (e.g., using a Grad-CAM library or method)
    # For now, let's assume `heatmap` is the generated heatmap.
    # heatmap = np.random.rand(224, 224)  # Placeholder for the actual heatmap generation logic

    # # Normalize the heatmap to a range of [0, 255] and apply a color map (e.g., jet)
    # heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    # heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)

    # # Save the heatmap image
    # cv2.imwrite(output_path, heatmap_colored)
    # print(f"Heatmap saved to {output_path}")

    # return heatmap

# # Example usage
# try:
#     image_path = 'C:/Users/Windows/Desktop/fracture detection/Backend/model/static/uploads/temp/Complete_Fracture_BeFunky_fracture_1.jpg'  # Replace with your actual image path
#     output_path = 'C:/Users/Windows/Desktop/fracture detection/Backend/model/static/uploads/uploads/heatmap.jpg'  # Replace with the path where you want to save the heatmap
#     heatmap_model = None  # Replace with your actual model
#     heatmap = generate_grad_cam_heatmap(heatmap_model, image_path, output_path)
#     print("Grad-CAM heatmap generated and saved successfully.")
# except Exception as e:
#     print(f"An error occurred: {e}")

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Save the uploaded file temporarily
#             image_path = "temp_image.jpg"
#             file.save(image_path)
            
#             # Generate heatmap
#             heatmap_image_path = generate_heatmap(image_path)
            
#             # Remove the temporary image file after processing
#             if os.path.exists(image_path):
#                 os.remove(image_path)
            
#             # Redirect to the heatmap page with the heatmap image path
#             return redirect(url_for('heatmap', heatmap_image=heatmap_image_path))
        
#         return "No file uploaded", 400
#     except Exception as e:
#         return str(e), 500

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Save the uploaded file temporarily
#             image_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_image.jpg")
#             file.save(image_path)
            
#             # Generate heatmap
#             heatmap_image_path = generate_grad_cam_heatmap(model, image_path)
            
#             # Remove the temporary image file after processing
#             if os.path.exists(image_path):
#                 os.remove(image_path)
            
#             # Redirect to the heatmap page with the heatmap image path
#             return redirect(url_for('heatmap', heatmap_image='heatmaps/heatmap_result.png'))
        
#         return "No file uploaded", 400
#     except Exception as e:
#         return str(e), 500

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Services route
@app.route('/services')
def services():
    return render_template('services.html')

# Contact route
@app.route('/contact')
def contact():
    return render_template('contact1.html')

# @app.route('/heatmap')
# def heatmap():
#     heatmap_image = request.args.get('heatmap_image', None)  # Fetch the generated heatmap image path
#     return render_template('heatmap.html', heatmap_image=heatmap_image)

# @app.route('/generate-heatmap', methods=['POST'])
# def generate_heatmap():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files['file']
#     img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

#     if img is None:
#         return jsonify({"error": "Invalid image"}), 400

#     img_resized = cv2.resize(img, (224, 224))
#     img_normalized = preprocess_input(np.expand_dims(img_resized, axis=0))

#     # Generate heatmap
#     heatmap = generate_grad_cam_heatmap(heatmap_model, img_normalized)

#     # Resize and overlay heatmap on original image
#     heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

#     overlay_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

#     # Save image to memory and return
#     img_io = BytesIO()
#     _, buffer = cv2.imencode('.png', overlay_img)
#     img_io.write(buffer)
#     img_io.seek(0)

#     return send_file(img_io, mimetype='image/png')


# @app.route('/multi-predict')
# def multi_output_page():
#     return render_template('new_predict.html')

# @app.route('/heatmap', methods=['GET', 'POST'])
# def maps():
#     print(f"Request Method: {request.method}") #for debugging.
#     print(f"Request Headers: {request.headers}") #for debugging.
#     print(f"Request Form Data: {request.form}") #for debugging.
#     if request.method == 'POST':
#         return "POST request received!" #for debugging.
#     return render_template('map.html')






def generate_grad_cam_heatmap(img_path, heatmap_model, alpha=0.5):
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: The image at {img_path} could not be loaded.")
        return None

    img_resized = cv2.resize(img, (224, 224))
    img_normalized = preprocess_input(np.expand_dims(img_resized, axis=0))

    last_conv_layer_name = 'conv_dw_13'

    grad_model = Model(inputs=heatmap_model.input, outputs=[heatmap_model.get_layer(last_conv_layer_name).output, heatmap_model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_normalized)
        class_idx = tf.argmax(preds[0])
        class_output = preds[:, class_idx]

    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(tf.multiply(last_conv_layer_output, pooled_grads), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    overlay_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    buf = BytesIO()
    plt.imshow(overlay_img)
    plt.axis('off')
    plt.savefig(buf, format='png') #save the figure to the buffer
    buf.seek(0)
    plt.clf()
    return buf

# # Grad-CAM heatmap generation function (integrated into app.py)
# def generate_grad_cam_heatmap(img_path, heatmap_model, alpha=0.5):
#     img = cv2.imread(img_path)

#     if img is None:
#         print(f"Error: The image at {img_path} could not be loaded.")
#         return None

#     img_resized = cv2.resize(img, (224, 224))
#     img_normalized = preprocess_input(np.expand_dims(img_resized, axis=0))

#     last_conv_layer_name = 'conv_dw_13'

#     grad_model = Model(inputs=heatmap_model.input, outputs=[heatmap_model.get_layer(last_conv_layer_name).output, heatmap_model.output])

#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_normalized)
#         class_idx = tf.argmax(preds[0])
#         class_output = preds[:, class_idx]

#     grads = tape.gradient(class_output, last_conv_layer_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = tf.reduce_sum(tf.multiply(last_conv_layer_output, pooled_grads), axis=-1)

#     heatmap = np.maximum(heatmap, 0)
#     heatmap = heatmap / np.max(heatmap)

#     heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

#     heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     overlay_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)

#     buf = BytesIO()
#     plt.imshow(overlay_img)
#     plt.axis('off')
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.clf()
#     return buf


# @app.route('/heatmap', methods=['GET', 'POST'])
# def maps():
#     print(f"Request Method: {request.method}")  # for debugging.
#     print(f"Request Headers: {request.headers}")  # for debugging.
#     print(f"Request Form Data: {request.form}")  # for debugging.

#     if request.method == 'POST':
#         file = request.files.get('file')  # Get uploaded file

#         if file:
#             try:
#                 image_path = "temp_image.jpg"
#                 file.save(image_path)

#                 heatmap_image_buffer = generate_grad_cam_heatmap(image_path, heatmap_model)

#                 if heatmap_image_buffer:
#                     heatmap_base64 = base64.b64encode(heatmap_image_buffer.getvalue()).decode('utf-8')
#                     os.remove(image_path)
#                     return render_template('map.html', heatmap_image=heatmap_base64)
#                 else:
#                     os.remove(image_path)
#                     return "Error generating heatmap.", 500

#             except Exception as e:
#                 os.remove(image_path)
#                 return f"Error: {str(e)}", 500

#         else:
#             return "No file uploaded.", 400

#     return render_template('map.html', heatmap_image=None)


# @app.route('/heatmap', methods=['GET', 'POST'])
# def maps():
#     if request.method == 'POST':
#         file = request.files.get('file')
#         print(f"File: {file}") # Debugging
#         if file:
#             try:
#                 image_path = "temp_image.jpg"
#                 file.save(image_path)
#                 print(f"Image saved to: {image_path}") # Debugging

#                 # Original image to base64
#                 with open(image_path, "rb") as image_file:
#                     original_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

#                 # Processed Image to base64
#                 processed_img_array = preprocess_image(image_path)
#                 processed_img = (processed_img_array[0] * 255).astype(np.uint8)
#                 processed_img_pil = Image.fromarray(processed_img)

#                 processed_buf = BytesIO()
#                 processed_img_pil.save(processed_buf, format="PNG")
#                 processed_image_base64 = base64.b64encode(processed_buf.getvalue()).decode('utf-8')

#                 heatmap_image_buffer = generate_grad_cam_heatmap(image_path, heatmap_model)

#                 if heatmap_image_buffer:
#                     heatmap_base64 = base64.b64encode(heatmap_image_buffer.getvalue()).decode('utf-8')
#                     os.remove(image_path)
#                     return render_template('map.html', original_image=original_image_base64, processed_image=processed_image_base64, heatmap_image=heatmap_base64)
#                 else:
#                     os.remove(image_path)
#                     return "Error generating heatmap.", 500

#             except Exception as e:
#                 os.remove(image_path)
#                 print(f"Error: {str(e)}") # Debugging
#                 return f"Error: {str(e)}", 500

#         else:
#             return "No file uploaded.", 400

#     return render_template('map.html', original_image=None, processed_image=None, heatmap_image=None)

@app.route('/heatmap', methods=['GET', 'POST'])
def maps():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                image_path = "temp_image.jpg"
                file.save(image_path)

                # Server-side validation using imghdr
                image_type = imghdr.what(image_path)
                allowed_types = ['png', 'jpeg', 'gif', 'bmp', 'dcm'] # Add 'dcm' if you support it.
                if image_type not in allowed_types:
                    os.remove(image_path)  # Remove invalid file
                    return "Error: Invalid image type. Please upload a PNG, JPG, GIF, BMP, or DICOM image.", 400

                # Original image to base64
                with open(image_path, "rb") as image_file:
                    original_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                # Processed Image to base64
                processed_img_array = preprocess_image(image_path)
                processed_img = (processed_img_array[0] * 255).astype(np.uint8)
                processed_img_pil = Image.fromarray(processed_img)

                processed_buf = BytesIO()
                processed_img_pil.save(processed_buf, format="PNG")
                processed_image_base64 = base64.b64encode(processed_buf.getvalue()).decode('utf-8')

                heatmap_image_buffer = generate_grad_cam_heatmap(image_path, heatmap_model)

                if heatmap_image_buffer:
                    heatmap_base64 = base64.b64encode(heatmap_image_buffer.getvalue()).decode('utf-8')
                    os.remove(image_path)
                    return render_template('map.html', original_image=original_image_base64, processed_image=processed_image_base64, heatmap_image=heatmap_base64)
                else:
                    os.remove(image_path)
                    return "Error generating heatmap.", 500

            except Exception as e:
                os.remove(image_path)
                return f"Error: {str(e)}", 500

        else:
            return "No file uploaded.", 400

    return render_template('map.html', original_image=None, processed_image=None, heatmap_image=None)

# Direct Model Paths (Adjust as necessary)
recovery_model_path = Path(r"C:\Users\Windows\Desktop\fracture detection\Backend\model\static\uploads\model\recovery_model.pkl")
treatment_model_path = Path(r"C:\Users\Windows\Desktop\fracture detection\Backend\model\static\uploads\model\treatment_model.pkl")

# Load Models (with error handling)
if not recovery_model_path.is_file():
    raise FileNotFoundError(f"Recovery model not found at: {recovery_model_path}")

if not treatment_model_path.is_file():
    raise FileNotFoundError(f"Treatment model not found at: {treatment_model_path}")

with open(recovery_model_path, "rb") as f:
    recovery_model, label_encoder_recovery_time = pickle.load(f)

with open(treatment_model_path, "rb") as f:
    treatment_model, label_encoder_treatment = pickle.load(f)


severity_model_path = Path(r"C:\Users\Windows\Desktop\fracture detection\Backend\model\static\uploads\model\severity_model.pkl")
fracture_type_model_path = Path(r"C:\Users\Windows\Desktop\fracture detection\Backend\model\static\uploads\model\fracture_model.pkl")
prognosis_model_path = Path(r"C:\Users\Windows\Desktop\fracture detection\Backend\model\static\uploads\model\prognosis_model.pkl")

# Load Models (with error handling)
if not recovery_model_path.is_file():
    raise FileNotFoundError(f"Recovery model not found at: {recovery_model_path}")

if not treatment_model_path.is_file():
    raise FileNotFoundError(f"Treatment model not found at: {treatment_model_path}")

if not severity_model_path.is_file():
    raise FileNotFoundError(f"Severity model not found at: {severity_model_path}")

if not fracture_type_model_path.is_file():
    raise FileNotFoundError(f"Fracture type model not found at: {fracture_type_model_path}")

if not prognosis_model_path.is_file():
    raise FileNotFoundError(f"Prognosis model not found at: {prognosis_model_path}")


with open(severity_model_path, "rb") as f:
    severity_model, label_encoder_severity = pickle.load(f)

with open(fracture_type_model_path, "rb") as f:
    fracture_type_model, label_encoder_fracture_type = pickle.load(f)

with open(prognosis_model_path, "rb") as f:
    prognosis_model, label_encoder_prognosis = pickle.load(f)

# Load MobileNetV2
mobilenet_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")

# CLAHE and Preprocessing
def apply_clahe(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_images(image_path):
    processed_image = apply_clahe(image_path)
    if processed_image is not None:
        img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_processed = preprocess_input(img_resized)
        return np.expand_dims(img_processed, axis=0)
    return None

# @app.route('/', methods=['GET', 'POST'])
# def multi_output_page():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             image_path = "temp_image.jpg"
#             file.save(image_path)

#             img_array = preprocess_images(image_path)
#             if img_array is not None:
#                 features = mobilenet_model.predict(img_array).flatten().reshape(1, -1)

#                 pred_recovery = recovery_model.predict(features)
#                 predicted_recovery_time = label_encoder_recovery_time.inverse_transform(pred_recovery)[0]

#                 pred_treatment = treatment_model.predict(features)
#                 predicted_treatment = label_encoder_treatment.inverse_transform(pred_treatment)[0]

#                 return render_template('new_predict.html', recovery_time=predicted_recovery_time, treatment=predicted_treatment)
#             else:
#                 return render_template('new_predict.html', error="Failed to process image.")
#         else:
#             return render_template('new_predict.html', error="No file uploaded.")
#     return render_template('new_predict.html')

# @app.route('/', methods=['GET', 'POST'])
# def multi_outputs():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             image_path = "temp_image.jpg"
#             file.save(image_path)

#             img_array = preprocess_images(image_path)
#             if img_array is not None:
#                 features = mobilenet_model.predict(img_array).flatten().reshape(1, -1)

#                 pred_recovery = recovery_model.predict(features)
#                 predicted_recovery_time = label_encoder_recovery_time.inverse_transform(pred_recovery)[0]

#                 pred_treatment = treatment_model.predict(features)
#                 predicted_treatment = label_encoder_treatment.inverse_transform(pred_treatment)[0]

#                 return render_template('new_predict.html', recovery_time=predicted_recovery_time, treatment=predicted_treatment)
#             else:
#                 return render_template('new_predict.html', error="Failed to process image.")
#         else:
#             return render_template('new_predict.html', error="No file uploaded.")
#     return render_template('new_predict.html')


@app.route('/multi-output', methods=['POST'])
def multi_output():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = "temp_image.jpg"
            file.save(image_path)

            img_array = preprocess_image(image_path)
            if img_array is not None:
                features = mobilenet_model.predict(img_array).flatten().reshape(1, -1)

                pred_recovery = recovery_model.predict(features)
                predicted_recovery_time = label_encoder_recovery_time.inverse_transform(pred_recovery)[0]

                pred_treatment = treatment_model.predict(features)
                predicted_treatment = label_encoder_treatment.inverse_transform(pred_treatment)[0]

                pred_severity = severity_model.predict(features)
                predicted_severity = label_encoder_severity.inverse_transform(pred_severity)[0]

                pred_fracture_type = fracture_type_model.predict(features)
                predicted_fracture_type = label_encoder_fracture_type.inverse_transform(pred_fracture_type)[0]

                pred_prognosis = prognosis_model.predict(features)
                predicted_prognosis = label_encoder_prognosis.inverse_transform(pred_prognosis)[0]

                return render_template('new_predict.html',
                                       recovery_time=predicted_recovery_time,
                                       treatment=predicted_treatment,
                                       severity=predicted_severity,
                                       fracture_type=predicted_fracture_type,
                                       prognosis=predicted_prognosis)
            else:
                return render_template('new_predict.html', error="Failed to process image.")
        else:
            return render_template('new_predict.html', error="No file uploaded.")
    return render_template('new_predict.html')


@app.route('/edge', methods=['GET', 'POST'])
def index():
    image_data = None
    error = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = "temp_image.jpg"
            file.save(image_path)

            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    error = "Error: Could not read image."
                else:
                    edges = cv2.Canny(image, threshold1=100, threshold2=200)
                    _, buffer = cv2.imencode('.png', edges)
                    image_data = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                error = f"Error: {e}"
        else:
            error = "No file uploaded."

    return render_template('canny.html', image_data=image_data, error=error)


@app.route('/sample', methods=['GET', 'POST'])
def sample():
    img1_data, img2_data, img3_data, img4_data = None, None, None, None
    error = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = "xray_temp.jpg"
            file.save(image_path)

            try:
                xray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if xray_image is None:
                    error = "Error: Could not read image."
                else:
                    xray_image_normalized = np.float32(xray_image) / 255.0

                    # Visualize the X-ray image
                    plt.imshow(xray_image_normalized, cmap='gray')
                    plt.title("Preprocessed X-ray Image")
                    img1_buffer = io.BytesIO()
                    plt.savefig(img1_buffer, format='png')
                    img1_buffer.seek(0)
                    img1_data = base64.b64encode(img1_buffer.read()).decode('utf-8')
                    plt.close()

                    # Create a Voxel Grid
                    depth_volume = np.tile(xray_image_normalized, (50, 1, 1))

                    # Visualize a slice from the voxel grid
                    plt.imshow(depth_volume[25, :, :], cmap='gray')
                    plt.title("Voxel Grid Slice")
                    img2_buffer = io.BytesIO()
                    plt.savefig(img2_buffer, format='png')
                    img2_buffer.seek(0)
                    img2_data = base64.b64encode(img2_buffer.read()).decode('utf-8')
                    plt.close()

                    # Assign Depth Values
                    depth_volume = 1.0 - depth_volume

                    # Visualize a slice from the depth volume
                    plt.imshow(depth_volume[25, :, :], cmap='gray')
                    plt.title("Depth Volume Slice")
                    img3_buffer = io.BytesIO()
                    plt.savefig(img3_buffer, format='png')
                    img3_buffer.seek(0)
                    img3_data = base64.b64encode(img3_buffer.read()).decode('utf-8')
                    plt.close()

                    # 3D Voxel Grid
                    x, y = np.meshgrid(np.arange(xray_image.shape[1]), np.arange(xray_image.shape[0]))
                    z = np.arange(depth_volume.shape[0])

                    x, y, z = np.meshgrid(np.arange(xray_image.shape[1]), np.arange(xray_image.shape[0]), z)
                    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

                    grid = pv.StructuredGrid()
                    grid.points = points
                    grid.dimensions = (xray_image.shape[1], xray_image.shape[0], depth_volume.shape[0])

                    scalars = depth_volume.ravel()
                    grid.point_arrays['Depth'] = scalars

                    plotter = pv.Plotter(off_screen=True)
                    plotter.add_mesh(grid, show_edges=True, color='white', show_bounds=True)
                    plotter.camera_position = 'xy'
                    plotter.view_xy()
                    plotter.show(interactive=False)
                    img4_buffer = io.BytesIO()
                    plotter.screenshot(img4_buffer)
                    img4_buffer.seek(0)
                    img4_data = base64.b64encode(img4_buffer.read()).decode('utf-8')
                    plotter.close()

            except Exception as e:
                error = f"Error: {e}"
        else:
            error = "No file uploaded."

    return render_template('sample.html', img1_data=img1_data, img2_data=img2_data, img3_data=img3_data, img4_data=img4_data, error=error)


@app.route('/heatmap')
def map():
    return render_template('map.html')

@app.route('/multi-output')
def multi_output_pages():
    return render_template('new_predict.html')

@app.route('/edge')
def canny_edge():
    return render_template('canny.html')

@app.route('/sample')
def sample_image():
    return render_template('sample.html')

if __name__ == "__main__":
    app.run(debug=True)
