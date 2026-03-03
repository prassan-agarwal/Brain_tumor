import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gradio as gr
import keras

IMAGE_SIZE = 224
LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("Enabling unsafe deserialization to allow Lambda layer imports from Notebook...")
keras.config.enable_unsafe_deserialization()

print("Loading models... This may take a moment.")
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Expose to global namespace manually just in case
import builtins
builtins.resnet_preprocess = resnet_preprocess

autoencoder = None
vgg_model = None
resnet_model = None

try:
    vgg_model = load_model('vgg16_tumor_model.keras')
    resnet_model = load_model('resnet50_tumor_model.keras', custom_objects={'resnet_preprocess': resnet_preprocess})
    autoencoder = load_model('autoencoder_tumor_model.keras')
    print("All models successfully loaded!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Starting in safe mode without models for UI checking.")

try:
    with open('anomaly_threshold.txt', 'r') as f:
        ANOMALY_THRESHOLD = float(f.read().strip())
except:
    ANOMALY_THRESHOLD = 0.05 
print(f"Anomaly Threshold Loaded: {ANOMALY_THRESHOLD}")

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image) / 255.0
    return img_array

def is_valid_mri(img_array):
    if autoencoder is None:
        return True, 0.0 
    img_expanded = np.expand_dims(img_array, axis=0)
    reconstructed = autoencoder.predict(img_expanded, verbose=0)
    mse = np.mean(np.square(img_expanded - reconstructed))
    if mse > ANOMALY_THRESHOLD:
        return False, mse
    return True, mse

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if model is None:
        return np.ones((14, 14)) * 0.5
    vgg_base = model.layers[0]
    last_conv_layer = vgg_base.get_layer(last_conv_layer_name)
    vgg_output_extractor = tf.keras.models.Model(
        inputs=vgg_base.inputs,
        outputs=[last_conv_layer.output, vgg_base.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, vgg_preds = vgg_output_extractor(img_array)
        x = vgg_preds
        for layer in model.layers[1:]:
            x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_visualizations(img_array, heatmap):
    img = np.uint8(255 * img_array)
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    superimposed_img = cv2.addWeighted(img, 0.6, jet, 0.4, 0)
    
    thresh = cv2.threshold(heatmap, 128, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    box_info = "Tumor Localization Failed."
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(superimposed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        total_pixels = img.shape[0] * img.shape[1]
        tumor_area = w * h
        tumor_percentage = (tumor_area / total_pixels) * 100
        box_info = f"Tumor Detected at: X({x}-{x+w}), Y({y}-{y+h}) | Size: {tumor_percentage:.2f}% of brain area."
        
    return superimposed_img, box_info

def analyze_mri(image):
    if image is None:
        return None, "Please upload an image.", "Waiting for input..."
        
    img_array = preprocess_image(image)
    
    valid, mse = is_valid_mri(img_array)
    if not valid:
        return image, f"ERROR: Invalid Image Type (MSE: {mse:.4f}). Ensure you upload a Brain MRI.", "Anomaly Detected. Classification Halted."
        
    img_expanded = np.expand_dims(img_array, axis=0)
    
    if vgg_model is None or resnet_model is None:
        # Fallback if models didn't load properly
        return image, "Error: Neural Network weights missing on disk. Run model.save() in Jupyter notebook first.", "Systems Offline."

    vgg_probs = vgg_model.predict(img_expanded, verbose=0)
    resnet_probs = resnet_model.predict(img_expanded, verbose=0)
    
    ensemble_preds = (vgg_probs * 0.5) + (resnet_probs * 0.5)
    pred_index = np.argmax(ensemble_preds[0])
    pred_class = LABELS[pred_index]
    confidence = ensemble_preds[0][pred_index] * 100
    
    if pred_class == 'notumor':
        return image, f"Prediction: No Tumor ({confidence:.2f}% Confidence)", "Healthy Brain Detected."
        
    heatmap = get_gradcam_heatmap(img_expanded, vgg_model, 'block5_conv3', pred_index)
    superimposed_img, box_info = generate_visualizations(img_array, heatmap)
    
    report = (f"Tumor Detected: YES\n"
              f"Diagnosis: {pred_class.upper()} TUMOR\n"
              f"Ensemble Confidence: {confidence:.2f}%\n\n"
              f"Localization Insights:\n{box_info}")
              
    return superimposed_img, report, "Analysis Complete."

# Disable telemetry and start app
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

with gr.Blocks(theme="soft") as app:
    gr.Markdown("# 🧠 Advanced Brain Tumor Diagnosis & Analysis")
    gr.Markdown("Upload a Brain MRI scan to analyze for Glioma, Meningioma, Pituitary, or No Tumor. The system uses an Anomaly Detector to reject invalid images and an Ensemble (VGG16 + ResNet50) for robust classification.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload MRI Image", type="numpy")
            btn = gr.Button("Analyze MRI", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Grad-CAM Tumor Visualization")
            output_text = gr.Textbox(label="Diagnostic Report", lines=6)
            status_text = gr.Textbox(label="System Status")
            
    btn.click(fn=analyze_mri, inputs=input_image, outputs=[output_image, output_text, status_text])

if __name__ == "__main__":
    app.launch()
