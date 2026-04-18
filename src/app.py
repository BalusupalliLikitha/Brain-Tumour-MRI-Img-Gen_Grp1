import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


# We move model loading outside so it stays in memory
@st.cache_resource
def get_model():
    model = load_model("models/classifier.keras")
    return model


def get_gradcam(model, img_array):
    if hasattr(model.layers[0], 'layers'):
        target_model = model.layers[0]
    else:
        target_model = model

    last_conv_layer = None
    for layer in reversed(target_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    img_input = tf.keras.Input(shape=(64, 64, 1))
    x = img_input
    conv_out = None

    for layer in target_model.layers:
        x = layer(x)
        if layer == last_conv_layer:
            conv_out = x

    grad_model = tf.keras.Model(inputs=img_input, outputs=[conv_out, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        preds = tf.clip_by_value(predictions[:, 0], 1e-7, 1.0 - 1e-7)
        loss = tf.math.log(preds / (1 - preds))

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return np.array(heatmap)


# THE NEW PART: Wrap everything in this function
def show_detection():
    model = get_model()

    st.title(" Multi-Image Tumor Detection with Grad-CAM")

    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    uploaded_files = st.file_uploader("Upload MRI Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        num_images = len(uploaded_files)
        originals, gradcams, metadata = [], [], []

        if hasattr(model.layers[0], 'layers'):
            internal_layers = model.layers[0].layers
        else:
            internal_layers = model.layers

        logit_input = tf.keras.Input(shape=(64, 64, 1))
        curr_x = logit_input
        for layer in internal_layers[:-1]:
            curr_x = layer(curr_x)
        logit_model = tf.keras.Model(inputs=logit_input, outputs=curr_x)

        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            img_resized = cv2.resize(img_raw, (64, 64))
            img_input = (img_resized / 255.0).astype(np.float32).reshape(1, 64, 64, 1)

            raw_logit = logit_model(img_input).numpy()[0][0]
            pred = model(img_input).numpy()[0][0]

            if raw_logit > 12.0:
                label, l_color = "Tumor (High Risk)", "red"
            elif raw_logit > 9.5:
                label, l_color = "Suspicious / Borderline", "orange"
            else:
                label, l_color = "Normal", "green"

            heatmap = get_gradcam(model, img_input)
            heatmap[heatmap < 0.01] = 0
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)

            heatmap_vis = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]))
            heatmap_vis = np.uint8(255 * heatmap_vis)
            heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

            original_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            superimposed = cv2.addWeighted(original_bgr, 0.5, heatmap_color, 0.5, 0)
            superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

            originals.append(img_raw)
            gradcams.append(superimposed_rgb)
            metadata.append((label, pred, l_color, raw_logit))

        st.subheader(" Original MRI Images")
        cols_orig = st.columns(num_images)
        for i in range(num_images):
            cols_orig[i].image(originals[i], use_container_width=True)

        st.subheader(" Grad-CAM Highlights")
        cols_grad = st.columns(num_images)
        for i in range(num_images):
            cols_grad[i].image(gradcams[i], use_container_width=True)

        st.subheader("Diagnostic Results")
        cols_pred = st.columns(num_images)
        for i in range(num_images):
            label, pred, color, raw_logit = metadata[i]
            cols_pred[i].markdown(f"**:{color}[{label}]**")
            cols_pred[i].write(f"Raw Intensity: {raw_logit:.2f}")