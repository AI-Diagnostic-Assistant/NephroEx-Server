import re
import uuid

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import io
import torch
import os
import cv2
import joblib
from PIL import Image
from scipy.stats import skew, kurtosis
import shap
import ollama



from app.logic.CNN import Simple3DCNN

feature_maps = None
gradients = None


def save_image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf

def forward_hook(module, input, output):
    global feature_maps
    feature_maps = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def generate_gradcam():
    """
    Generate the Grad-CAM heatmap.
    :param feature_maps: Feature maps from the target convolutional layer (torch.Tensor).
    :param gradients: Gradients for the feature maps (torch.Tensor).
    :return: Heatmap (numpy array).
    """
    global feature_maps
    global gradients
    # Global average pooling of gradients to compute weights
    weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)  # Shape: [1, 128, 1, 1, 1]

    # Weighted sum of the feature maps
    gradcam = torch.sum(weights * feature_maps, dim=1).squeeze(0)  # Shape: [45, 32, 32]

    # Apply ReLU to the heatmap
    gradcam = torch.relu(gradcam)

    # Convert to numpy
    gradcam = gradcam.detach().cpu().numpy()

    # Normalize the heatmap to [0, 1]
    gradcam /= np.max(gradcam)

    return gradcam

def aggregate_heatmap(gradcam, method="max"):
    """
    Aggregate the heatmap across the depth dimension.
    :param gradcam: 3D Grad-CAM heatmap (numpy array of shape [depth, height, width]).
    :param method: Aggregation method ("max" or "mean").
    :return: 2D heatmap.
    """
    if method == "max":
        return np.max(gradcam, axis=0)  # Maximum Intensity Projection
    elif method == "mean":
        return np.mean(gradcam, axis=0)  # Average Projection
    else:
        raise ValueError("Invalid method. Use 'max' or 'mean'.")


def create_composite_image(dicom_image):
    # Generate a composite image (e.g., by averaging)
    pixel_array = dicom_image.pixel_array.astype(np.float32)
    composite_image = np.max(pixel_array, axis=0)

    # Normalize composite image to 0-255 for visualization
    composite_image = (composite_image - composite_image.min()) / (composite_image.max() - composite_image.min()) * 255
    composite_image = composite_image.astype(np.uint8)

    return composite_image


def overlay_heatmap(composite_image, heatmap, beta=0.3):
    """
    Overlay the heatmap on the composite image.
    :param composite_image: 2D composite image of the input volume.
    :param heatmap: 2D Grad-CAM heatmap.
    :param beta: Transparency level for the heatmap.
    :return: Overlayed image.
    """
    # Resize heatmap to match the composite image size
    heatmap_resized = cv2.resize(heatmap, (composite_image.shape[1], composite_image.shape[0]))

    # Apply color map to heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Convert BGR to RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay the heatmap on the composite image
    overlay = cv2.addWeighted(cv2.cvtColor(composite_image, cv2.COLOR_GRAY2BGR), 1 - beta, heatmap_color, beta, 0)
    return overlay


def run_single_classification_cnn(dicom_read):
    model = Simple3DCNN()
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    model_path = os.path.join(script_dir, "../../models/cnn/best_3dcnn_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    img = dicom_read.pixel_array.astype(np.float32)
    img /= np.max(img)  # Normalize
    volume_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    model.conv3.register_forward_hook(forward_hook)
    model.conv3.register_backward_hook(backward_hook)

    output = model(volume_tensor)
    predicted = torch.argmax(output, dim=1)

    predicted_class = predicted.item()

    model.zero_grad()

    # Perform backward pass for the target class
    target_class = predicted_class  # Example target class
    output[:, target_class].backward()

    # Access the captured feature maps and gradients
    print(f"Feature Maps Shape: {feature_maps.shape}")
    print(f"Gradients Shape: {gradients.shape}")

    print(f"Predicted class: {predicted_class}")
    print("Output probabilities:", output.flatten())

    return predicted_class, output.flatten()


def run_single_classification_svm(roi_activity_array):
    script_dir = os.path.dirname(__file__)
    svm_model_path = os.path.join(script_dir, "../../models/svm/svm_best_model_total.joblib")
    svm_model = joblib.load(svm_model_path)

    roi_activity_array = np.array(roi_activity_array).reshape(1, -1)

    print("roi_activity_array", roi_activity_array.shape)

    probabilities = svm_model.predict_proba(roi_activity_array)[0]
    predicted_class = np.argmax(probabilities)

    predicted_class = int(predicted_class)

    return predicted_class, probabilities


def create_composite_image_rgb(dicom_read):
    pixel_data_list = []

    pixel_array = dicom_read.pixel_array.astype(float)
    pixel_data_list.append(pixel_array)

    pixel_data = np.stack(pixel_data_list, axis=0)

    # Create composite image by summing all frames
    composite_image = np.sum(pixel_data, axis=1)
    composite_image = composite_image.squeeze()  # Remove singleton dimensions
    composite_image_normalized = composite_image / np.max(composite_image)

    # Convert to 3-channel RGB by repeating the grayscale channel
    composite_image_normalized = np.expand_dims(composite_image_normalized, axis=-1)
    composite_image_normalized = np.repeat(composite_image_normalized, 3, axis=-1)

    return composite_image_normalized


def load_image(path: str):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    img /= np.max(img)

    return img


def predict_kidney_masks(composite_image):
    script_dir = os.path.dirname(__file__)
    unet_model_path = os.path.join(script_dir, "../../models/unet/fold_1_pretrained_unet_model.pth")
    unet_model = torch.load(unet_model_path, map_location=torch.device('cpu'))  # Load the entire model
    unet_model.eval()

    image_tensor = torch.tensor(composite_image).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        pred_masks = unet_model(image_tensor)

        pred_left_mask = (pred_masks[:, 0, :, :] > 0.5).cpu().numpy().squeeze()
        pred_right_mask = (pred_masks[:, 1, :, :] > 0.5).cpu().numpy().squeeze()

    return pred_left_mask, pred_right_mask


def align_masks_over_frames(left_kidney_mask, right_kidney_mask, dicom_read):
    left_mask_alignments = []
    right_mask_alignments = []

    global_max = max(np.max(dicom_read.pixel_array[frame_idx]) for frame_idx in range(dicom_read.NumberOfFrames))

    for frame_idx in range(dicom_read.NumberOfFrames):
        pixel_array = dicom_read.pixel_array[frame_idx]
        normalized_pixel_array = (pixel_array / global_max * 255).astype(np.uint8)

        # Ensure the kidney masks are in the range [0, 1]
        if left_kidney_mask.max() <= 1.0 and right_kidney_mask.max() <= 1.0:
            left_kidney_mask = (left_kidney_mask * 255).astype(np.uint8)
            right_kidney_mask = (right_kidney_mask * 255).astype(np.uint8)

        # Ensure the kidney ROI is single-channel (grayscale)
        if left_kidney_mask.ndim == 3 and right_kidney_mask.ndim == 3:
            left_kidney_mask = cv2.cvtColor(left_kidney_mask, cv2.COLOR_BGR2GRAY)
            right_kidney_mask = cv2.cvtColor(right_kidney_mask, cv2.COLOR_BGR2GRAY)

        # Resize and apply the left kidney mask
        resized_left_kidney_roi = cv2.resize(left_kidney_mask,
                                             (normalized_pixel_array.shape[1], normalized_pixel_array.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)

        left_masked_frame = cv2.bitwise_and(normalized_pixel_array, normalized_pixel_array,
                                            mask=resized_left_kidney_roi)

        # Resize and apply the right kidney mask
        resized_right_kidney_roi = cv2.resize(right_kidney_mask,
                                              (normalized_pixel_array.shape[1], normalized_pixel_array.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
        right_masked_frame = cv2.bitwise_and(normalized_pixel_array, normalized_pixel_array,
                                             mask=resized_right_kidney_roi)

        left_mask_alignments.append(left_masked_frame)
        right_mask_alignments.append(right_masked_frame)

    return left_mask_alignments, right_mask_alignments


def create_renogram(left_mask_alignments, right_mask_alignments):
    left_activities = []
    right_activities = []
    total_activities = []

    for frame_idx in range(len(left_mask_alignments)):
        left_activity = compute_activity(left_mask_alignments[frame_idx])
        right_activity = compute_activity(right_mask_alignments[frame_idx])

        left_activities.append(left_activity)
        right_activities.append(right_activity)
        total_activities.append(left_activity + right_activity)

    return np.array(left_activities), np.array(right_activities), np.array(total_activities)


def compute_activity(image):
    return np.mean(image)


def visualize_masks(image, left_mask, right_mask):
    plt.figure(figsize=(15, 8))

    # Display the original image
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.title("Composite Image")
    plt.axis("off")

    # Display the left kidney mask
    plt.subplot(2, 3, 4)
    plt.imshow(left_mask, cmap='gray')
    plt.title("Left Kidney Mask")
    plt.axis("off")

    # Display the right kidney mask
    plt.subplot(2, 3, 5)
    plt.imshow(right_mask, cmap='gray')
    plt.title("Right Kidney Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def perform_svm_analysis(dicom_read, supabase_client):
    # Create composite image of the request file
    composite_image = create_composite_image_rgb(dicom_read)

    # Predict kidney masks
    left_mask, right_mask = predict_kidney_masks(composite_image)

    # align masks over all frames in the original dicom file
    left_mask_alignments, right_mask_alignments = align_masks_over_frames(left_mask, right_mask, dicom_read)

    # Create renogram from the predicted masks
    left_activities, right_activities, total_activities = create_renogram(left_mask_alignments, right_mask_alignments)
    roi_activity_array = [left_activities.tolist(), right_activities.tolist(), total_activities.tolist()]

    # Predict CKD stage with SVM model
    svm_predicted, svm_probabilities = run_single_classification_svm(roi_activity_array[2]) # total activities

    return svm_predicted, svm_probabilities, roi_activity_array, left_mask, right_mask, total_activities.tolist()


def group_2_min_frames(dicom_read):
    frames = dicom_read.pixel_array

    if len(frames) != 180:
        raise ValueError("Expected 180 frames in the DICOM file.")

    # Group into 15 summed frames
    grouped_frames = []
    for i in range(0, 180, 12):
        group = frames[i:i + 12]
        summed_frame = np.sum(group, axis=0)
        grouped_frames.append(summed_frame)

    return grouped_frames


def save_summed_frames_to_storage(grouped_frames, sb_client):
    storage_ids = []

    try:
        for idx, frame in enumerate(grouped_frames):
            normalized_frame = (255 * (frame - frame.min()) / (frame.max() - frame.min())).astype(np.uint8)

            path = save_png(normalized_frame, 'grouped-dicom-frames', sb_client)

            storage_ids.append(path)

        return storage_ids

    except Exception as e:
        raise RuntimeError(f"Error processing DICOM file: {str(e)}")


def save_total_dicom(file_stream, sb_client):
    try:
        # Upload to Supabase storage
        storage_id = uuid.uuid4()
        file_name = f"{storage_id}.dcm"

        response = sb_client.storage.from_('patient-dicom-files').upload(file_name, file_stream, file_options={
            'content-type': 'application/dicom'})

        return response.path

    except Exception as e:
        raise RuntimeError(f"Error processing DICOM file: {str(e)}")


def visualize_grouped_frames(grouped_frames):
    num_frames = len(grouped_frames)
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))
    for i, frame in enumerate(grouped_frames):
        normalized_frame = (255 * (frame - frame.min()) / (frame.max() - frame.min())).astype(np.uint8)
        axes[i].imshow(normalized_frame, cmap="gray")
        axes[i].set_title(f"Frame Group {i + 1}")
        axes[i].axis("off")
    plt.show()


def create_ROI_contours_png(mask_left, mask_right):
    # Initialize the RGBA image for transparency
    height, width = mask_left.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

    # Find contours of the left and right kidney masks
    mask_left_uint8 = (mask_left * 255).astype(np.uint8)
    mask_right_uint8 = (mask_right * 255).astype(np.uint8)

    left_contours, _ = cv2.findContours(mask_left_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_contours, _ = cv2.findContours(mask_right_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    cv2.drawContours(rgba_image, left_contours, -1, (255, 0, 0), 1)
    cv2.drawContours(rgba_image, right_contours, -1, (255, 0, 0), 1)

    rgba_image[:, :, 3] = rgba_image[:, :, 0]

    return rgba_image


def save_png(image, bucket: str, supabase_client):
    img = Image.fromarray(image)

    image_io = save_image_to_bytes(img)

    # Upload to Supabase storage
    storage_id = uuid.uuid4()
    file_name = f"{storage_id}.png"

    response = supabase_client.storage.from_(bucket).upload(file_name, image_io.getvalue(),
                                                            file_options={'content-type': 'image/png'})
    return response.path


def extract_curve_features(curve):
    """Extract interpretable features from a single activity curve."""
    time_to_peak = np.argmax(curve)  # Index of the peak
    peak_value = np.max(curve)  # Peak activity level
    auc = np.sum(curve)  # Total activity (area under the curve)
    rising_slope = peak_value / (time_to_peak + 1)  # Avoid division by zero
    recovery_time = len(curve) - time_to_peak  # Time from peak to end
    return [time_to_peak, peak_value, auc, rising_slope, recovery_time]


def extract_statistical_features(curve):
    """Extract statistical features from a single activity curve."""
    mean_value = np.mean(curve)  # Mean
    variance = np.var(curve)  # Variance
    skewness = skew(curve)  # Skewness (asymmetry)
    kurt = kurtosis(curve)  # Kurtosis (tailedness)
    return [mean_value, variance, skewness, kurt]


def process_renograms_total(X):
    """Extract features for the total kidney curve."""
    total_features = np.array([extract_curve_features(x) for x in X])  # Use the whole curve
    return total_features


def process_statistical_features_total(X):
    """Extract statistical features for the total kidney curve."""
    total_features = np.array([extract_statistical_features(x) for x in X])  # Use the whole curve
    return total_features


def combine_features_total(X):
    """Combine temporal and statistical features for the total kidney curve."""
    # Extract temporal features
    temporal_features = process_renograms_total(X)

    # Extract statistical features
    statistical_features = process_statistical_features_total(X)

    # Combine the two sets of features
    return np.hstack((temporal_features, statistical_features))

def run_single_classification_dt(extracted_features):
    script_dir = os.path.dirname(__file__)
    dt_model_path = os.path.join(script_dir, "../../models/decision_tree/decision_tree_model.joblib")
    dt_model = joblib.load(dt_model_path)  # Load the entire model

    probabilities = dt_model.predict_proba(extracted_features)[0]
    predicted_class = dt_model.predict(extracted_features)

    predicted_class = int(predicted_class)

    return predicted_class, probabilities, dt_model


def remove_think_tags_deepseek(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def interpret_shap_feature(name, shap_val, value):
    """Provide a medically relevant interpretation of SHAP values for each feature."""

    if name == "Time to Peak":
        return (f"A **Time to Peak** of {value:.1f} frames means the kidney took this long to reach maximum uptake. "
                f"A higher SHAP value ({shap_val:.3f}) suggests the model considers this an important factor in the classification. "
                "Delayed Time to Peak may indicate impaired renal function, while a lower value suggests normal or hyperactive function.")

    elif name == "Peak Value":
        return (f"A **Peak Value** of {value:.3f} represents the highest tracer uptake in the renogram. "
                f"The SHAP impact ({shap_val:.3f}) suggests its role in the prediction. "
                "A lower peak might indicate poor renal perfusion, while a higher peak suggests strong uptake.")

    elif name == "AUC":
        return (f"The **Area Under the Curve (AUC)** is {value:.1f}, representing the total tracer activity over time. "
                f"The model's SHAP impact ({shap_val:.3f}) indicates its influence. "
                "Lower AUC may suggest reduced kidney function, while higher values imply better tracer retention.")

    elif name == "Rising Slope":
        return (
            f"A **Rising Slope** of {value:.3f} (peak value divided by time-to-peak) indicates how quickly the kidney reaches maximum uptake. "
            f"A SHAP impact of {shap_val:.3f} suggests its significance in the decision. "
            "A steeper slope may reflect good function, while a lower slope might indicate delayed uptake.")

    elif name == "Recovery Time":
        return (
            f"The **Recovery Time** is {value:.1f} frames, representing how long the kidney takes to clear the tracer after peak uptake. "
            f"With a SHAP impact of {shap_val:.3f}, this feature affects the model's prediction. "
            "A prolonged recovery may indicate slow clearance and impaired renal function.")

    elif name == "Mean":
        return (
            f"The **Mean Activity Level** is {value:.3f}, showing the average tracer uptake throughout the renogram. "
            f"A SHAP impact of {shap_val:.3f} suggests how much this factor matters. "
            "Lower values might indicate overall poor kidney uptake, while higher values suggest sustained function.")

    elif name == "Variance":
        return (f"The **Variance** of {value:.3f} represents fluctuations in tracer uptake. "
                f"With a SHAP value of {shap_val:.3f}, its role in the classification is noted. "
                "High variance suggests irregular kidney function, while low variance may reflect stable uptake.")

    elif name == "Skewness":
        return (f"A **Skewness** of {value:.3f} describes the asymmetry of uptake over time. "
                f"SHAP impact: {shap_val:.3f}. "
                "A positive skew indicates early uptake dominance, while a negative skew suggests delayed retention.")

    elif name == "Kurtosis":
        return (f"A **Kurtosis** of {value:.3f} represents how peaked or flat the tracer distribution is. "
                f"SHAP impact: {shap_val:.3f}. "
                "Higher kurtosis suggests sharp peaks in uptake, which might indicate abnormal renal behavior.")

    else:
        return "Feature impact needs further interpretation."


def generate_textual_shap_explanation(shap_values, feature_names, predicted_label, confidence):

    # Extract necessary information
    shap_contributions = shap_values.values[0].tolist() # SHAP values for the current instance

    feature_values = shap_values.data[0].tolist()  # Extracted feature values

    feature_importance = sorted(
        zip(feature_names, shap_contributions, feature_values), key=lambda x: abs(x[1]), reverse=True
    )

    # Identify top features
    top_features = feature_importance[:3]  # Taking the top 3 most impactful features

    # Format the features into a structured text prompt
    feature_text = "\n".join([
        f"- {interpret_shap_feature(name, shap_val, value)}"
        for name, shap_val, value in top_features
    ])

    print("Feature text", feature_text)

    PROMPT = f"""
    The model predicts that the patient is {predicted_label} with a confidence of {confidence:.2f}.

    The decision was based on the following key factors:

    {feature_text}

    Based on these factors, provide a structured explanation in **clear and readable plain text**.

    - Do NOT use any special formatting like asterisks (*), markdown symbols, or bullet points.
    - The response should be in full sentences and structured like a professional medical explanation.
    - Keep it concise but informative.
    - Do not include disclaimers.

    Output Example:

    "The model classifies the patient as [healthy/sick] due to [brief reason]. The most important factors were [Factor 1], [Factor 2], and [Factor 3]. [Explain what each factor means in relation to kidney function]. Overall, these indicators suggest that [summary of model reasoning]."
    """

    MODEL = "deepseek-r1:7b"

    result = ollama.generate(model=MODEL, prompt=PROMPT)
    response = result["response"]

    textual_explanation = remove_think_tags_deepseek(response)

    print("Textual explanation", textual_explanation)

    return textual_explanation








def perform_decision_tree_analysis(total_activities):

    total_activities = np.array(total_activities).reshape(1, -1)

    print("total_activities", total_activities.shape)

    extracted_features = combine_features_total(total_activities)  # Shape: (1, num_features)

    #print("extracted features shape", extracted_features.shape)
    #print("extracted features", extracted_features)


    pred, prob, dt_model = run_single_classification_dt(extracted_features)

    script_dir = os.path.dirname(__file__)
    training_data_path = os.path.join(script_dir, "../../models/decision_tree/decision_tree_training_data.npy")
    X_train_sample = np.load(training_data_path, allow_pickle=True)

    if X_train_sample.shape[1] != extracted_features.shape[1]:
        raise ValueError(
            f"Feature size mismatch! Expected {X_train_sample.shape[1]}, got {extracted_features.shape[1]}")

    #Explain the prediction
    explainer = shap.Explainer(dt_model, X_train_sample)
    shap_values = explainer(extracted_features)

    shap_values = shap_values[..., pred]  # Extract SHAP values for the predicted class

    shap_explanation_list = shap_values.values[0].tolist()  # Convert NumPy array to a Python list

    feature_names = ["Time to Peak", "Peak Value", "AUC", "Rising Slope", "Recovery Time", "Mean", "Variance", "Skewness", "Kurtosis"]

    predicted_label = "healthy" if pred == 0 else "sick"

    textual_explanation = generate_textual_shap_explanation(shap_values, feature_names, predicted_label, prob[pred])

    print("textual_explanation", textual_explanation)

    return pred, prob, shap_explanation_list, textual_explanation




