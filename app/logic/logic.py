import uuid
import cv2
import pydicom
import numpy as np
import io
import torch
import os
import joblib
from PIL import Image
from scipy.stats import skew, kurtosis
import shap
import google.generativeai as genai
from app.logic.CNN import Simple3DCNN

feature_maps = None
gradients = None


def fetch_model_from_supabase(supabase_client, filename, is_cnn_model=False, is_unet_model=False):
    response = supabase_client.storage.from_("ai-models").download(filename)

    if response:
        model_bytes = io.BytesIO(response)
        if is_cnn_model:
            model = Simple3DCNN()
            model.load_state_dict(torch.load(model_bytes, map_location=torch.device("cpu"), weights_only=False))
            model.eval()
            print(f"{filename} (Torch Model) loaded successfully.")

            return model
        if is_unet_model:
            model = torch.load(model_bytes, map_location=torch.device('cpu'), weights_only=False)
            model.eval()
            print(f"{filename} (Torch Model) loaded successfully.")

            return model
        else:
            model = joblib.load(model_bytes)
            print(f"{filename} (Joblib Model/Scaler) loaded successfully.")
            return model
    else:
        raise RuntimeError("Failed to fetch model from Supabase. Check permissions or file existence.")


def load_training_data_supabase(supabase_client, filename):
    response = supabase_client.storage.from_("ai-models").download(filename)

    if response:
        np_bytes = io.BytesIO(response)
        np_bytes.seek(0)
        training_data = np.load(np_bytes, allow_pickle=True)
        print(f"{filename} (NumPy Array) loaded successfully.")

        return training_data
    else:
        raise RuntimeError(f"Failed to fetch {filename} from Supabase. Check permissions or file existence.")


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


def create_composite_image(pixel_array, dicom_image=None):
    # Generate a composite image (e.g., by averaging)
    if dicom_image:
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


def create_and_overlay_heatmaps(dicom_read):
    composite_images = []
    heatmaps = []
    overlayed_images = []

    pixel_array = dicom_read.pixel_array.astype(np.float32)
    gradcam = generate_gradcam()

    for i in range(pixel_array.shape[0]):
        if i % 12 == 0:
            group_frames_composite = pixel_array[i:i + 12]
            composite_image = create_composite_image(pixel_array=group_frames_composite)
            composite_images.append(composite_image)

        if i % 12 == 0:
            gradcam_index = i // 4
            if gradcam_index < gradcam.shape[0]:
                group_frames_heatmap = gradcam[gradcam_index:gradcam_index + 3]
                heatmap = aggregate_heatmap(group_frames_heatmap)
                heatmaps.append(heatmap)

    # Overlay heatmaps on composite images
    for composite_image, heatmap in zip(composite_images, heatmaps):
        overlay_image = overlay_heatmap(composite_image, heatmap)
        overlayed_images.append(overlay_image)

    return overlayed_images


def run_single_classification_cnn(dicom_read, cnn_model):
    img = dicom_read.pixel_array.astype(np.float32)
    img /= np.max(img)  # Normalize
    volume_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    cnn_model.conv3.register_forward_hook(forward_hook)
    cnn_model.conv3.register_backward_hook(backward_hook)

    output = cnn_model(volume_tensor)
    predicted = torch.argmax(output, dim=1)

    predicted_class = predicted.item()

    cnn_model.zero_grad()

    # Perform backward pass for the target class
    target_class = predicted_class  # Example target class
    output[:, target_class].backward()

    print(f"Predicted class: {predicted_class}")
    print("Output probabilities:", output.flatten())

    print("output", output)

    confidence = output.flatten().tolist()[predicted_class]

    print(confidence)

    return predicted_class, confidence


def run_single_classification_svm(roi_activity_array, svm_model, svm_scaler):
    roi_activity_array = np.array(roi_activity_array).reshape(1, -1)

    scaled_data = svm_scaler.transform(roi_activity_array)

    print("scaled roi activity", scaled_data.shape)

    probabilities = svm_model.predict_proba(scaled_data)[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    return int(predicted_class), float(confidence)


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


def predict_kidney_masks(composite_image, unet_model):

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


def align_masks_over_summed_frames(left_kidney_mask, right_kidney_mask, dicom_read, frame_interval_seconds=10,
                                   sum_duration_minutes=3):
    left_mask_alignments_summed = []
    right_mask_alignments_summed = []

    global_max = 0  # Initialize global max for summed frames

    # First Pass: Find Global Max
    num_frames = dicom_read.NumberOfFrames
    frames_per_sum = int((sum_duration_minutes * 60) / frame_interval_seconds)

    for start_idx in range(0, num_frames, frames_per_sum):
        end_idx = min(start_idx + frames_per_sum, num_frames)
        summed_frame = np.sum(dicom_read.pixel_array[start_idx:end_idx], axis=0).astype(np.uint16)
        global_max = max(global_max, np.max(summed_frame))  # Update global max

    if global_max == 0:
        print("Global max is zero, skipping processing to avoid division by zero.")
        return

    for start_idx in range(0, num_frames, frames_per_sum):
        end_idx = min(start_idx + frames_per_sum, num_frames)
        summed_frame = np.sum(dicom_read.pixel_array[start_idx:end_idx], axis=0).astype(np.uint16)

        normalized_summed_frame = (summed_frame / global_max * 255).astype(np.uint8)

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
                                             (normalized_summed_frame.shape[1], normalized_summed_frame.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)

        left_masked_frame = cv2.bitwise_and(normalized_summed_frame, normalized_summed_frame,
                                            mask=resized_left_kidney_roi)

        # Resize and apply the right kidney mask
        resized_right_kidney_roi = cv2.resize(right_kidney_mask,
                                              (normalized_summed_frame.shape[1], normalized_summed_frame.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
        right_masked_frame = cv2.bitwise_and(normalized_summed_frame, normalized_summed_frame,
                                             mask=resized_right_kidney_roi)

        left_mask_alignments_summed.append(left_masked_frame)
        right_mask_alignments_summed.append(right_masked_frame)

    return left_mask_alignments_summed, right_mask_alignments_summed


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


def calculate_shap_data(model, training_data, explainer_data, prediction, svm_scaler):

    explainer_data = np.array(explainer_data).reshape(1, -1)

    # Scale the explainer_data to match what is used in Google Colab
    scaled_explainer_data = svm_scaler.transform(explainer_data)

    explainer = shap.KernelExplainer(model.predict_proba, training_data)

    shap_values = explainer(scaled_explainer_data)

    shap_values = shap_values[..., prediction]

    feature_values_raw = svm_scaler.inverse_transform(scaled_explainer_data)[0]  # Get original feature values

    shap_values_list = np.array(shap_values.values[0]).tolist()
    feature_values_list = np.array(feature_values_raw).tolist()
    base_values_list = np.full_like(shap_values_list, shap_values.base_values[0]).tolist()

    shap_data = [shap_values_list, feature_values_list, base_values_list]  # Explicit conversion

    return shap_data


def perform_svm_analysis(dicom_read, svm_model, svm_scaler, svm_training_data, unet_model):
    # Create composite image of the request file
    composite_image = create_composite_image_rgb(dicom_read)

    # Predict kidney masks
    left_mask, right_mask = predict_kidney_masks(composite_image, unet_model)

    # align masks over all frames in the original dicom file
    left_mask_alignments, right_mask_alignments = align_masks_over_frames(left_mask, right_mask, dicom_read)
    left_mask_alignments_summed, right_mask_alignments_summed = align_masks_over_summed_frames(left_mask, right_mask,
                                                                                               dicom_read)

    # Create renogram over all 180 frames from the predicted masks
    left_activities, right_activities, total_activities = create_renogram(left_mask_alignments, right_mask_alignments)
    roi_activity_array = [left_activities.tolist(), right_activities.tolist(), total_activities.tolist()]

    # Create renogram over 3 min summed frames from the predicted masks
    _, _, total_activities_summed = create_renogram(left_mask_alignments_summed, right_mask_alignments_summed)

    # Predict CKD stage with SVM model
    prediction, confidence = run_single_classification_svm(total_activities_summed, svm_model, svm_scaler)

    shap_data = calculate_shap_data(svm_model, svm_training_data, total_activities_summed.tolist(), prediction, svm_scaler)

    time_groups = list(range(0, 30, 3))

    predicted_label = "healthy" if prediction == 0 else "sick"

    textual_explanation = generate_textual_shap_explanation_datapoints(
        shap_data=shap_data,
        time_groups=time_groups,
        predicted_label=predicted_label,
        confidence=confidence
    )

    return prediction, confidence, roi_activity_array, left_mask, right_mask, total_activities.tolist(), shap_data, textual_explanation


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


def save_composite_heatmaps(overlayed_images, sb_client):
    storage_paths = []

    try:
        for idx, overlayed_image in enumerate(overlayed_images):
            path = save_png(overlayed_image, 'heatmaps', sb_client)

            storage_paths.append(path)

        return storage_paths

    except Exception as e:
        raise RuntimeError(f"Error processing DICOM file: {str(e)}")


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

    storage_id = uuid.uuid4()
    file_name = f"{storage_id}.png"

    response = supabase_client.storage.from_(bucket).upload(file_name, image_io.getvalue(),
                                                            file_options={'content-type': 'image/png'})
    return response.path


def extract_statistical_features(X):
    return np.array([
        [np.mean(curve), np.var(curve), skew(curve), kurtosis(curve)]
        for curve in X
    ])


def run_single_classification_dt(extracted_features, dt_model):
    probabilities = dt_model.predict_proba(extracted_features)[0]
    predicted_class = dt_model.predict(extracted_features)

    confidence = probabilities[predicted_class]

    return int(predicted_class), float(confidence), dt_model


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


def generate_textual_shap_explanation_features(shap_values, feature_names, predicted_label, confidence):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Extract necessary information
    shap_contributions = shap_values.values[0].tolist()

    feature_values = shap_values.data[0].tolist()

    feature_importance = sorted(
        zip(feature_names, shap_contributions, feature_values), key=lambda x: abs(x[1]), reverse=True
    )

    # Identify top features
    top_features = feature_importance[:3]

    # Format the features into a structured text prompt
    feature_text = "\n".join([
        f"- {interpret_shap_feature(name, shap_val, value)}"
        for name, shap_val, value in top_features
    ])

    PROMPT = f"""
    The model predicts that the patient is {predicted_label} with a confidence of {confidence:.2f}.

    The decision was based on the following key factors:

    {feature_text}

    Based on these factors, provide a structured explanation in **markdown**.

    - The response should be in full sentences and structured like a professional medical explanation.
    - Keep it concise but informative.
    - Do not include disclaimers.

    Output Example:

    "The model classifies the patient as [healthy/sick] due to [brief reason]. The most important factors were [Factor 1], [Factor 2], and [Factor 3]. [Explain what each factor means in relation to kidney function]. Overall, these indicators suggest that [summary of model reasoning]."
    """

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(PROMPT)

    textual_explanation = response.text.strip() if hasattr(response, "text") else "No explanation generated."

    return textual_explanation


def interpret_shap_time_group(time_group, shap_value, activity_value):
    """
    Interprets a SHAP value associated with a 3-minute renogram time group.

    :param time_group: The starting minute of the 3-minute group (e.g., 0, 3, 6, ... 27)
    :param shap_value: The SHAP value (model impact at this time group)
    :param activity_value: The summed renogram activity for this group
    :return: A structured medical interpretation
    """

    impact = "strongly" if abs(shap_value) > 0.1 else "moderately"

    # General interpretation structure
    if shap_value > 0:
        influence_text = f"Increased tracer activity during this period {impact} influenced the model's decision, suggesting that retention patterns were relevant to the classification."
    else:
        influence_text = f"Lower tracer activity during this time frame {impact} contributed to the decision, indicating that reduced uptake or faster clearance was important for classification."

    return f"Between {time_group}-{time_group + 3} minutes, the summed activity was {activity_value:.2f}. {influence_text}"


def generate_textual_shap_explanation_datapoints(shap_data, time_groups, predicted_label, confidence):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    shap_values = np.array(shap_data[0])
    feature_values = np.array(shap_data[1])

    prompt = f"""
      You are given renogram data where feature values represent summed intensities over {time_groups}-minute intervals, and SHAP values indicate each interval's contribution to the model's prediction. Your task is to provide an accurate textual explanation of both the renogram trend and SHAP value impact, strictly based on the provided numerical data.

      - Describe the overall trend of the feature values over time **only in terms of increasing, decreasing, or fluctuating**, without assuming any biological meaning.
      - Explain the SHAP values accurately, noting whether they increase, decrease, or fluctuate over time, without making assumptions about their significance beyond their numerical contribution.
      - Do not assume a smooth or monotonic trend for either feature values or SHAP values unless explicitly reflected in the data.
      - Ensure the explanation is factually correct based on the provided values.
      - Present the explanation in *markdown* format.

      Feature values: {feature_values.tolist()}
      SHAP values: {shap_values.tolist()}
      Model Prediction: {predicted_label}
      Model Confidence: {confidence}
      """

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    textual_explanation = response.text.strip() if hasattr(response, "text") else "No explanation generated."

    return textual_explanation


def perform_decision_tree_analysis(total_activities, dt_model, dt_training_data):
    total_activities = np.array(total_activities).reshape(1, -1)

    extracted_features = extract_statistical_features(total_activities)

    prediction, confidence, dt_model = run_single_classification_dt(extracted_features, dt_model)

    explainer = shap.Explainer(dt_model, dt_training_data)
    shap_values = explainer(extracted_features)

    shap_values = shap_values[..., prediction]

    shap_values_list = np.array(shap_values.values[0]).tolist()
    feature_values_list = np.array(shap_values.data[0]).tolist()
    base_values_list = np.full_like(shap_values_list, shap_values.base_values[0]).tolist()

    shap_data = [shap_values_list, feature_values_list, base_values_list]

    feature_names = ["Mean", "Variance", "Skewness", "Kurtosis"]
    predicted_label = "healthy" if prediction == 0 else "sick"

    textual_explanation = generate_textual_shap_explanation_features(shap_values, feature_names, predicted_label,
                                                                     confidence)

    return prediction, confidence, shap_data, textual_explanation
