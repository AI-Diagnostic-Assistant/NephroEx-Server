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

def load_model(filename, is_cnn_model=False, is_unet_model=False):
    try:
        with open(filename, 'rb') as file:
            file_content = file.read()
    except FileNotFoundError:
        raise RuntimeError("Failed to load model. Check file existence.")

    model_bytes = io.BytesIO(file_content)

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


def calculate_aquasition_time(dicom_read):




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



def load_training_data_local(filename):
    try:
        training_data = np.load(filename, allow_pickle=True)
        print(f"{filename} (NumPy Array) loaded successfully.")
        return training_data
    except FileNotFoundError:
        raise RuntimeError(f"Failed to load {filename} from local directory. Check file existence.")


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

    confidence = output.flatten().tolist()[predicted_class]

    return predicted_class, confidence


def run_single_classification_datapoints(roi_activity_array, svm_model, scaler):
    roi_activity_array = np.array(roi_activity_array).reshape(1, -1)

    scaled_data = scaler.transform(roi_activity_array)

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


def create_renogram_raw(left_kidney_mask, right_kidney_mask, dicom_read):
    left_activities, right_activities, total_activities = [], [], []

    # Ensure masks are in proper format
    if left_kidney_mask.max() <= 1.0:
        left_kidney_mask = (left_kidney_mask * 255).astype(np.uint8)
    if right_kidney_mask.max() <= 1.0:
        right_kidney_mask = (right_kidney_mask * 255).astype(np.uint8)

    resized_left_kidney_roi = cv2.resize(left_kidney_mask,
                                         (dicom_read.pixel_array.shape[2], dicom_read.pixel_array.shape[1]),
                                         interpolation=cv2.INTER_LINEAR)

    # Resize and apply the right kidney mask
    resized_right_kidney_roi = cv2.resize(right_kidney_mask,
                                          (dicom_read.pixel_array.shape[2], dicom_read.pixel_array.shape[1]),
                                          interpolation=cv2.INTER_LINEAR)

    if hasattr(dicom_read, "PhaseInformationSequence"):
        phase_info = dicom_read.PhaseInformationSequence[0]
        frame_time_ms = float(phase_info.ActualFrameDuration)
    else:
        # Default to 10000 ms (10 seconds) if not available
        frame_time_ms = 10000

    frame_duration = frame_time_ms / 1000.0  # convert milliseconds to seconds

    for frame_idx in range(dicom_read.NumberOfFrames):
        pixel_array = dicom_read.pixel_array[frame_idx].astype(np.float32)

        left_masked_frame = cv2.bitwise_and(pixel_array, pixel_array, mask=resized_left_kidney_roi)
        right_masked_frame = cv2.bitwise_and(pixel_array, pixel_array, mask=resized_right_kidney_roi)

        # Compute activity
        left_activity = compute_activity(left_masked_frame, frame_duration)
        right_activity = compute_activity(right_masked_frame, frame_duration)

        left_activities.append(left_activity)
        right_activities.append(right_activity)

    return np.array(left_activities), np.array(right_activities)


def create_renogram_summed(left_kidney_mask, right_kidney_mask, dicom_read, sum_duration_minutes=2):
    left_activities, right_activities, total_activities = [], [], []

    if hasattr(dicom_read, "PhaseInformationSequence"):
        phase_info = dicom_read.PhaseInformationSequence[0]
        frame_time_ms = float(phase_info.ActualFrameDuration)
    else:
        frame_time_ms = 10000

    frame_duration = frame_time_ms / 1000.0  # convert milliseconds to seconds

    num_frames = dicom_read.NumberOfFrames
    frames_per_sum = int((sum_duration_minutes * 60) / frame_duration)

    # Ensure masks are in proper format
    if left_kidney_mask.max() <= 1.0:
        left_kidney_mask = (left_kidney_mask * 255).astype(np.uint8)
    if right_kidney_mask.max() <= 1.0:
        right_kidney_mask = (right_kidney_mask * 255).astype(np.uint8)

    # Resize kidney masks **once before the loop**
    resized_left_kidney_roi = cv2.resize(left_kidney_mask,
                                         (dicom_read.pixel_array.shape[2], dicom_read.pixel_array.shape[1]),
                                         interpolation=cv2.INTER_LINEAR)
    resized_right_kidney_roi = cv2.resize(right_kidney_mask,
                                          (dicom_read.pixel_array.shape[2], dicom_read.pixel_array.shape[1]),
                                          interpolation=cv2.INTER_LINEAR)

    for start_idx in range(0, num_frames, frames_per_sum):
        end_idx = min(start_idx + frames_per_sum, num_frames)

        summed_pixel_array = np.sum(dicom_read.pixel_array[start_idx:end_idx], axis=0).astype(np.uint16)

        left_masked_frame = cv2.bitwise_and(summed_pixel_array, summed_pixel_array, mask=resized_left_kidney_roi)
        right_masked_frame = cv2.bitwise_and(summed_pixel_array, summed_pixel_array, mask=resized_right_kidney_roi)

        left_activity = compute_activity(left_masked_frame, frame_duration)
        right_activity = compute_activity(right_masked_frame, frame_duration)

        # Store activities
        left_activities.append(left_activity)
        right_activities.append(right_activity)

    return np.array(left_activities), np.array(right_activities)


def compute_activity(masked_array, frame_duration):
    """Compute counts per second by dividing the sum of counts by the frame duration (in seconds)."""
    total_counts = np.sum(masked_array)
    return total_counts / frame_duration


def calculate_shap_data_datapoints(model, training_data, explainer_data, prediction, svm_scaler):
    explainer_data = np.array(explainer_data).reshape(1, -1)

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


def calculate_shap_data_features(model, training_data, explainer_data, classification):
    explainer = shap.Explainer(model, training_data)

    print("classification", classification)

    print("explainer_data", explainer_data)

    shap_values = explainer(explainer_data)

    print("shap_values", shap_values)

    shap_values = shap_values[..., classification]

    shap_values_list = np.array(shap_values.values[0]).tolist()
    feature_values_list = np.array(shap_values.data[0]).tolist()

    base_values_list = np.full_like(shap_values_list, shap_values.base_values[0]).tolist()

    shap_data = [shap_values_list, feature_values_list, base_values_list]

    return shap_data, shap_values


def perform_datapoints_analysis(svm_model, svm_scaler, svm_training_data, left_activities_summed,
                                right_activities_summed):

    # Pad sequences to fixed length
    left_activities_summed = pad_or_truncate_sequence(left_activities_summed, fixed_length=20, padding_value=0)
    right_activities_summed = pad_or_truncate_sequence(right_activities_summed, fixed_length=20, padding_value=0)

    # Classify UTO for left and right kidneys
    left_uto_classification, left_uto_confidence = run_single_classification_datapoints(left_activities_summed,
                                                                                        svm_model, svm_scaler)
    right_uto_classification, right_uto_confidence = run_single_classification_datapoints(right_activities_summed,
                                                                                          svm_model, svm_scaler)

    shap_data_left_uto_classification = calculate_shap_data_datapoints(svm_model, svm_training_data,
                                                                       left_activities_summed.tolist(),
                                                                       left_uto_classification, svm_scaler)
    shap_data_right_uto_classification = calculate_shap_data_datapoints(svm_model, svm_training_data,
                                                                        right_activities_summed.tolist(),
                                                                        right_uto_classification, svm_scaler)

    time_groups = list(range(0, 40, 2))

    classified_left_label = "healthy" if left_uto_classification == 0 else "sick"
    classified_right_label = "healthy" if right_uto_classification == 0 else "sick"

    left_textual_explanation = generate_single_textual_shap_explanation_datapoints(
        shap_data=shap_data_left_uto_classification,
        time_groups=time_groups,
        classified_label=classified_left_label,
        confidence=left_uto_confidence,
        kidney_label="left"
    )

    right_textual_explanation = generate_single_textual_shap_explanation_datapoints(
        shap_data=shap_data_right_uto_classification,
        time_groups=time_groups,
        classified_label=classified_right_label,
        confidence=right_uto_confidence,
        kidney_label="right"
    )

    return left_uto_classification, right_uto_classification, left_uto_confidence, right_uto_confidence, shap_data_left_uto_classification, shap_data_right_uto_classification, left_textual_explanation, right_textual_explanation, classified_left_label, classified_right_label


def group_2_min_frames(dicom_read):
    frames = dicom_read.pixel_array

    # Group into 15 summed frames
    grouped_frames = []
    for i in range(0, len(frames), 12):
        group = frames[i:i + 12]
        summed_frame = np.sum(group, axis=0)
        grouped_frames.append(summed_frame)

    return grouped_frames


def group_2_min_frames_and_save_to_storage(dicom_read, sb_client):
    storage_ids = []

    grouped_2_min_frames = group_2_min_frames(dicom_read)

    try:
        for idx, frame in enumerate(grouped_2_min_frames):
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


def pad_or_truncate_sequence(seq, fixed_length=20, padding_value=0):
    seq_length = seq.shape[0]
    if seq_length < fixed_length:
        pad_length = fixed_length - seq_length
        extra_padding = np.full((pad_length,), padding_value, dtype=seq.dtype)
        seq = np.concatenate([seq, extra_padding])
    elif seq_length > fixed_length:
        seq = seq[:fixed_length]
    return seq


def save_png(image, bucket: str, supabase_client):
    img = Image.fromarray(image)

    image_io = save_image_to_bytes(img)

    storage_id = uuid.uuid4()
    file_name = f"{storage_id}.png"

    response = supabase_client.storage.from_(bucket).upload(file_name, image_io.getvalue(),
                                                            file_options={'content-type': 'image/png'})
    return response.path

def extract_quantitative_features(curve, frame_interval=10, diuretic_time=20):

    if isinstance(curve, torch.Tensor):
        curve = curve.cpu().detach().numpy()

    # Basic statistical features
    mean_val = np.mean(curve)
    var_val = np.var(curve)
    skew_val = skew(curve)
    kurt_val = kurtosis(curve)

    print("curve", curve)

    # Compute time vector (in seconds)
    n_frames = len(curve)

    print("n_frames", n_frames)

    time_vector = np.arange(n_frames) * (frame_interval / 60.0)

    print("time_vector", time_vector)

    # Time-to-Peak: time at which maximum count occurs
    peak_index = np.argmax(curve)

    print('peak_index', peak_index)

    time_to_peak = time_vector[peak_index]

    print('time_to_peak', time_to_peak)

    peak_count = curve[peak_index]

    print('peak_count', peak_count)

    injection_frame = int(diuretic_time / (frame_interval / 60 ))

    # 20-min Count Ratio: count at 20 minutes relative to the peak count.
    #if injection_frame < n_frames:
      #  count_20min = curve[injection_frame]
      #  ratio_20min = count_20min / peak_count if peak_count != 0 else np.nan
    #else:
       # ratio_20min = np.nan

    # Baseline Half-Time (T½ Pre-Furosemide):
    # Look for the time (after the peak) where the count falls to half the peak,
    # but only consider frames before the injection.
    baseline_half_time = np.nan
    if peak_index < injection_frame:
        for i in range(peak_index, min(injection_frame, n_frames)):
            if curve[i] <= peak_count / 2:
                baseline_half_time = time_vector[i] - time_to_peak
                break


    # Get the count at injection time
    injection_count = curve[injection_frame]

    # Diuretic Half-Time (T½ Post-Furosemide based on injection count)
    diuretic_half_time = np.nan

    if injection_frame < n_frames:
      # Threshold = half of the injection-time count
      half_threshold = injection_count / 2.0

      for i in range(injection_frame, n_frames):
          if curve[i] <= half_threshold:
              # Time from injection (in seconds) to the frame i
              diuretic_half_time = time_vector[i] - diuretic_time
              break


    # Additional Feature: 30 min/Peak Ratio
    frame_30min = int((30 * 60) / frame_interval)
    if frame_30min < n_frames:
        count_30min = curve[frame_30min]
        ratio_30min = count_30min / peak_count if peak_count != 0 else np.nan
    else:
        ratio_30min = np.nan

    # Additional Feature: 30 min/3 min Ratio
    frame_3min = int((3 * 60) / frame_interval)
    if frame_30min < n_frames and frame_3min < n_frames:
        count_3min = curve[frame_3min]
        ratio_30_3 = count_30min / count_3min if count_3min != 0 else np.nan
    else:
        ratio_30_3 = np.nan


    baseline_half_time = -1 if np.isnan(baseline_half_time) else baseline_half_time
    diuretic_half_time = -1 if np.isnan(diuretic_half_time) else diuretic_half_time
    ratio_30min = -1 if np.isnan(ratio_30min) else ratio_30min
    ratio_30_3 = -1 if np.isnan(ratio_30_3) else ratio_30_3

    features = {
        "mean_val": mean_val,
        "var_val": var_val,
        "skew_val": skew_val,
        "kurt_val": kurt_val,
        "time_to_peak": time_to_peak,
        "baseline_half_time": baseline_half_time,
        "diuretic_half_time": diuretic_half_time,
        "ratio_30min": ratio_30min,
        "ratio_30_3": ratio_30_3
    }

    print(features)

    return [mean_val, var_val, skew_val, kurt_val,
            time_to_peak, baseline_half_time, diuretic_half_time,
            ratio_30min, ratio_30_3]


def run_single_uto_classification_features(extracted_features, dt_model):

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

    # New features
    elif name == "Baseline Half-Time":
        return (
            f"A **Baseline Half-Time** of {value:.1f} frames indicates the time it takes for the baseline tracer uptake to reduce by half. "
            f"The SHAP impact of {shap_val:.3f} suggests this feature is influential. "
            "A prolonged half-time may reflect delayed clearance, whereas a shorter half-time indicates prompt processing."
        )

    elif name == "Diuretic Half-Time":
        return (
            f"A **Diuretic Half-Time** of {value:.1f} frames represents the time after diuretic administration for the tracer activity to decrease by half. "
            f"With a SHAP impact of {shap_val:.3f}, this factor contributes to the model's prediction. "
            "A longer half-time might indicate a sluggish diuretic response, while a shorter half-time suggests efficient tracer clearance."
        )

    elif name == "Ratio 30min/Peak":
        return (
            f"A **Ratio 30min/Peak** value of {value:.3f} compares the tracer uptake at 30 minutes to the peak uptake value. "
            f"The SHAP impact of {shap_val:.3f} underscores its role in the decision process. "
            "A lower ratio may suggest rapid tracer clearance, whereas a higher ratio indicates sustained uptake over time."
        )

    elif name == "Ratio 30min/3min":
        return (
            f"A **Ratio 30min/3min** value of {value:.3f} contrasts the tracer uptake at 30 minutes with that at 3 minutes. "
            f"With a SHAP impact of {shap_val:.3f}, this feature helps differentiate uptake patterns. "
            "A higher ratio may indicate prolonged tracer retention, while a lower ratio points to a quicker clearance."
        )

    else:
        return "Feature impact needs further interpretation."


def generate_single_textual_shap_explanation_features(shap_values, predicted_label, confidence, kidney_label):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    feature_names = ["Mean", "Variance", "Skewness", "Kurtosis", "Time to Peak", "Baseline Half-Time", "Diuretic Half-Time", "Ratio 30min/Peak", "Ratio 30min/3min"]

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
    The model classifies that the {kidney_label} kidney as {predicted_label} with a confidence of {confidence:.2f}.

    The decision was based on the following key factors:

    {feature_text}

    Based on these factors, provide a structured explanation in **markdown**.

    - The response should be in full sentences and structured like a professional medical explanation.
    - Keep it concise but informative.
    - Do not include disclaimers.

    Output Example:

    "The model classifies the patient as [healthy/sick] due to [brief reason]. The most important factors were [Factor 1], [Factor 2], and [Factor 3]. [Explain what each factor means in relation to kidney function]. Overall, these indicators suggest that [summary of model reasoning]."
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(PROMPT)

    textual_explanation = response.text.strip() if hasattr(response, "text") else "No explanation generated."

    return textual_explanation


def generate_single_textual_shap_explanation_datapoints(shap_data, time_groups, classified_label, confidence,
                                                        kidney_label):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    shap_values = np.array(shap_data[0])
    feature_values = np.array(shap_data[1])

    prompt = f"""
      You are given renogram data for the {kidney_label} kidney, where feature values represent summed intensities over {time_groups}-minute intervals, and SHAP values indicate each interval's contribution to the model's prediction.
      
      - Feature values: {feature_values.tolist()}
      - SHAP values: {shap_values.tolist()}
      - Model Prediction: {classified_label}
      - Model Confidence: {confidence}
      
      Provide a textual explanation that:
        - Describes the overall trend of the feature values over time for the {kidney_label} (increasing, decreasing, or fluctuating).
        - Explains the trend of the SHAP values in similar terms.
        - Presents the explanation in **markdown** format, but **do not** use code blocks (no triple backticks).
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    textual_explanation = response.text.strip() if hasattr(response, "text") else "No explanation generated."

    return textual_explanation


def perform_features_analysis(left_activities, right_activities, dt_model, dt_training_data, diuretic_time):
    left_activities = np.array(left_activities).reshape(1, -1)
    right_activities = np.array(right_activities).reshape(1, -1)

    extracted_features_left = extract_quantitative_features(left_activities[0], 10, int(diuretic_time))
    extracted_features_right = extract_quantitative_features(right_activities[0], 10, int(diuretic_time))

    extracted_features_left = np.array(extracted_features_left).reshape(1, -1)
    extracted_features_right = np.array(extracted_features_right).reshape(1, -1)

    left_uto_classification, left_uto_confidence, dt_model = run_single_uto_classification_features(
        extracted_features_left, dt_model)
    right_uto_classification, right_uto_confidence, dt_model = run_single_uto_classification_features(
        extracted_features_right, dt_model)

    classified_left_label = "healthy" if left_uto_classification == 0 else "sick"
    classified_right_label = "healthy" if right_uto_classification == 0 else "sick"

    left_shap_data, left_shap_values = calculate_shap_data_features(dt_model, dt_training_data, extracted_features_left,
                                                                    left_uto_classification)
    right_shap_data, right_shap_values = calculate_shap_data_features(dt_model, dt_training_data,
                                                                      extracted_features_right,
                                                                      right_uto_classification)

    textual_explanation_left = generate_single_textual_shap_explanation_features(left_shap_values,
                                                                                 classified_left_label,
                                                                                 left_uto_confidence,
                                                                                 kidney_label="left")
    textual_explanation_right = generate_single_textual_shap_explanation_features(right_shap_values,
                                                                                  classified_right_label,
                                                                                  right_uto_confidence,
                                                                                  kidney_label="right")

    return left_uto_classification, right_uto_classification, left_uto_confidence, right_uto_confidence, left_shap_data, right_shap_data, textual_explanation_left, textual_explanation_right, classified_left_label, classified_right_label
