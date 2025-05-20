import uuid
import cv2
import pandas as pd
import pydicom
import numpy as np
import io
import torch
import os
import joblib
from PIL import Image
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
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


def run_single_classification_datapoints(roi_activity_array, model):

    roi_activity_array = np.array(roi_activity_array).reshape(1, -1)

    probabilities = model.predict_proba(roi_activity_array)[0]
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


def create_uptake_composite_image(ds, target_size=(128, 128), uptake_window=(120, 180)):
    arr = ds.pixel_array  # shape = (n_frames, H, W)
    n_frames = arr.shape[0]

    # 3) build the durations list from PhaseInformationSequence
    try:
        phases = ds.PhaseInformationSequence
    except AttributeError:
        print("⚠️ DICOM has no PhaseInformationSequence; can't build time axis")
        return None

    durations_ms = []
    for item in phases:
        dur = float(item.ActualFrameDuration)  # in ms
        count = int(item.NumberOfFramesInPhase)
        durations_ms.extend([dur] * count)
    # trim or pad to exactly n_frames
    durations_ms = (durations_ms + durations_ms)[: n_frames]

    # 4) cumulative times (ms) and convert to seconds
    cum_ms = np.cumsum(durations_ms)
    # make the first frame t=0 rather than t=durations_ms[0]
    rel_s = (cum_ms - durations_ms[0]) / 1000.0  # array length = n_frames

    # 5) pick only frames in [120,180) seconds
    idxs = np.where((rel_s >= uptake_window[0]) &
                    (rel_s < uptake_window[1]))[0]
    if idxs.size == 0:
        print(f"⚠️ No frames in {uptake_window[0]}–{uptake_window[1]}  s")
        return None

    # 6) sum them
    comp = np.sum(arr[idxs, ...], axis=0)  # (H, W)

    # 7) normalize safely
    mx = comp.max()
    if mx > 0:
        comp = comp / mx
    else:
        print(f"⚠️ Uptake composite all zeros")
        return None
    comp = np.nan_to_num(comp, nan=0.0)

    # 8) resize if needed
    if comp.shape != target_size:
        comp = cv2.resize(
            comp,
            dsize=target_size[::-1],  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        comp -= comp.min()
        m2 = comp.max()
        if m2 > 0:
            comp = comp / m2

    return comp.astype(np.float32)


def load_image(path: str):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    img /= np.max(img)

    return img


def predict_kidney_masks(composite_image, unet_model):
    # composite_image could be a numpy array (H,W) or (H,W,3)
    img = np.array(composite_image)

    # if grayscale (2D), replicate into 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # if shape is (3,H,W) (i.e. already channel-first), convert to HWC
    # (you can skip this if you know your array is always HxWxC)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    image_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        pred_masks = unet_model(image_tensor)

        pred_left_mask = (pred_masks[:, 0, :, :] > 0.5).cpu().numpy().squeeze()
        pred_right_mask = (pred_masks[:, 1, :, :] > 0.5).cpu().numpy().squeeze()

    return pred_left_mask, pred_right_mask


def trim_bad_by_fraction(left, right, times, frac_thresh=0.8, span=5):
    df = pd.DataFrame({"L": left, "R": right})
    # rolling means (center=False so only past frames)
    roll_L = df["L"].rolling(span, min_periods=1).mean()
    roll_R = df["R"].rolling(span, min_periods=1).mean()

    # drop until last frame is ≥ threshold × rolling mean
    while len(df) > 1:
        if (df["L"].iloc[-1] < frac_thresh * roll_L.iloc[-1]
            or df["R"].iloc[-1] < frac_thresh * roll_R.iloc[-1]):
            df = df.iloc[:-1]
            roll_L = roll_L.iloc[:-1]; roll_R = roll_R.iloc[:-1]
        else:
            break

    return df["L"].to_numpy(), df["R"].to_numpy(), times[: len(df)]



def cubic_smooth_renograms(curves, time_vector):
    smoothed_curves = []

    for seq in curves:
        # get a numpy array of the counts
        seq_np = seq.numpy() if torch.is_tensor(seq) else np.array(seq, dtype=float)
        # choose a smoothing factor (tweak this to taste)
        s = 0.005 * len(time_vector) * np.var(seq_np)
        spline = UnivariateSpline(time_vector, seq_np, s=s)
        # evaluate the smoothing spline back at the *original* time points
        smooth_seq = spline(time_vector)
        smoothed_curves.append(smooth_seq.tolist())

    plt.figure(figsize=(8, 4))
    plt.plot(time_vector, curves[0], '-', lw=1, alpha=0.6, label='Raw (noisy)')
    plt.plot(time_vector, smoothed_curves[0], '-', lw=3, label='Smoothed')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Activity')
    plt.title('Left Kidney Renogram — raw vs smoothed vs interpolated')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return smoothed_curves


def interpolate_renograms(smoothed_curves, time_vector, target_len=220):
    resampled_seqs = []

    new_t = np.linspace(time_vector[0],
                        time_vector[-1],
                        target_len)

    for smooth_seq in smoothed_curves:

        # build interpolant on original minutes-grid
        f = interp1d(time_vector, smooth_seq, kind='linear', fill_value='extrapolate')
        new_seq = f(new_t)
        resampled_seqs.append(new_seq.tolist())

    return new_t, resampled_seqs


def create_renogram_raw(left_kidney_mask, right_kidney_mask, ds):
    left_activities, right_activities = [], []
    time_s_list = []

    if hasattr(ds, "PhaseInformationSequence"):
        # Multi-phase: each phase entry has ActualFrameDuration + NumberOfFramesInPhase
        frame_times_ms = []
        for phase in ds.PhaseInformationSequence:
            dur_ms = float(phase.ActualFrameDuration)
            nfr = int(phase.NumberOfFramesInPhase)
            frame_times_ms += [dur_ms] * nfr
        frame_times_s = np.array(frame_times_ms, dtype=float) / 1000.0
    else:
        # Fallback: assume 10 s per frame
        frame_times_s = np.full(int(ds.NumberOfFrames), 10.0)

    time_s_list.extend(frame_times_s)

    # Ensure masks are in proper format
    if left_kidney_mask.max() <= 1.0:
        left_kidney_mask = (left_kidney_mask * 255).astype(np.uint8)
    if right_kidney_mask.max() <= 1.0:
        right_kidney_mask = (right_kidney_mask * 255).astype(np.uint8)

    resized_left_kidney_roi = cv2.resize(left_kidney_mask,
                                         (ds.pixel_array.shape[2], ds.pixel_array.shape[1]),
                                         interpolation=cv2.INTER_LINEAR)

    # Resize and apply the right kidney mask
    resized_right_kidney_roi = cv2.resize(right_kidney_mask,
                                          (ds.pixel_array.shape[2], ds.pixel_array.shape[1]),
                                          interpolation=cv2.INTER_LINEAR)

    for frame_idx in range(ds.NumberOfFrames):
        pixel_array = ds.pixel_array[frame_idx].astype(np.float32)
        dur = frame_times_s[frame_idx]

        left_masked_frame = cv2.bitwise_and(pixel_array, pixel_array, mask=resized_left_kidney_roi)
        right_masked_frame = cv2.bitwise_and(pixel_array, pixel_array, mask=resized_right_kidney_roi)

        # Compute activity
        left_activity = compute_activity(left_masked_frame, dur)
        right_activity = compute_activity(right_masked_frame, dur)

        left_activities.append(left_activity)
        right_activities.append(right_activity)

        # Cut of last frame if below treshold here
    times_arr = np.cumsum(np.array(time_s_list))
    left_arr = np.array(left_activities, dtype=float)
    right_arr = np.array(right_activities, dtype=float)

    # apply the drop‐last‐if‐bad to each kidney
    left_arr, right_arr, times_arr = trim_bad_by_fraction(left_arr, right_arr, times_arr, 0.8, 6)

    time_min = times_arr / 60.0

    # back to Python lists (if you need them)
    left_activities = left_arr.tolist()
    right_activities = right_arr.tolist()


    return np.array(left_activities), np.array(right_activities), time_min


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


def calculate_shap_data_datapoints(model, training_data, explainer_data, prediction, time_vector, bin_size_min=2):

    explainer_data = np.array(explainer_data).reshape(1, -1)

    explainer = shap.KernelExplainer(model.predict_proba, training_data)
    shap_mat = explainer.shap_values(explainer_data, nsamples=1000)
    sv_class = shap_mat[..., prediction]

    shap_vals = sv_class[0]  # shape (n_feats,)
    feat_vals = explainer_data[0]  # shape (n_feats,)
    base_val = float(explainer.expected_value[prediction])

    # 2) build bin edges & assign each feature to a bin
    edges = np.arange(time_vector.min(), time_vector.max()+bin_size_min, bin_size_min)
    bin_idx = np.digitize(time_vector, edges, right=False) - 1
    n_bins = len(edges) - 1

    # 3) aggregate into 2-min bins
    time_bins = []  # list of (start,end)
    binned_shap = []  # mean SHAP per bin
    binned_feature = []  # mean raw feature per bin

    for b in range(n_bins):
        mask = (bin_idx == b)
        time_bins.append((float(edges[b]), float(edges[b + 1])))
        if mask.any():
            binned_shap.append(float(shap_vals[mask].sum()))
            binned_feature.append(float(feat_vals[mask].sum()))
        else:
            binned_shap.append(0.0)
            binned_feature.append(0.0)


    base_values_list = [base_val] * len(binned_shap)

    shap_data = [binned_shap, binned_feature, base_values_list]

    return shap_data, time_bins


# def calculate_shap_data_features(model, training_data, explainer_data, classification):
#     explainer = shap.Explainer(model, training_data)
#     sv = explainer(explainer_data)
#     sv_class = sv[..., classification]
#
#     shap_values_list = np.array(sv_class.values[0]).tolist()
#     feature_values_list = np.array(sv_class.data[0]).tolist()
#
#     base_values_list = np.full_like(shap_values_list, sv_class.base_values[0]).tolist()
#
#     shap_data = [shap_values_list, feature_values_list, base_values_list]
#
#     return shap_data, sv

def calculate_shap_data_features(model, training_data, explainer_data, classification):
    explainer = shap.Explainer(model, training_data)

    shap_values = explainer(explainer_data)

    shap_values = shap_values[..., classification]

    shap_values_list = np.array(shap_values.values[0]).tolist()
    feature_values_list = np.array(shap_values.data[0]).tolist()

    base_values_list = np.full_like(shap_values_list, shap_values.base_values[0]).tolist()

    shap_data = [shap_values_list, feature_values_list, base_values_list]

    return shap_data, shap_values


def perform_datapoints_analysis(model, training_data,  smoothed_activity_array, interpolated_tv, diuretic_time):

    seq_left, seq_right = smoothed_activity_array

    left_tensor = torch.tensor(seq_left, dtype=torch.float32).unsqueeze(0)
    right_tensor = torch.tensor(seq_right, dtype=torch.float32).unsqueeze(0)

    left_uto_classification, left_uto_confidence = run_single_classification_datapoints(left_tensor, model)
    right_uto_classification, right_uto_confidence = run_single_classification_datapoints(right_tensor, model)

    shap_data_left, time_bins_left = calculate_shap_data_datapoints(model, training_data, seq_left, left_uto_classification, interpolated_tv)
    shap_data_right, time_bins_right = calculate_shap_data_datapoints(model, training_data, seq_right, right_uto_classification, interpolated_tv)

    classified_left_label = "healthy" if left_uto_classification == 0 else "sick"
    classified_right_label = "healthy" if right_uto_classification == 0 else "sick"

    left_textual_explanation = generate_single_textual_shap_explanation_datapoints(
        shap_data=shap_data_left,
        time_bins=time_bins_left,
        classified_label=classified_left_label,
        confidence=left_uto_confidence,
        kidney_label="left",
        diuretic_time=diuretic_time
    )

    right_textual_explanation = generate_single_textual_shap_explanation_datapoints(
        shap_data=shap_data_right,
        time_bins=time_bins_right,
        classified_label=classified_right_label,
        confidence=right_uto_confidence,
        kidney_label="right",
        diuretic_time=diuretic_time
    )

    return left_uto_classification, right_uto_classification, left_uto_confidence, right_uto_confidence, shap_data_left, shap_data_right, left_textual_explanation, right_textual_explanation, classified_left_label, classified_right_label, time_bins_left, time_bins_right


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

def extract_quantitative_features(curve, time_vector, diuretic_time=20):

    if isinstance(curve, torch.Tensor):
        curve = curve.cpu().detach().numpy()

    t = time_vector  # in minutes
    n = len(curve)

    # Basic statistical features
    mean = np.mean(curve)
    var = np.var(curve)
    skewness = skew(curve)
    kurt = kurtosis(curve)

    # determine if scan covers enough post-furo time
    end_time = t[-1]
    covers_5 = end_time >= diuretic_time + 5.0
    covers_20 = end_time >= diuretic_time + 20.0

    # find indices
    inj_idx = np.searchsorted(t, diuretic_time, side='right') - 1
    inj_ct = curve[inj_idx] if 0 <= inj_idx < n else np.nan
    f30_idx = np.searchsorted(t, 30.0, side='right') - 1
    post5_idx = np.searchsorted(t, diuretic_time + 5.0, side='right') - 1
    post15_idx = np.searchsorted(t, diuretic_time + 15.0, side='right') - 1
    post20_idx = np.searchsorted(t, diuretic_time + 20.0, side='right') - 1

    # C_last as a replacement for C30 due to variable length of sequences.
    # The idea is to catch the percent clearance at the last available time point
    last_idx = len(curve) - 1
    C_last = 100.0 * (curve[inj_idx] - curve[last_idx]) / curve[inj_idx]

    # mean slope during first 5 min post-furo
    if covers_5 and post5_idx > inj_idx:
        ds = np.diff(curve[inj_idx:post5_idx + 1])
        dt = np.diff(t[inj_idx:post5_idx + 1])
        slope_0_5 = np.mean(ds / dt)
    else:
        slope_0_5 = np.nan

    # mean slope between 15–20 min post-furo
    if covers_20 and post20_idx > post15_idx:
        ds = np.diff(curve[post15_idx:post20_idx + 1])
        dt = np.diff(t[post15_idx:post20_idx + 1])
        slope_15_20 = np.mean(ds / dt)
    else:
        slope_15_20 = np.nan

        # Curve length: arc-length from injection to end on smoothed curve
    ds_len = np.diff(curve[inj_idx:])
    dt_len = np.diff(t[inj_idx:])
    if len(ds_len) > 0:
        length = np.sum(np.sqrt(ds_len ** 2 + dt_len ** 2))
    else:
        length = np.nan

    peak_idx = np.argmax(curve)
    peak_ct = curve[peak_idx]
    ttp = t[peak_idx]

    # baseline half-time
    bh = np.nan
    if peak_idx < inj_idx:
        for i in range(peak_idx, inj_idx + 1):
            if curve[i] <= peak_ct / 2.0:
                bh = t[i] - ttp
                break

    # diuretic half-time
    dh = np.nan
    if 0 <= inj_idx < n and not np.isnan(inj_ct):
        thr = inj_ct / 2.0
        for i in range(inj_idx, n):
            if curve[i] <= thr:
                dh = t[i] - diuretic_time
                break

    # Now the two ratios, but only if t[-1] >= 30 min
    if t[-1] < 30.0:
        # scan too short → mark as missing
        ratio_30 = np.nan
        ratio_30_3 = np.nan
    else:
        # find the frame at (or just before) 30 min
        f30_idx = np.searchsorted(t, 30.0, side='right') - 1
        ratio_30 = curve[f30_idx] / peak_ct if peak_ct != 0 else np.nan

        # 3-min frame index
        f3_idx = np.searchsorted(t, 3.0, side='right') - 1
        if 0 <= f3_idx < n and curve[f3_idx] != 0:
            ratio_30_3 = curve[f30_idx] / curve[f3_idx]
        else:
            ratio_30_3 = np.nan


    features = {
        "mean_val": mean,
        "var_val": var,
        "skew_val": skewness,
        "kurt_val": kurt,
        "time_to_peak": ttp,
        "baseline_half_time": bh,
        "diuretic_half_time": dh,
        "ratio_30min": ratio_30,
        "ratio_30_3": ratio_30_3,
        "C_last": C_last,
        "slope_0_5": slope_0_5,
        "slope_15_20": slope_15_20
}

    def _fix(x): return -1 if np.isnan(x) else x


    return [mean, var, _fix(skewness), _fix(kurt),
        _fix(C_last), _fix(slope_0_5), _fix(slope_15_20), _fix(length), _fix(ttp), _fix(bh), _fix(dh),
        _fix(ratio_30), _fix(ratio_30_3)]


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

    elif name == "Split Function":
        return (
            f"The **Split Function** of {value:.1f}% represents the differential renal function measured during the uptake phase. "
            f"A SHAP impact of {shap_val:.3f} shows its weight in the model. "
            "An asymmetric split (far from 50:50) may indicate relative impairment in one kidney's function."
        )


    elif name == "Mean Slope 0-5min":
        return (
            f"A **Mean Slope 0–5 min** of {value:.3f} frames⁻¹ indicates the average clearance rate between 0 and 5 minutes after diuretic injection. "
            f"A SHAP impact of {shap_val:.3f} highlights its significance. "
            "A steeper slope in this early post-diuretic window suggests an efficient initial response to furosemide."
        )

    elif name == "Mean Slope 15-20min":
        return (
            f"A **Mean Slope 15–20 min** of {value:.3f} frames⁻¹ represents the average clearance rate between 15 and 20 minutes post-diuretic. "
            f"With a SHAP impact of {shap_val:.3f}, this parameter is weighted in the prediction. "
            "A persistently low slope here may point to ongoing drainage issues."
        )
    elif name == "C_last":
        return (
            f"A **C_last** value of {value:.1f}% represents the percent tracer clearance at the last frame compared to activity at injection time. "
            f"The SHAP impact ({shap_val:.3f}) indicates its influence on the model. "
            "Lower C_last suggests poor overall clearance over the entire renogram, whereas higher values reflect effective elimination."
        )

    else:
        return "Feature impact needs further interpretation."


def generate_single_textual_shap_explanation_features(shap_values, predicted_label, confidence, kidney_label):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


    feature_names = [
        "Mean", "Variance", "Skewness", "Kurtosis",
        "C_last", "slope_0_5_min", "slope_15_20_min",
        "Length", "Time to peek", "Peak to 1/2 peak",
        "Diuretic T1/2", "30min/peak", "30min/3min", "Split function"
    ]


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


def generate_single_textual_shap_explanation_datapoints(shap_data, time_bins, classified_label, confidence, kidney_label, diuretic_time):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    shap_values = np.array(shap_data[0])
    feature_values = np.array(shap_data[1])
    bins_for_prompt = [
        {
            "interval": f"{int(start)}–{int(end)} min",
            "shap_contribution": float(s),
            "summed_counts": float(v)
        }
        for (start, end), s, v in zip(time_bins, shap_values, feature_values)
    ]

    prompt = f"""
    You are an AI clinical assistant.  You’ve been given:
    
    - A **binary classification** of upper urinary tract obstruction from diuretic renography.
    - The **full renogram curve** has been **binned into 2-minute intervals** (from time 0 to the end).
    - The **diuretic injection** occurred at **{diuretic_time} minutes**.
    
    **Inputs:**
    - **Predicted class:** {classified_label.upper()}  
    - **Model confidence:** {confidence:.0%}  
    - **Kidney side:** {kidney_label.capitalize()}  
    - **2-Minute bins:** {bins_for_prompt}

    > **Note:** 
    > - A **positive** SHAP for a bin means that that time-window *increased* the probability of the *predicted* class; a **negative** SHAP means it *decreased* it.  
    > - “summed_counts” are just the sum of tracer counts in that bin. Use them to see relative retention vs. wash-out, but don’t over-interpret absolute numbers.

    **Tasks**  
    1. **Overview.**  Start by restating “The model classified this kidney as {classified_label} with {confidence:.0%} confidence.”  
    3. **Physiological interpretation.**  For each key bin, explain *why* that uptake pattern would be interpreted by the model as evidence for or against obstruction.  E.g. “High uptake at 0–2 min suggests retention of tracer—consistent with obstruction—so a +SHAP to ‘obstructed’ makes sense.”  
    5. **Conclusion.**  Summarize in one clear clinical sentence why the combination of these bins led to the model’s decision.

    **Format:**  
    - Use bullet points or short paragraphs.  
    - Always link the *sign* of SHAP to “toward/away from the predicted class.”  
    - Tie each SHAP back to the *uptake value* and known physiology (wash-out vs. retention).  

    Now generate the explanation.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    textual_explanation = response.text.strip() if hasattr(response, "text") else "No explanation generated."

    return textual_explanation


def compute_split_function(left_activities, right_activities, time_vector):

    print("Left activities:", left_activities)
    print("Right activities:", right_activities)

    left_activities = np.asarray(left_activities).ravel()
    right_activities = np.asarray(right_activities).ravel()
    time_vector= np.asarray(time_vector).ravel()

    durations_min = np.diff(time_vector, prepend=0.0)
    durations_s = durations_min * 60.0

    # mask uptake window
    uptake_mask = (time_vector >= 2.0) & (time_vector <= 3.0)

    # uptake areas
    area_left = np.sum(left_activities[uptake_mask] * durations_s[uptake_mask])
    area_right = np.sum(right_activities[uptake_mask] * durations_s[uptake_mask])
    total = area_left + area_right

    if total > 0:
        split_left = area_left / total
        split_right = area_right / total
    else:
        split_left = split_right = 0.5

    return split_left, split_right



def perform_features_analysis(smoothed_activity_array, dt_model, dt_training_data, diuretic_time, time_vector):
    left_activities, right_activities = smoothed_activity_array

    split_left, split_right = compute_split_function(left_activities, right_activities, time_vector)

    extracted_features_left = extract_quantitative_features(left_activities, time_vector, int(diuretic_time))
    extracted_features_right = extract_quantitative_features(right_activities, time_vector, int(diuretic_time))

    extracted_features_left = np.hstack([extracted_features_left, [split_left]])
    extracted_features_right = np.hstack([extracted_features_right, [split_right]])

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
