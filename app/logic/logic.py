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
from app.logic.CNN import Simple3DCNN


def save_image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def run_single_classification_cnn(path: str):
    model = Simple3DCNN()
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    model_path = os.path.join(script_dir, "../../models/best_3dcnn_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # path = os.path.join(script_dir, '../../test/drsprg_001_POST.dcm')

    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    img /= np.max(img)  # Normalize
    volume_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(volume_tensor)
        predicted = torch.argmax(output, dim=1)

    predicted_class = predicted.item()
    print(f"Predicted class: {predicted_class}")
    print("Output probabilities:", output.flatten())

    return predicted_class, output.flatten()


def run_single_classification_svm(roi_activity_array):
    script_dir = os.path.dirname(__file__)
    svm_model_path = os.path.join(script_dir, "../../models/svm/svm_model_1.joblib")
    svm_model = joblib.load(svm_model_path)  # Load the entire model

    roi_activity_array = np.array(roi_activity_array).reshape(1, -1)

    probabilities = svm_model.predict_proba(roi_activity_array)[0]  # [0] to extract single sample probabilities
    predicted_class = np.argmax(probabilities)

    predicted_class = int(predicted_class)

    return predicted_class, probabilities


def create_composite_image(path: str):
    pixel_data_list = []

    #img = load_image(path)

    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(float)
    pixel_data_list.append(pixel_array)

    pixel_data = np.stack(pixel_data_list, axis=0)

    # Create composite image by summing all frames
    composite_image = np.sum(pixel_data, axis=1)
    composite_image = composite_image.squeeze()  # Remove singleton dimensions
    composite_image_normalized = composite_image / np.max(composite_image)

    print(f"Composite Image Shape: {composite_image_normalized.shape}")
    print(f"Composite Image Dtype: {composite_image_normalized.dtype}")

    # Convert to 3-channel RGB by repeating the grayscale channel
    composite_image_normalized = np.expand_dims(composite_image_normalized, axis=-1)
    composite_image_normalized = np.repeat(composite_image_normalized, 3, axis=-1)

    print(f"Composite Image Shape: {composite_image_normalized.shape}")
    print(f"Composite Image Dtype: {composite_image_normalized.dtype}")

    return composite_image_normalized


def load_image(path: str):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    img /= np.max(img)

    return img


def predict_kidney_masks(composite_image):
    script_dir = os.path.dirname(__file__)
    unet_model_path = os.path.join(script_dir, "../../models/segmentation/fold_1_pretrained_unet_model.pth")
    unet_model = torch.load(unet_model_path, map_location=torch.device('cpu'))  # Load the entire model
    unet_model.eval()

    image_tensor = torch.tensor(composite_image).permute(2, 0, 1).unsqueeze(0).float()

    print("Image Tensor:", image_tensor.shape, image_tensor.min().item(), image_tensor.max().item())  # Debug

    with torch.no_grad():
        pred_masks = unet_model(image_tensor)

        print("Raw Predictions (Min, Max):", pred_masks.min().item(), pred_masks.max().item())  # Debug

        pred_left_mask = (pred_masks[:, 0, :, :] > 0.5).cpu().numpy().squeeze()
        pred_right_mask = (pred_masks[:, 1, :, :] > 0.5).cpu().numpy().squeeze()

    return pred_left_mask, pred_right_mask


def align_masks_over_frames(left_kidney_mask, right_kidney_mask, dcm_file_path):
    left_mask_alignments = []
    right_mask_alignments = []

    ds = pydicom.dcmread(dcm_file_path)

    global_max = max(np.max(ds.pixel_array[frame_idx]) for frame_idx in range(ds.NumberOfFrames))

    #left_kidney_mask = (left_kidney_mask * 255).astype(np.uint8)
    #right_kidney_mask = (right_kidney_mask * 255).astype(np.uint8)

    for frame_idx in range(ds.NumberOfFrames):
        pixel_array = ds.pixel_array[frame_idx]
        #normalized_pixel_array = (pixel_array / np.max(pixel_array) * 255).astype(np.uint8)
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


def perform_svm_analysis(dicom_file, supabase_client):
    # Create composite image of the request file
    composite_image = create_composite_image(dicom_file[0].stream)

    # Reset the stream pointer
    dicom_file[0].stream.seek(0)

    # Predict kidney masks
    left_mask, right_mask = predict_kidney_masks(composite_image)

    transparent_contour_image = create_ROI_contours_png(left_mask, right_mask)

    save_png(transparent_contour_image, "roi_contours", supabase_client)

    visualize_masks(composite_image, left_mask, right_mask)

    # align masks over all frames in the original dicom file
    left_mask_alignments, right_mask_alignments = align_masks_over_frames(left_mask, right_mask, dicom_file[0].stream)

    # Create renogram from the predicted masks
    left_activities, right_activities, total_activities = create_renogram(left_mask_alignments, right_mask_alignments)
    roi_activity_array = [left_activities.tolist(), right_activities.tolist(), total_activities.tolist()]

    # print("roi_activity_array: ", roi_activity_array)

    # Predict CKD stage with SVM model
    svm_predicted, svm_probabilities = run_single_classification_svm(roi_activity_array)

    return svm_predicted, svm_probabilities, roi_activity_array


def group_2_min_frames(dicom_file_stream):
    dicom_data = pydicom.dcmread(dicom_file_stream)
    frames = dicom_data.pixel_array

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