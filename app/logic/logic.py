import pydicom
import numpy as np
import matplotlib.pyplot as plt
import io
import torch
import os

from app.logic.CNN import Simple3DCNN


def create_composite_image(files):
    pixel_data_list = []

    for file in files:
        # Read the DICOM file from the file storage object
        ds = pydicom.dcmread(file)
        pixel_array = ds.pixel_array.astype(float)
        pixel_data_list.append(pixel_array)

    # Stack the pixel arrays along a new axis (frames)
    pixel_data = np.stack(pixel_data_list, axis=0)

    # Create composite image by summing all frames
    composite_image = np.sum(pixel_data, axis=1)
    composite_image = composite_image.squeeze()    # Remove singleton dimensions
    composite_image_normalized = composite_image / np.max(composite_image)

    return composite_image_normalized


def save_image_to_bytes(image_array):
    buf = io.BytesIO()
    plt.imsave(buf, image_array, cmap='gray', format='png')
    buf.seek(0)
    return buf

def run_single_classification(path: str):
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
