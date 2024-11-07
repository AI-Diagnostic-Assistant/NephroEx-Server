import pydicom
import numpy as np
import matplotlib.pyplot as plt
import io

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