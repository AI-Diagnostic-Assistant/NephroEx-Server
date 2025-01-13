import os

import numpy as np
import torch
from flask import request, jsonify, Blueprint
import uuid
from app.logic.logic import create_composite_image, save_image_to_bytes, run_single_classification_cnn, \
    run_single_classification_svm, create_composite_image_test, load_image, predict_kidney_masks, create_renogram, \
    align_masks_over_frames, visualize_masks
from app.client import create_sb_client, authenticate_request

api = Blueprint('api', __name__)

@api.route('/')
@api.route('/index')
def index():
    return "Hello, World!"


@api.route('/users')
def get_users():
    supabase_client = create_sb_client()
    response = supabase_client.table('profiles').select('*').execute()
    return response.data

@api.route('/compositeImages')
def get_composite_images():
    supabase_client = create_sb_client()
    response = supabase_client.storage.from_('composite-images').list()
    return response


@api.route('/process_dicom', methods=['POST'])
def process_dicom():
    supabase_client = create_sb_client()
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    try:
        # Process the DICOM files
        composite_image = create_composite_image(files)

        image_io = save_image_to_bytes(composite_image)

        # Generate a unique filename
        image_filename = f"{uuid.uuid4()}.png"

        # Upload to Supabase Storage
        bucket = supabase_client.storage.from_('composite-images')
        bucket.upload(image_filename, image_io.getvalue(), file_options={'content-type': 'image/png'})
        public_url = bucket.get_public_url(image_filename)

        data = {
            'image_url': public_url,
        }
        supabase_client.table('composite_image').insert(data).execute()

        return jsonify({'image_url': public_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/classify_cnn', methods=['POST'])
@authenticate_request
def classify_cnn(supabase_client, user_info):
    if 'file' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    file = request.files.getlist('file')
    if len(file) == 0:
        return jsonify({'error': 'No files selected'}), 400

    print(file[0])
    print("Type of files: ", type(file[0]))
    predicted, probabilities = run_single_classification_cnn(file[0].stream)

    response = (
        supabase_client.table("analysis")
        .insert({"user_id": user_info.user.id,
                 "ckd_stage_prediction": predicted,
                 "probabilities": probabilities.tolist()}).execute()
    )

    print(response)

    return jsonify({'message': 'Classify endpoint', "id": response.data[0]["id"]}), 200

@api.route('/classify_svm', methods=['POST'])
@authenticate_request
def classify_svm(supabase_client, user_info):
    if 'file' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    dicom_file = request.files.getlist('file')
    if len(dicom_file) == 0:
        return jsonify({'error': 'No files selected'}), 400

    print(dicom_file[0])
    print("Type of files: ", type(dicom_file[0]))

    # Create composite image of the request file
    composite_image = create_composite_image_test(dicom_file[0].stream)

    # Reset the stream pointer
    dicom_file[0].stream.seek(0)

    # Predict kidney masks
    left_mask, right_mask = predict_kidney_masks(composite_image)

    #visualize_masks(composite_image, left_mask, right_mask)

    #align masks over all frames in the original dicom file
    left_mask_alignments, right_mask_alignments = align_masks_over_frames(left_mask, right_mask, dicom_file[0].stream)

    # Create renogram from the predicted masks
    left_activities, right_activities, total_activities = create_renogram(left_mask_alignments, right_mask_alignments)

    roi_activity_array = np.concatenate([left_activities, right_activities, total_activities])

    # Predict CKD stage with SVM model
    predicted, probabilities = run_single_classification_svm(roi_activity_array)

    response = (
        supabase_client.table("analysis")
        .insert({"user_id": user_info.user.id,
                 "ckd_stage_prediction": predicted,
                 "probabilities": probabilities.tolist()}).execute()
    )

    print(response)

    return jsonify({'message': 'SVM classify endpoint', "id": response.data[0]["id"]}), 200
