import os

import numpy as np
from flask import request, jsonify, Blueprint
import uuid
from app.logic.logic import create_composite_image, save_image_to_bytes, run_single_classification_cnn, \
    run_single_classification_svm, predict_kidney_masks, create_renogram, \
    align_masks_over_frames, perform_svm_analysis
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

    if "patientId" not in request.form:
        return jsonify({'error': 'No patientId in the request'}), 400

    patientId = request.form.get('patientId')
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
                 "probabilities": probabilities.tolist(),
                 "patient_id": patientId}).execute()
    )

    print(response)

    return jsonify({'message': 'Classify endpoint', "id": response.data[0]["id"]}), 200


@api.route('/classify', methods=['POST'])
@authenticate_request
def classify(supabase_client, user_info):
    if 'file' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    dicom_file = request.files.getlist('file')
    if len(dicom_file) == 0:
        return jsonify({'error': 'No files selected'}), 400

    print(dicom_file[0])
    print("Type of files: ", type(dicom_file[0]))

    cnn_predicted, cnn_probabilities = run_single_classification_cnn(dicom_file[0].stream)

    dicom_file[0].stream.seek(0)

    svm_predicted, svm_probabilities, roi_activity_array = perform_svm_analysis(dicom_file)

    svm_predicted_label = "healthy" if svm_predicted == 0 else "sick"
    cnn_predicted_label = "healthy" if cnn_predicted == 0 else "sick"

    try:
        analysis_response = (
            supabase_client.table("analysis")
            .insert({
                "user_id": user_info.user.id,
                "ckd_stage_prediction": cnn_predicted,
                "probabilities": cnn_probabilities.tolist(),
            })
            .execute()
        )

        if not analysis_response.data or len(analysis_response.data) == 0:
            return jsonify({'error': 'Failed to insert into analysis table'}), 500

        analysis_id = analysis_response.data[0]["id"]

        classification_response = (
            supabase_client.table("classification")
            .insert([
                {
                    "analysis_id": analysis_id,
                    "prediction": svm_predicted_label,
                    "confidence": svm_probabilities.tolist(),
                    "type": "svm",
                },
                {
                    "analysis_id": analysis_id,
                    "prediction": cnn_predicted_label,
                    "confidence": cnn_probabilities.tolist(),
                    "type": "cnn",
                },
            ])
            .execute()
        )

        if not classification_response.data or len(classification_response.data) < 2:
            return jsonify({'error': 'Failed to insert classifications'}), 500


        svm_classification_id = classification_response.data[0]["id"]
        cnn_classification_id = classification_response.data[1]["id"]

        explanation_response = (
            supabase_client.table("explanation")
            .insert([
                {
                    "classification_id": svm_classification_id,
                    "description": "This is a description of the Renogram technique",
                    "roi_activity": roi_activity_array,
                },
                {
                    "classification_id": cnn_classification_id,
                    "technique": "Grad-CAM",
                    "description": "This is a description of the GradCAM technique",
                }
            ])
            .execute()
        )

        if not explanation_response.data or len(explanation_response.data) < 2:
            return jsonify({'error': 'Failed to insert explanations'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'classify endpoint', "id": analysis_id}), 200
