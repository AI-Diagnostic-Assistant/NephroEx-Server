import io

import pydicom
from flask import request, jsonify, Blueprint
import uuid
from app.logic.logic import create_composite_image_rgb, save_image_to_bytes, run_single_classification_cnn, \
    perform_svm_analysis, group_2_min_frames, save_summed_frames_to_storage, save_total_dicom, create_ROI_contours_png, \
    save_png, perform_decision_tree_analysis, generate_gradcam, aggregate_heatmap, create_composite_image, \
    overlay_heatmap
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
        composite_image = create_composite_image_rgb(files)

        image_io = save_image_to_bytes(composite_image)

        # Generate a unique filename
        image_filename = f"{uuid.uuid4()}.png"

        # Upload to Supabase Storage
        bucket = supabase_client.storage.from_('composite-images')
        bucket.upload(image_filename, image_io, file_options={'content-type': 'image/png'})
        public_url = bucket.get_public_url(image_filename)

        data = {
            'image_url': public_url,
        }
        supabase_client.table('composite_image').insert(data).execute()

        return jsonify({'image_url': public_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/classify', methods=['POST'])
@authenticate_request
def classify(supabase_client):
    if 'file' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    if "patientId" not in request.form:
        return jsonify({'error': 'No patientId in the request'}), 400

    dicom_file = request.files.getlist('file')
    patient_id = request.form.get('patientId')

    if len(dicom_file) == 0:
        return jsonify({'error': 'No files selected'}), 400

    #Save stream to memory and read one time
    dicom_stream = dicom_file[0].stream.read()
    dicom_read = pydicom.dcmread(io.BytesIO(dicom_stream))

    dicom_storage_id = save_total_dicom(dicom_stream, supabase_client)

    grouped_frames = group_2_min_frames(dicom_read)

    storage_ids = save_summed_frames_to_storage(grouped_frames, supabase_client)

    #CNN and SVM predictions
    cnn_predicted, cnn_probabilities = run_single_classification_cnn(dicom_read)
    svm_predicted, svm_probabilities, roi_activity_array, left_mask, right_mask, total_activities = perform_svm_analysis(dicom_read, supabase_client)
    decision_tree_predicted, decision_tree_probabilities, decision_tree_explanation, decision_tree_textual_explanation = perform_decision_tree_analysis(total_activities)

    svm_predicted_label = "healthy" if svm_predicted == 0 else "sick"
    cnn_predicted_label = "healthy" if cnn_predicted == 0 else "sick"
    decision_tree_predicted_label = "healthy" if decision_tree_predicted == 0 else "sick"

    #Create and upload Grad-CAM
    gradcam = generate_gradcam()

    # Aggregate heatmap across depth
    aggregated_heatmap = aggregate_heatmap(gradcam, method="max")

    # Create composite image
    composite_image = create_composite_image(dicom_read)

    # Overlay heatmap on the composite image
    overlay_image = overlay_heatmap(composite_image, aggregated_heatmap)
    grad_cam_object_path = save_png(overlay_image, "heatmaps", supabase_client)

    #Create and upload ROI contours
    transparent_contour_image = create_ROI_contours_png(left_mask, right_mask)
    roi_contour_object_path = save_png(transparent_contour_image, "roi_contours", supabase_client)

    #Insert into supabase database
    try:
        report_response = (
            supabase_client.table("report")
            .insert({
                "patient_id": patient_id,
                "dicom_storage_ids": storage_ids,
                "patient_dicom_storage_id": dicom_storage_id,
                "roi_contour_object_path": roi_contour_object_path,
            })
            .execute()
        )

        if not report_response.data or len(report_response.data) == 0:
            return jsonify({'error': 'Failed to insert into report table'}), 500

        report_id = report_response.data[0]["id"]

        analysis_response = (
            supabase_client.table("analysis")
            .insert([
                {
                    "report_id": report_id,
                    "category": "renogram",
                    "roi_activity": roi_activity_array,
                },
                {
                    "report_id": report_id,
                    "category": "image",
                },

            ])
            .execute()
        )

        if not analysis_response.data:
            return jsonify({'error': 'Failed to insert analyses'}), 500

        renogram_analysis_id = analysis_response.data[0]["id"]
        image_analysis_id = analysis_response.data[1]["id"]

        classification_response = (
            supabase_client.table("classification")
            .insert([
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": svm_predicted_label,
                    "confidence": svm_probabilities.tolist(),
                    "type": "svm",
                },
                {
                    "analysis_id": image_analysis_id,
                    "prediction": cnn_predicted_label,
                    "confidence": cnn_probabilities.tolist(),
                    "type": "cnn",
                },
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": decision_tree_predicted_label,
                    "confidence": decision_tree_probabilities.tolist(),
                    "type": "decision_tree",
                },
            ])
            .execute()
        )

        if not classification_response.data or len(classification_response.data) < 2:
            return jsonify({'error': 'Failed to insert classifications'}), 500


        print("Classification response", classification_response)

        svm_classification_id = classification_response.data[0]["id"]
        cnn_classification_id = classification_response.data[1]["id"]
        decision_tree_classification_id = classification_response.data[2]["id"]

        explanation_response = (
            supabase_client.table("explanation")
            .insert([
                {
                    "classification_id": cnn_classification_id,
                    "technique": "Grad-CAM",
                    "description": "This is a description of the GradCAM technique",
                    "heatmap_object_path": grad_cam_object_path
                },
                {
                    "classification_id": decision_tree_classification_id,
                    "technique": "SHAP",
                    "description": decision_tree_textual_explanation,
                    "shap_values_curve": decision_tree_explanation
                }
            ])
            .execute()
        )

        if not explanation_response.data or len(explanation_response.data) < 2:
            return jsonify({'error': 'Failed to insert explanations'}), 500

        print("Explanation response", explanation_response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'classify endpoint', "id": report_id}), 200
