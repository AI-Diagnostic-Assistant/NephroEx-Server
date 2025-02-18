import io
import pydicom
from flask import request, jsonify, Blueprint
from app.logic.logic import run_single_classification_cnn, \
    perform_svm_analysis, group_2_min_frames, save_summed_frames_to_storage, save_total_dicom, create_ROI_contours_png, \
    save_png, perform_decision_tree_analysis, create_and_overlay_heatmaps, save_composite_heatmaps, \
    fetch_model_from_supabase, load_training_data_supabase
from app.client import authenticate_request, create_service_account_client

api = Blueprint('api', __name__)

supabase_client = create_service_account_client()

cnn_model = fetch_model_from_supabase(supabase_client, "best_3dcnn_model.pth", is_cnn_model=True)
svm_model = fetch_model_from_supabase(supabase_client, "svm_model_summed.joblib")
svm_scaler = fetch_model_from_supabase(supabase_client, "svm_scaler_summed.joblib")
svm_training_data = load_training_data_supabase(supabase_client, "svm_training_data.npy")
dt_model = fetch_model_from_supabase(supabase_client, "random_forest_best_final.joblib")
dt_training_data = load_training_data_supabase(supabase_client, "random_forest_training_data_best_final.npy")
unet_model = fetch_model_from_supabase(supabase_client, "unet_model.pth", is_unet_model=True)


@api.route('/')
@api.route('/index')
def index():
    return "Hello, World!"

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

    # Save stream to memory and read one time
    dicom_stream = dicom_file[0].stream.read()
    dicom_read = pydicom.dcmread(io.BytesIO(dicom_stream))

    dicom_storage_id = save_total_dicom(dicom_stream, supabase_client)

    grouped_frames = group_2_min_frames(dicom_read)

    storage_ids = save_summed_frames_to_storage(grouped_frames, supabase_client)

    # CNN and SVM predictions
    cnn_predicted, cnn_confidence = run_single_classification_cnn(dicom_read, cnn_model)
    svm_predicted, svm_confidence, roi_activity_array, left_mask, right_mask, total_activities, shap_data_svm, svm_textual_explanation = perform_svm_analysis(dicom_read, svm_model, svm_scaler, svm_training_data, unet_model)
    decision_tree_predicted, decision_tree_confidence, shap_data_dt, decision_tree_textual_explanation = perform_decision_tree_analysis(total_activities, dt_model, dt_training_data)

    svm_predicted_label = "healthy" if svm_predicted == 0 else "sick"
    cnn_predicted_label = "healthy" if cnn_predicted == 0 else "sick"
    decision_tree_predicted_label = "healthy" if decision_tree_predicted == 0 else "sick"

    heatmap_overlays = create_and_overlay_heatmaps(dicom_read)

    storage_heatmap_paths = save_composite_heatmaps(heatmap_overlays, supabase_client)

    # Create and upload ROI contours
    transparent_contour_image = create_ROI_contours_png(left_mask, right_mask)
    roi_contour_object_path = save_png(transparent_contour_image, "roi_contours", supabase_client)

    # Insert into supabase database
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
                    "confidence": float(svm_confidence),
                    "type": "svm",
                },
                {
                    "analysis_id": image_analysis_id,
                    "prediction": cnn_predicted_label,
                    "confidence": float(cnn_confidence),
                    "type": "cnn",
                },
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": decision_tree_predicted_label,
                    "confidence": float(decision_tree_confidence),
                    "type": "decision_tree",
                },
            ])
            .execute()
        )

        if not classification_response.data or len(classification_response.data) < 2:
            return jsonify({'error': 'Failed to insert classifications'}), 500

        svm_classification_id = classification_response.data[0]["id"]
        cnn_classification_id = classification_response.data[1]["id"]
        decision_tree_classification_id = classification_response.data[2]["id"]

        explanation_response = (
            supabase_client.table("explanation")
            .insert([
                {
                    "classification_id": cnn_classification_id,
                    "technique": "Grad-CAM",
                    "heatmap_object_paths": storage_heatmap_paths
                },
                {
                    "classification_id": svm_classification_id,
                    "technique": "SHAP",
                    "description": svm_textual_explanation,
                    "shap_values_renogram_summed": shap_data_svm
                },
                {
                    "classification_id": decision_tree_classification_id,
                    "technique": "SHAP",
                    "description": decision_tree_textual_explanation,
                    "shap_values_renogram": shap_data_dt
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
