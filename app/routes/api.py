import io
import pydicom
from flask import request, jsonify, Blueprint
from app.logic.logic import perform_datapoints_analysis, perform_features_analysis, group_2_min_frames_and_save_to_storage, \
    save_total_dicom, create_ROI_contours_png, \
    save_png, \
    fetch_model_from_supabase, load_training_data_supabase, create_renogram_raw, create_renogram_summed, \
    predict_kidney_masks, create_composite_image_rgb
from app.client import authenticate_request, create_service_account_client

api = Blueprint('api', __name__)

supabase_client = create_service_account_client()

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

    # Save original and grouped 2 min frames to storage
    dicom_storage_id = save_total_dicom(dicom_stream, supabase_client)
    grouped_2_min_storage_ids = group_2_min_frames_and_save_to_storage(dicom_read, supabase_client)

    # Create composite image of the request file
    composite_image = create_composite_image_rgb(dicom_read)

    # Predict kidney masks
    left_mask, right_mask = predict_kidney_masks(composite_image, unet_model)

    # create renograms for raw and summed data
    left_activities, right_activities = create_renogram_raw(left_mask, right_mask, dicom_read)
    left_activities_summed, right_activities_summed = create_renogram_summed(left_mask, right_mask, dicom_read)

    roi_activity_array = [left_activities.tolist(), right_activities.tolist()]

    # Datapoint and feature importance uto classifications
    left_uto_classsification_datapoints, right_uto_classification_datapoints, left_uto_confidence_datapoints, right_uto_confidence_datapoints, shap_data_left_uto_classification_datapoints, shap_data_right_uto_classification_datapoints, left_textual_explanation_datapoints, right_textual_explanation_datapoints, classified_left_label_datapoints, classified_right_label_datapoints = perform_datapoints_analysis(svm_model, svm_scaler, svm_training_data, left_activities_summed, right_activities_summed)
    left_uto_classification_features, right_uto_classification_features, left_uto_confidence_features, right_uto_confidence_features, shap_data_left_uto_classification_features, shap_data_right_uto_classification_features, left_textual_explanation_features, right_textual_explanation_features, classified_left_label_features, classified_right_label_features = perform_features_analysis(left_activities, right_activities, dt_model, dt_training_data)

    # Create and upload ROI contours
    transparent_contour_image = create_ROI_contours_png(left_mask, right_mask)
    roi_contour_object_path = save_png(transparent_contour_image, "roi_contours", supabase_client)

    # Insert into supabase database
    try:
        report_response = (
            supabase_client.table("report")
            .insert({
                "patient_id": patient_id,
                "dicom_storage_ids": grouped_2_min_storage_ids,
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
            ])
            .execute()
        )

        if not analysis_response.data:
            return jsonify({'error': 'Failed to insert analyses'}), 500

        renogram_analysis_id = analysis_response.data[0]["id"]

        classification_response = (
            supabase_client.table("classification")
            .insert([
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": classified_left_label_datapoints,
                    "confidence": float(left_uto_confidence_datapoints),
                    "type": "svm",
                    "kidney_label": "left",
                },
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": classified_right_label_datapoints,
                    "confidence": float(right_uto_confidence_datapoints),
                    "type": "svm",
                    "kidney_label": "right",
                },
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": classified_left_label_features,
                    "confidence": float(left_uto_confidence_features),
                    "type": "decision_tree",
                    "kidney_label": "left",
                },
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": classified_right_label_features,
                    "confidence": float(right_uto_confidence_features),
                    "type": "decision_tree",
                    "kidney_label": "right",
                },
            ])
            .execute()
        )

        if not classification_response.data or len(classification_response.data) < 2:
            return jsonify({'error': 'Failed to insert classifications'}), 500

        datapoints_classification_left_id = classification_response.data[0]["id"]
        datapoints_classification_right_id = classification_response.data[1]["id"]
        features_classification_left_id = classification_response.data[2]["id"]
        features_classification_right_id = classification_response.data[3]["id"]

        explanation_response = (
            supabase_client.table("explanation")
            .insert([
                {
                    "classification_id": datapoints_classification_left_id,
                    "technique": "SHAP",
                    "description": left_textual_explanation_datapoints,
                    "shap_values_renogram_summed": shap_data_left_uto_classification_datapoints
                },
                {
                    "classification_id": datapoints_classification_right_id,
                    "technique": "SHAP",
                    "description": right_textual_explanation_datapoints,
                    "shap_values_renogram_summed": shap_data_right_uto_classification_datapoints
                },
                {
                    "classification_id": features_classification_left_id,
                    "technique": "SHAP",
                    "description": left_textual_explanation_features,
                    "shap_values_renogram": shap_data_left_uto_classification_features
                },
                {
                    "classification_id": features_classification_right_id,
                    "technique": "SHAP",
                    "description": right_textual_explanation_features,
                    "shap_values_renogram": shap_data_right_uto_classification_features
                },
            ])
            .execute()
        )

        if not explanation_response.data or len(explanation_response.data) < 2:
            return jsonify({'error': 'Failed to insert explanations'}), 500

        print("Explanation response", explanation_response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'classify endpoint', "id": report_id}), 200
