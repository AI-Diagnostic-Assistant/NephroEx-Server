import io
import pydicom
from flask import request, jsonify, Blueprint

from app.logic.logic import perform_datapoints_analysis, perform_features_analysis, \
    group_2_min_frames_and_save_to_storage, \
    save_total_dicom, create_ROI_contours_png, \
    save_png, load_training_data_supabase, create_renogram_raw, \
    predict_kidney_masks, fetch_model_from_supabase, create_uptake_composite_image, cubic_smooth_renograms, interpolate_renograms
from app.client import authenticate_request, create_service_account_client

api = Blueprint('api', __name__)

supabase_client = create_service_account_client()

knn_model = fetch_model_from_supabase(supabase_client, "knn_best.joblib")
knn_training_data = load_training_data_supabase(supabase_client, "knn_training_data_best.npy")
rf_model = fetch_model_from_supabase(supabase_client, "rf_best.joblib")
rf_training_data = load_training_data_supabase(supabase_client, "rf_training_data_best.npy")
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

    if "diuretic" not in request.form:
        return jsonify({'error': 'No diuretic in the request'}), 400

    diuretic_time = request.form.get('diuretic')
    try:
        diuretic = int(diuretic_time)
    except ValueError:
        return jsonify({'error': 'Diuretic value must be a number'}), 400

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

    # Create uptake composite image of the request file
    composite_image = create_uptake_composite_image(dicom_read)

    # Predict kidney masks
    left_mask, right_mask = predict_kidney_masks(composite_image, unet_model)

    # create renograms
    left_activities, right_activities, time_vector = create_renogram_raw(left_mask, right_mask, dicom_read)

    roi_activity_array = [left_activities.tolist(), right_activities.tolist()]

    interpolated_tv, interpolated_seqs = interpolate_renograms(roi_activity_array, time_vector, target_len=220)

    interpolated_smoothed_activity_array = cubic_smooth_renograms(interpolated_seqs, interpolated_tv)
    original_smoothed_activity_array = cubic_smooth_renograms(roi_activity_array, time_vector)

    left_uto_classsification_datapoints, right_uto_classification_datapoints, left_uto_confidence_datapoints, right_uto_confidence_datapoints, shap_data_left_uto_classification_datapoints, shap_data_right_uto_classification_datapoints, left_textual_explanation_datapoints, right_textual_explanation_datapoints, classified_left_label_datapoints, classified_right_label_datapoints, time_bins_left, time_bins_right = perform_datapoints_analysis(knn_model, knn_training_data, interpolated_smoothed_activity_array, interpolated_tv, diuretic)
    left_uto_classification_features, right_uto_classification_features, left_uto_confidence_features, right_uto_confidence_features, shap_data_left_uto_classification_features, shap_data_right_uto_classification_features, left_textual_explanation_features, right_textual_explanation_features, classified_left_label_features, classified_right_label_features = perform_features_analysis(original_smoothed_activity_array, rf_model, rf_training_data, diuretic_time, time_vector)

    # Create and upload ROI contours
    transparent_contour_image = create_ROI_contours_png(left_mask, right_mask)
    roi_contour_object_path = save_png(transparent_contour_image, "roi_contours", supabase_client)


    print("interpolated_renograms", interpolated_seqs)
    print("interpolated_smoothed_renograms", interpolated_smoothed_activity_array)
    print("original_tv", time_vector)
    print("interpolated_tv", interpolated_tv)

    # Insert into supabase database
    try:
        report_response = (
            supabase_client.table("report")
            .insert({
                "patient_id": patient_id,
                "dicom_storage_ids": grouped_2_min_storage_ids,
                "patient_dicom_storage_id": dicom_storage_id,
                "roi_contour_object_path": roi_contour_object_path,
                "diuretic_timing": diuretic,
                "interpolated_smoothed_renograms": interpolated_smoothed_activity_array,
                "interpolated_renograms": interpolated_seqs,
                "original_tv": time_vector.tolist(),
                "interpolated_tv": interpolated_tv.tolist(),

            })
            .execute()
        )

        if not report_response.data or len(report_response.data) == 0:
            return jsonify({'error': 'Failed to insert into report table'}), 500

        report_id = report_response.data[0]["id"]

        print("report_id", report_id)
        print("report_response", report_response)


        analysis_response = (
            supabase_client.table("analysis")
            .insert([
                {
                    "report_id": report_id,
                    "category": "renogram",
                },
                {
                    "report_id": report_id,
                    "category": "feature",
                },
            ])
            .execute()
        )

        if not analysis_response.data:
            return jsonify({'error': 'Failed to insert analyses'}), 500

        renogram_analysis_id = analysis_response.data[0]["id"]
        feature_analysis_id = analysis_response.data[1]["id"]

        classification_response = (
            supabase_client.table("classification")
            .insert([
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": classified_left_label_datapoints,
                    "confidence": float(left_uto_confidence_datapoints),
                    "type": "svm",
                    "kidney_label": "left",
                    "time_bins": time_bins_left
                },
                {
                    "analysis_id": renogram_analysis_id,
                    "prediction": classified_right_label_datapoints,
                    "confidence": float(right_uto_confidence_datapoints),
                    "type": "svm",
                    "kidney_label": "right",
                    "time_bins": time_bins_right
                },
                {
                    "analysis_id": feature_analysis_id,
                    "prediction": classified_left_label_features,
                    "confidence": float(left_uto_confidence_features),
                    "type": "decision_tree",
                    "kidney_label": "left",
                },
                {
                    "analysis_id": feature_analysis_id,
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
                    "shap_values_renogram_summed": shap_data_left_uto_classification_datapoints,
                },
                {
                    "classification_id": datapoints_classification_right_id,
                    "technique": "SHAP",
                    "description": right_textual_explanation_datapoints,
                    "shap_values_renogram_summed": shap_data_right_uto_classification_datapoints,
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
