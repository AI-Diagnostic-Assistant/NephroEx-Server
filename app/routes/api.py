from flask import request, jsonify, Blueprint
import uuid
from app.logic.logic import create_composite_image, save_image_to_bytes, run_single_classification
from app.client import supabase_client

api = Blueprint('api', __name__)

@api.route('/')
@api.route('/index')
def index():
    return "Hello, World!"


@api.route('/users')
def get_users():
    response = supabase_client.table('profiles').select('*').execute()
    return response.data

@api.route('/compositeImages')
def get_composite_images():
    response = supabase_extension.client.storage.from_('composite-images').list()
    return response


@api.route('/process_dicom', methods=['POST'])
def process_dicom():
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
        bucket = supabase_extension.client.storage.from_('composite-images')
        bucket.upload(image_filename, image_io.getvalue(), file_options={'content-type': 'image/png'})
        public_url = bucket.get_public_url(image_filename)

        data = {
            'image_url': public_url,
        }
        supabase_extension.client.from_('composite_image').insert(data).execute()

        return jsonify({'image_url': public_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    file = request.files.getlist('file')
    if len(file) == 0:
        return jsonify({'error': 'No files selected'}), 400

    print(file[0])
    print("Type of files: ", type(file[0]))
    run_single_classification(file[0].stream)

    return jsonify({'message': 'Classify endpoint'}), 200