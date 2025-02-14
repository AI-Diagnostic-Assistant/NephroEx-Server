from supabase import create_client
from flask import request, jsonify
from functools import wraps
from dotenv import load_dotenv
import os

load_dotenv()

def create_service_account_client():
    url: str = os.getenv('SUPABASE_PROJECT_URL')
    key: str = os.getenv('SUPABASE_SERVICE_ACCOUNT_KEY')
    print("URL: ", url)
    return create_client(url, key)


def authenticate_user(supabase_client):
    """Helper function to authenticate a user using the Authorization header."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, jsonify({'error': 'Missing or invalid Authorization header'}), 401

    access_token = auth_header.split("Bearer ")[1].strip()
    user_info = supabase_client.auth.get_user(access_token)

    print("User info: ", user_info)

    if not user_info or not user_info.user:
        return None, jsonify({'error': 'Invalid access token or no session'}), 401

    if user_info.user.role != 'authenticated':
        return None, jsonify({'error': 'Unauthorized'}), 403

    supabase_client.postgrest.auth(access_token)
    return None, None


def authenticate_request(func):
    """Decorator to handle authentication and inject user_info into the route."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        #Have to change this perhaps
        #supabase_client = create_sb_client()
        supabase_client = create_service_account_client()

        error_response, status_code = authenticate_user(supabase_client)
        if error_response:
            return error_response, status_code
        return func(supabase_client, *args, **kwargs)

    return wrapper
