from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import librosa
import numpy as np
from keras.models import load_model
import tempfile
import os
import logging
import jwt

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# JWT Configuration
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a secure key
app.config['JWT_EXPIRATION_DELTA'] = timedelta(days=1)  # Token expiration time

db = SQLAlchemy(app)

# Load your trained model
model_path = 'D:/staffs/Msc/code/App/backend/testing10_model.h5'
model = load_model(model_path)

# Define the emotions mapping
emotion_labels = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    address = db.Column(db.String(255), nullable=True)
    secondary_number = db.Column(db.String(20), nullable=True)
    secondary_email = db.Column(db.String(150), nullable=True)
    school_company = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<User {self.email}>'

def hash_password(password):
    return generate_password_hash(password)

def verify_password(password, password_hash):
    return check_password_hash(password_hash, password)

def generate_token(user_id):
    payload = {
        'sub': user_id,
        'exp': datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA']
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

def decode_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def extract_feature(file_path):
    X, sample_rate = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()

    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')

    if not full_name or not email or not password:
        return jsonify({'error': 'Please provide full_name, email, and password'}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({'error': 'User with this email already exists'}), 409

    try:
        new_user = User(
            full_name=full_name,
            email=email,
            password_hash=hash_password(password)
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({
            'message': 'User registered successfully',
            'user': {
                'id': new_user.id,
                'full_name': new_user.full_name,
                'email': new_user.email,
                'created_at': new_user.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error registering user: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/signin', methods=['POST'])
def signin_user():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Please provide email and password'}), 400

    user = User.query.filter_by(email=email).first()
    if user and verify_password(password, user.password_hash):
        token = generate_token(user.id)
        return jsonify({
            'message': 'Sign in successful',
            'token': token,
            'user': {
                'id': user.id,
                'full_name': user.full_name,
                'email': user.email,
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/upload', methods=['POST'])
def upload_file():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401

    user_id = decode_token(token.split(" ")[1])  # Extract token from "Bearer TOKEN"
    if not user_id:
        return jsonify({'error': 'Invalid or expired token'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Use tempfile to handle the temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        feature = extract_feature(temp_file_path)
        feature = np.expand_dims(feature, axis=0)
        predictions = model.predict(feature)
        predicted_label = np.argmax(predictions)
        predicted_emotion = emotion_labels.get(predicted_label, "Unknown")
        
        # Clean up temporary file
        os.remove(temp_file_path)

        return jsonify({'emotion': predicted_emotion})
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check-auth', methods=['GET'])
def check_auth():
    token = request.headers.get('Authorization')
    if token:
        token = token.split(" ")[1]  # Remove "Bearer " prefix
        user_id = decode_token(token)
        if user_id:
            return jsonify({'isAuthenticated': True}), 200
    return jsonify({'isAuthenticated': False}), 401

@app.route('/user-info', methods=['GET'])
def get_user_info():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401

    user_id = decode_token(token.split(" ")[1])  # Extract token from "Bearer TOKEN"
    if not user_id:
        return jsonify({'error': 'Invalid or expired token'}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'user': {
            'id': user.id,
            'full_name': user.full_name,
            'email': user.email,
            'address': user.address,
            'secondary_number': user.secondary_number,
            'secondary_email': user.secondary_email,
            'school_company': user.school_company,
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
    }), 200

@app.route('/update-user-info', methods=['PUT'])
def update_user_info():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Token is missing'}), 401

    user_id = decode_token(token.split(" ")[1])
    if not user_id:
        return jsonify({'error': 'Invalid or expired token'}), 401

    data = request.get_json()
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Update user information
    user.address = data.get('address', user.address)
    user.secondary_number = data.get('secondary_number', user.secondary_number)
    user.secondary_email = data.get('secondary_email', user.secondary_email)
    user.school_company = data.get('school_company', user.school_company)
    
    try:
        db.session.commit()
        return jsonify({
            'message': 'User information updated successfully',
            'user': {
                'id': user.id,
                'full_name': user.full_name,
                'email': user.email,
                'address': user.address,
                'secondary_number': user.secondary_number,
                'secondary_email': user.secondary_email,
                'school_company': user.school_company,
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating user info: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
   app.run(port=5000, debug=True)
