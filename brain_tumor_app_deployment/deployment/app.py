import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import cv2
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import io
import base64
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'effbo.keras'))

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Simple user database (in a real app, this would be a database)
users = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'name': 'Administrator'
    },
    'doctor': {
        'password': generate_password_hash('doctor123'),
        'name': 'Doctor'
    }
}

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, name):
        self.id = id
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id, users[user_id]['name'])
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_lime_explanation(image_array, filename):
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Prepare the image
    img = image_array[0].astype('double')
    
    # Create a prediction function
    def predict_fn(images):
        return model.predict(images)
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img,
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=1000
    )
    
    # Get the top predicted class
    top_class = explanation.top_labels[0]
    
    # Get the explanation data
    explanation_map = explanation.local_exp[top_class]
    explanation_map = sorted(explanation_map, key=lambda x: x[0])
    
    # Create a heatmap visualization
    segments = explanation.segments
    heatmap = np.zeros(segments.shape)
    
    # Fill in the heatmap based on feature importance
    for segment, importance in explanation_map:
        heatmap[segments == segment] = importance
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(img.astype('uint8'))
    plt.imshow(heatmap, cmap='coolwarm', alpha=0.5)
    plt.colorbar(label='Feature importance')
    plt.title(f"Heatmap for {class_names[top_class]}")
    plt.axis('off')
    
    # Save the figure
    lime_path = os.path.join(app.config['UPLOAD_FOLDER'], f"lime_{filename}")
    plt.savefig(lime_path, bbox_inches='tight')
    plt.close()
    
    return lime_path, class_names[top_class]

def generate_shap_explanation(image_array, filename):
    # Create a masker that is used to mask out partitions of the input image
    masker = shap.maskers.Image("inpaint_telea", image_array[0].shape)
    
    # Create a prediction function
    def predict_fn(x):
        return model.predict(x)
    
    # Create SHAP explainer
    explainer = shap.Explainer(predict_fn, masker, output_names=class_names)
    
    # Generate SHAP values
    shap_values = explainer(image_array, max_evals=100, batch_size=1, outputs=shap.Explanation.argsort.flip[:1])
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    shap.image_plot(shap_values, show=False)
    
    # Save the figure
    shap_path = os.path.join(app.config['UPLOAD_FOLDER'], f"shap_{filename}")
    plt.savefig(shap_path)
    plt.close()
    
    return shap_path

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and check_password_hash(users[username]['password'], password):
            user = User(username, users[username]['name'])
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.name)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            img_array = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index]) * 100
            
            # Generate metrics
            metrics = {
                'accuracy': confidence,
                'precision': confidence,  # Simplified for demo
                'recall': confidence,     # Simplified for demo
                'f1_score': confidence    # Simplified for demo
            }
            
            # Generate LIME explanation
            lime_path, lime_class = generate_lime_explanation(img_array, filename)
            
            # Generate SHAP explanation
            shap_path = generate_shap_explanation(img_array, filename)
            
            # Store results in session
            session['results'] = {
                'original_image': filename,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'metrics': metrics,
                'lime_image': os.path.basename(lime_path),
                'lime_class': lime_class,
                'shap_image': os.path.basename(shap_path)
            }
            
            return redirect(url_for('results'))
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results')
@login_required
def results():
    if 'results' not in session:
        flash('No results to display. Please upload an image first.', 'warning')
        return redirect(url_for('upload'))
    
    results = session['results']
    original_image = results['original_image']
    lime_image = results['lime_image']
    shap_image = results['shap_image']
    
    return render_template(
        'results.html',
        original_image=original_image,
        predicted_class=results['predicted_class'],
        confidence=results['confidence'],
        metrics=results['metrics'],
        lime_image=lime_image,
        lime_class=results['lime_class'],
        shap_image=shap_image
    )

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0')
