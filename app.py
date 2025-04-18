from flask import Flask, render_template, request, redirect, url_for
from flask import Flask, jsonify, request, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
import shutil
import numpy as np
from sklearn.neighbors import NearestNeighbors
from colorthief import ColorThief # type: ignore

import random

app = Flask(__name__)

# ======================
# LOCAL CONFIGURATION
# ======================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Use current directory
APP_ROOT = BASE_DIR
dataset_path = 'NEW_COLORS'

# Ensure the dataset path exists
if not os.path.exists(dataset_path):
    raise Exception(f"The dataset folder {dataset_path} does not exist")


# Configure folders
app.config.update(
    UPLOAD_FOLDER=os.path.join(APP_ROOT, 'static', 'uploads'),
    RESULTS_FOLDER=os.path.join(APP_ROOT, 'static', 'results'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB file limit
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'}
)

# Create folders if missing
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# ======================
# IMAGE PROCESSING HELPERS
# ======================
def preprocess_image(image_path):
    """Preprocess image to normalize lighting and enhance color consistency."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB').resize((50, 50))
            
            # Auto-enhance contrast and brightness
            img = ImageEnhance.Contrast(img).enhance(1.2)
            img = ImageEnhance.Brightness(img).enhance(1.1)
            
            # Convert to LAB color space for better color matching
            lab = np.array(img) / 255.0  # Normalize
            return np.mean(lab, axis=(0, 1))
    except Exception as e:
        print(f"⚠️ Preprocessing failed: {str(e)}")
        return None

# Utility function to calculate the "average" color of an image
def calculate_average_color(image_path):
    img = Image.open(image_path)
    img = img.resize((50, 50))  # Resize to a smaller size for faster processing
    img_data = np.array(img)
    avg_color = np.mean(img_data, axis=(0, 1))  # Calculate the average color (RGB)
    return tuple(avg_color.astype(int))


# ======================
# COLOR MATCHING ENGINE
# ======================
class ColorMatcher:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.ref_features, self.labels = self._process_reference_images()
        self.nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        self.nn.fit(self.ref_features)

    def _process_reference_images(self):
        features, labels = [], []
        print(f"🔍 Loading color dataset from: {self.dataset_root}")
        
        for shade in os.listdir(self.dataset_root):
            shade_dir = os.path.join(self.dataset_root, shade)
            if os.path.isdir(shade_dir):
                for file in os.listdir(shade_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                        img_path = os.path.join(shade_dir, file)
                        feature = preprocess_image(img_path)
                        if feature is not None:
                            features.append(feature)
                            labels.append(shade)
                        break  # Use first image per shade
        
        print(f"✅ Loaded {len(labels)} reference colors")
        return np.array(features), np.array(labels)

    def find_matches(self, image_path):
        feature = preprocess_image(image_path)
        if feature is None:
            return []  

        distances, indices = self.nn.kneighbors([feature])
        
        return [{
            'shade': self.labels[i],
            'similarity': max(0, 100 - d * 10),  # Adjust similarity scaling
            'image': self._get_shade_image(self.labels[i])
        } for d, i in zip(distances[0], indices[0])][:3]

    def _get_shade_image(self, shade_name):
        shade_dir = os.path.join(self.dataset_root, shade_name)
        for file in os.listdir(shade_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                return os.path.join(shade_dir, file)
        return None

# ======================
# INITIALIZE MATCHER
# ======================
DATASET_PATH = os.path.join(APP_ROOT, 'NEW_COLORS')
matcher = ColorMatcher(DATASET_PATH)

# ======================
# INITIALIZE MATCHER
# ======================
DATASET_PATH = os.path.join(APP_ROOT, 'NEW_COLORS')
matcher = ColorMatcher(DATASET_PATH)

# ======================
# FLASK ROUTES
# ======================
@app.route('/')
def redirect_to_intro():
    return redirect(url_for('intro'))  # Redirect root URL to intro page

@app.route('/intro')
def intro():
    return render_template('intro.html')  # Show intro.html page

@app.route('/index')
def home():
    return render_template('index.html')  # Show index.html page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/pay')
def pay():
    return render_template('pay.html')

@app.route('/upload_click')
def upload_form():
    return render_template('upload_click.html')

@app.route('/color_wall')
def color_wall():
    return render_template('color_wall.html')

@app.route('/contact', methods=['POST'])
def handle_contact():
    # Process form submission here
    name = request.form.get('name')
    email = request.form.get('email')
    # ... etc
    return redirect(url_for('contact'))


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)
        
        if file and '.' in file.filename and \
           file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Get the matches for the uploaded image
            matches = matcher.find_matches(upload_path)
            if not matches:  # If no matches found
                return render_template('no_matches.html')  # Show no matches page
            
            results = []
            for match in matches:
                if match['image']:
                    match_file = f"match_{match['shade']}_{os.path.basename(match['image'])}"
                    dest_path = os.path.join(app.config['RESULTS_FOLDER'], match_file)
                    shutil.copy2(match['image'], dest_path)
                    results.append({
                        'shade': match['shade'],
                        'similarity': round(match['similarity'], 1),
                        'image': match_file
                    })
            
            return render_template('results.html', 
                                original=filename,
                                matches=results)
    
    return render_template('about.html')

# Route to handle image upload
@app.route('/upload_click', methods=['POST'])
def upload_click():
    # Get the uploaded image file
    file = request.files['file']
    if file:
        # Save the uploaded image temporarily
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Return the path of the uploaded image for preview
        return jsonify({'success': True, 'image_url': f'/uploads/{file.filename}'})
    else:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400

# Route to handle color matching on image click
@app.route('/match_click', methods=['POST'])
def match_click():
    data = request.json
    image_url = data['image_url']
    x, y = data['x'], data['y']

    # Load the uploaded image to get the clicked color (optional for your use case)
    # For now, we will simulate the matching process with random sample matches

    # Sample a few shades from the dataset to simulate matching
    shades = os.listdir(dataset_path)  # List all the subfolders (shades)
    matches = []

    for shade in random.sample(shades, 5):  # Simulate matching with 5 random shades
        shade_folder = os.path.join(dataset_path, shade)
        if os.path.isdir(shade_folder):
            # Select a random image from the shade folder
            images = os.listdir(shade_folder)
            image_file = random.choice(images)

            # Get the image path and calculate its average color
            image_path = os.path.join(shade_folder, image_file)
            avg_color = calculate_average_color(image_path)

            # Generate a dummy similarity score for demo purposes
            similarity = random.randint(70, 100)

            match = {
                'shade': shade,
                'image': image_file,
                'similarity': similarity,
                'color_code': '#'+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])  # Dummy color code
            }
            matches.append(match)

    # Return the matches to the frontend
    return jsonify({'matches': matches})

# Route to serve images from the dataset folder (non-static)
@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory('uploads', filename)

@app.route('/static/NEW_COLORS/<shade>/<filename>')
def serve_image(shade, filename):
    # Serve images dynamically from the NEW_COLORS folder
    shade_folder = os.path.join(dataset_path, shade)
    return send_from_directory(shade_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
