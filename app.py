"""
Flask Web Uygulaması - Modern Arayüz
Tüm modellerin sonuçlarını gösterir, en iyi modeli vurgular.
Kullanıcılar bir CSV dosyası yükleyerek tüm modellerden tahmin alabilirler.
"""
import os
import json
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from pathlib import Path

from config import UPLOAD_DIR, TRAINED_MODELS_DIR
from predictor import Predictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>🔬 Time Series Stationarity Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #f1f5f9;
            --accent: #10b981;
            --error: #ef4444;
            --warning: #f59e0b;
            --text: #1f2937;
            --text-light: #6b7280;
            --bg: #ffffff;
            --bg-secondary: #f8fafc;
            --border: #e5e7eb;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        [data-theme="dark"] {
            --primary: #818cf8;
            --primary-dark: #6366f1;
            --secondary: #1e293b;
            --accent: #34d399;
            --error: #f87171;
            --warning: #fbbf24;
            --text: #f1f5f9;
            --text-light: #94a3b8;
            --bg: #0f172a;
            --bg-secondary: #1e293b;
            --border: #334155;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, var(--bg-secondary) 100%);
            color: var(--text);
            min-height: 100vh;
            transition: all 0.3s ease;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .header {
            text-align: center;
            margin-bottom: 3rem;
            position: relative;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-light);
            font-size: 1.1rem;
        }

        .theme-toggle {
            position: absolute;
            top: 0;
            right: 0;
            background: var(--secondary);
            border: none;
            padding: 0.5rem;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 40px;
            height: 40px;
        }

        .theme-toggle:hover {
            transform: scale(1.1);
            background: var(--border);
        }

        .upload-section {
            background: var(--bg);
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--border);
            margin-bottom: 2rem;
        }

        .file-upload-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
            margin-bottom: 1.5rem;
        }

        .file-upload {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }

        .file-upload input[type=file] {
            position: absolute;
            left: -9999px;
            opacity: 0;
        }

        .file-upload-label {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            border: 2px dashed var(--border);
            border-radius: 0.75rem;
            background: var(--bg-secondary);
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            flex-direction: column;
            gap: 0.5rem;
        }

        .file-upload-label:hover {
            border-color: var(--primary);
            background: var(--secondary);
        }

        .file-upload-label.dragover {
            border-color: var(--primary);
            background: var(--secondary);
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 2rem;
            color: var(--text-light);
        }

        .submit-btn {
            width: 100%;
            padding: 1rem 2rem;
            background: linear-gradient(45deg, var(--primary), var(--primary-dark));
            color: white;
            border: none;
            border-radius: 0.75rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .loading {
            display: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results {
            margin-top: 2rem;
        }

        .error-box {
            background: linear-gradient(45deg, var(--error), #dc2626);
            color: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
        }

        .consensus-card {
            background: linear-gradient(45deg, var(--accent), #059669);
            color: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: var(--shadow-lg);
        }

        .consensus-card h3 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .consensus-percentage {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }

        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }

        .model-card {
            background: var(--bg);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            transition: all 0.3s ease;
            position: relative;
        }

        .model-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }

        .model-card.best {
            border: 2px solid var(--primary);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        }

        .model-card.best::before {
            content: "📌";
            position: absolute;
            top: -5px;
            right: -5px;
            background: var(--primary);
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
        }

        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .model-name {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text);
        }

        .model-name.best {
            color: var(--primary);
        }

        .prediction-badge {
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .prediction-stationary {
            background: #dcfce7;
            color: #166534;
        }

        .prediction-non-stationary {
            background: #fef3c7;
            color: #92400e;
        }

        .confidence-section {
            margin-top: 1rem;
        }

        .confidence-bar {
            background: var(--secondary);
            border-radius: 0.5rem;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }

        .confidence-fill {
            height: 8px;
            background: linear-gradient(45deg, var(--primary), var(--accent));
            border-radius: 0.5rem;
            transition: width 0.5s ease;
        }

        .confidence-text {
            font-size: 0.9rem;
            color: var(--text-light);
            display: flex;
            justify-content: space-between;
        }

        .expand-toggle {
            background: none;
            border: none;
            color: var(--primary);
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            padding: 0.25rem 0;
        }

        .confidence-details {
            display: none;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }

        .confidence-details.show {
            display: block;
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .models-grid {
                grid-template-columns: 1fr;
            }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <button class="theme-toggle" onclick="toggleTheme()">🌙</button>
            <h1>🔬 Stationarity Predictor</h1>
            <p>Upload a CSV file with a 'data' column to get predictions from all trained models</p>
        </div>

        <div class="upload-section">
            <form method="post" enctype="multipart/form-data" onsubmit="showLoading()">
                <div class="file-upload-wrapper">
                    <div class="file-upload">
                        <input type="file" name="file" id="file" accept=".csv" required onchange="updateFileName(this)">
                        <label for="file" class="file-upload-label" ondrop="dropHandler(event)" ondragover="dragOverHandler(event)" ondragenter="dragEnterHandler(event)" ondragleave="dragLeaveHandler(event)">
                            <div class="upload-icon">📊</div>
                            <div id="file-text">Choose CSV file or drag & drop</div>
                            <small>Only .csv files are accepted</small>
                        </label>
                    </div>
                </div>
                <button type="submit" class="submit-btn" id="submit-btn">
                    <span id="btn-text">🚀 Analyze with All Models</span>
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                    </div>
                </button>
            </form>
        </div>

        <div class="results">
            {% if error %}
            <div class="error-box fade-in">
                <h3>❌ Error</h3>
                <p>{{ error }}</p>
            </div>
            {% endif %}

            {% if result %}
            <div class="fade-in">
                {% set data = result %}
                
                <!-- Consensus Card -->
                <div class="consensus-card">
                    <h3>🎯 Model Consensus</h3>
                    <div class="consensus-percentage">{{ data.summary.consensus_percentage }}%</div>
                    <p><strong>{{ data.summary.consensus.upper() }}</strong></p>
                    <small>{{ data.summary.stationary_votes }} models predict stationary, {{ data.summary.non_stationary_votes }} predict non-stationary</small>
                </div>

                <!-- Models Grid -->
                <div class="models-grid">
                    {% for prediction in data.all_predictions %}
                    <div class="model-card {% if prediction.is_best %}best{% endif %}">
                        <div class="model-header">
                            <div class="model-name {% if prediction.is_best %}best{% endif %}">
                                {{ prediction.model_name }}
                                {% if prediction.is_best %}
                                <small style="font-weight: normal; color: var(--text-light);">(Best Model)</small>
                                {% endif %}
                            </div>
                            <div class="prediction-badge prediction-{{ prediction.prediction.replace('_', '-') }}">
                                {{ prediction.prediction.replace('_', ' ') }}
                            </div>
                        </div>

                        <div class="confidence-section">
                            <div class="confidence-text">
                                <span>Confidence</span>
                                <span><strong>{{ prediction.max_confidence if prediction.max_confidence != 'N/A' else 'N/A' }}</strong></span>
                            </div>
                            {% if prediction.max_confidence != 'N/A' %}
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {{ (prediction.max_confidence * 100) if prediction.max_confidence != 'N/A' else 0 }}%"></div>
                            </div>
                            {% endif %}
                        </div>

                        {% if prediction.confidence_scores %}
                        <button class="expand-toggle" onclick="toggleDetails('{{ loop.index }}')">
                            <span id="toggle-text-{{ loop.index }}">Show Details ▼</span>
                        </button>
                        <div class="confidence-details" id="details-{{ loop.index }}">
                            {% for label, score in prediction.confidence_scores.items() %}
                            <div class="confidence-text" style="margin-bottom: 0.3rem;">
                                <span>{{ label.replace('_', ' ').title() }}</span>
                                <span>{{ score }}</span>
                            </div>
                            {% endfor %}
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        function toggleTheme() {
            const body = document.body;
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            body.setAttribute('data-theme', newTheme);
            
            const toggle = document.querySelector('.theme-toggle');
            toggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';
            
            localStorage.setItem('theme', newTheme);
        }

        // Load saved theme
        document.addEventListener('DOMContentLoaded', function() {
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.body.setAttribute('data-theme', savedTheme);
            const toggle = document.querySelector('.theme-toggle');
            toggle.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
        });

        function updateFileName(input) {
            const fileName = input.files[0] ? input.files[0].name : 'Choose CSV file or drag & drop';
            document.getElementById('file-text').textContent = fileName;
        }

        function showLoading() {
            const btn = document.getElementById('submit-btn');
            const btnText = document.getElementById('btn-text');
            const loading = document.getElementById('loading');
            
            btn.disabled = true;
            btnText.style.opacity = '0';
            loading.style.display = 'block';
        }

        function toggleDetails(index) {
            const details = document.getElementById('details-' + index);
            const toggleText = document.getElementById('toggle-text-' + index);
            
            if (details.classList.contains('show')) {
                details.classList.remove('show');
                toggleText.textContent = 'Show Details ▼';
            } else {
                details.classList.add('show');
                toggleText.textContent = 'Hide Details ▲';
            }
        }

        // Drag and drop functionality
        function dragOverHandler(ev) {
            ev.preventDefault();
            ev.currentTarget.classList.add('dragover');
        }

        function dragEnterHandler(ev) {
            ev.preventDefault();
            ev.currentTarget.classList.add('dragover');
        }

        function dragLeaveHandler(ev) {
            ev.currentTarget.classList.remove('dragover');
        }

        function dropHandler(ev) {
            ev.preventDefault();
            ev.currentTarget.classList.remove('dragover');
            
            const files = ev.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.csv')) {
                document.getElementById('file').files = files;
                updateFileName(document.getElementById('file'));
            }
        }
    </script>
</body>
</html>
"""

try:
    predictor = Predictor()
    print("All models loaded successfully.")
    print(f"Available models: {list(predictor.models.keys())}")
    print(f"Best model: {predictor.best_model_name}")
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    predictor = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error, result = None, None
    if predictor is None:
        error = "Models not trained or loaded. Please run 'python main.py' from your terminal first."
        return render_template_string(HTML_TEMPLATE, error=error), 503

    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file'].filename:
            error = "No file selected. Please choose a file to upload."
        else:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                prediction_result = predictor.predict(filepath)
                if "error" not in prediction_result:
                    result = prediction_result
                else:
                    error = prediction_result["error"]
                    
                os.remove(filepath)
            else:
                error = "Invalid file type. Please upload a .csv file."
            
    return render_template_string(HTML_TEMPLATE, result=result, error=error)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    if predictor is None:
        return jsonify({"error": "Models not trained or loaded."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if not file or not file.filename.endswith('.csv'):
        return jsonify({"error": "No selected file or invalid file type (must be .csv)."}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    prediction = predictor.predict(filepath)
    os.remove(filepath)

    status_code = 500 if "error" in prediction else 200
    return jsonify(prediction), status_code

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        "status": "healthy" if predictor is not None else "unhealthy",
        "models_loaded": len(predictor.models) if predictor else 0,
        "best_model": predictor.best_model_name if predictor else None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔬 Time Series Stationarity Predictor")
    print("="*60)
    
    if predictor is None:
        print("❌ Web server cannot start due to model loading error.")
        print("💡 Please run 'python main.py' to train models first.")
    else:
        print(f"✅ {len(predictor.models)} models loaded successfully")
        print(f"🏆 Best model: {predictor.best_model_name}")
        print("\n🚀 Starting web server...")
        print("📝 Access the application at: http://localhost:5000")
        print("🔗 For ngrok deployment, run: ngrok http 5000")
        print("="*60)
        app.run(debug=True, host='0.0.0.0', port=5000)