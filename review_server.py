import datetime
from flask import Flask, render_template, jsonify, send_from_directory, request
import json
import csv
import os
import pandas as pd
from pathlib import Path

app = Flask(__name__, template_folder='templates', static_folder='static')

# Define paths to your data files
OUTPUT_DIR = './output'
RESULTS_PATH = os.path.join(
    OUTPUT_DIR, 'vlm_experiment', 'context_reaction_results_20250327_105647.json')
CSV_PATH = os.path.join(
    OUTPUT_DIR, 'context_reaction_pairs_20250327_105647.csv')
FRAMES_DIR = os.path.join(OUTPUT_DIR, 'frames')
HUMAN_ANNOTATIONS_PATH = os.path.join(OUTPUT_DIR, 'human_annotations.json')


def init_human_annotations():
    if not os.path.exists(HUMAN_ANNOTATIONS_PATH):
        with open(HUMAN_ANNOTATIONS_PATH, 'w') as f:
            json.dump({}, f)
    return load_human_annotations()


def load_human_annotations():
    try:
        with open(HUMAN_ANNOTATIONS_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_human_annotation(entry_id, annotation):
    annotations = load_human_annotations()

    # Add timestamp to the annotation
    annotation['timestamp'] = datetime.datetime.now().isoformat()

    # Store the annotation
    annotations[entry_id] = annotation

    with open(HUMAN_ANNOTATIONS_PATH, 'w') as f:
        json.dump(annotations, f, indent=2)

    return True


@app.route('/')
def index():
    """Serve the main review interface page"""
    return render_template('review_interface.html')


@app.route('/annotate')
def annotate():
    """Serve the human annotation interface page"""
    return render_template('annotation_interface.html')


@app.route('/api/save-annotation', methods=['POST'])
def save_annotation():
    """Save a human annotation to the database"""
    data = request.json
    if not data or 'entry_id' not in data or 'human_response' not in data:
        return jsonify({'success': False, 'error': 'Invalid data'}), 400

    success = save_human_annotation(data['entry_id'], {
        'human_response': data['human_response'],
        'confidence': data.get('confidence', 3),  # Default medium confidence
        'notes': data.get('notes', '')
    })

    return jsonify({'success': success})


@app.route('/api/human-annotations')
def get_human_annotations():
    """Return all human annotations"""
    annotations = load_human_annotations()
    return jsonify(annotations)


@app.route('/api/review-data')
def get_review_data():
    """Process and return the combined data for review"""
    # Load the model results
    with open(RESULTS_PATH, 'r') as f:
        model_results = json.load(f)

    # Load the CSV data using pandas for easier handling
    df = pd.read_csv(CSV_PATH, encoding='utf-8')

    human_annotations = load_human_annotations()

    # Process and combine the data
    combined_data = []
    for entry in model_results:
        video_id = entry['video_id']
        context_scene_number = entry['context_scene_number']
        true_reaction_scene_number = entry['true_reaction_scene_number']
        random_reaction_scene_number = entry['random_reaction_scene_number']

        # Find the corresponding rows in the CSV data
        context_row = df[(df['video_id'] == video_id) &
                         (df['context_scene_number'] == context_scene_number)].to_dict('records')

        true_reaction_row = df[(df['video_id'] == video_id) &
                               (df['context_scene_number'] == true_reaction_scene_number)].to_dict('records')

        random_reaction_row = df[(df['video_id'] == video_id) &
                                 (df['context_scene_number'] == random_reaction_scene_number)].to_dict('records')

        # Extract data if found
        context_data = context_row[0] if context_row else {}
        true_reaction_data = true_reaction_row[0] if true_reaction_row else {}
        random_reaction_data = random_reaction_row[0] if random_reaction_row else {
        }

        # Determine which reaction is option A and which is option B
        option_a_data = true_reaction_data if entry['correct_answer'] == 'A' else random_reaction_data
        option_b_data = random_reaction_data if entry['correct_answer'] == 'A' else true_reaction_data

        # Create relative paths for images that will work with our server
        context_image = context_data.get('context_middle_frame', '')
        if context_image:
            context_image = os.path.relpath(context_image, OUTPUT_DIR)

        option_a_image = ''
        if option_a_data:
            option_a_image = option_a_data.get('context_middle_frame', '')
            if option_a_image:
                option_a_image = os.path.relpath(option_a_image, OUTPUT_DIR)

        option_b_image = ''
        if option_b_data:
            option_b_image = option_b_data.get('context_middle_frame', '')
            if option_b_image:
                option_b_image = os.path.relpath(option_b_image, OUTPUT_DIR)

        human_annotation = human_annotations.get(str(entry['entry_id']), {})

        # Create a combined entry
        combined_entry = {
            **entry,
            'context_transcript': context_data.get('context_transcript', ''),
            'context_image_path': context_image,
            'option_a_transcript': '' if pd.isna(option_a_data.get('reaction_transcript', '')) else option_a_data.get('reaction_transcript', ''),
            'option_a_image_path': option_a_image,
            'option_b_transcript': '' if pd.isna(option_b_data.get('reaction_transcript', '')) else option_b_data.get('reaction_transcript', ''),
            'option_b_image_path': option_b_image,
            'is_correct': entry['correct_answer'] == entry['model_response'] or
            (entry['correct_answer'] == 'A' and entry['model_response'] == 'Reaction A') or
            (entry['correct_answer'] ==
             'B' and entry['model_response'] == 'Reaction B'),
            'human_annotation': human_annotation
        }

        combined_data.append(combined_entry)

    return jsonify(combined_data)


@app.route('/<path:filename>')
def serve_file(filename):
    """Serve frame images from the frames directory"""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/stats')
def stats():
    """Serve the statistics page"""
    return render_template('stats_page.html')


@app.route('/api/stats-data')
def get_stats_data():
    """Return statistics about model and human performance"""
    # Load the model results
    with open(RESULTS_PATH, 'r') as f:
        model_results = json.load(f)

    # Load human annotations
    human_annotations = load_human_annotations()

    # Calculate statistics
    total_entries = len(model_results)
    model_correct = 0
    human_correct = 0
    human_annotated = 0

    # For more detailed analysis
    stats = {
        'total': total_entries,
        'model_correct': 0,
        'model_accuracy': 0,
        'human_annotated': 0,
        'human_correct': 0,
        'human_accuracy': 0,
        'agreement_count': 0,
        'agreement_rate': 0,
        'confidence_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        'confidence_accuracy': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        'confidence_counts': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    }

    for entry in model_results:
        entry_id = str(entry['entry_id'])

        # Check model correctness
        model_is_correct = (
            entry['correct_answer'] == entry['model_response'] or
            (entry['correct_answer'] == 'A' and entry['model_response'] == 'Reaction A') or
            (entry['correct_answer'] ==
             'B' and entry['model_response'] == 'Reaction B')
        )

        if model_is_correct:
            stats['model_correct'] += 1

        # Check human annotation if available
        if entry_id in human_annotations:
            human_annotation = human_annotations[entry_id]
            stats['human_annotated'] += 1

            if human_annotation['human_response'] == entry['correct_answer']:
                stats['human_correct'] += 1

            # Track agreement between model and human
            human_response = human_annotation['human_response']
            model_response = entry['model_response']
            if model_response in ['A', 'B', 'Reaction A', 'Reaction B']:
                model_choice = model_response[0] if model_response in [
                    'A', 'B'] else model_response[9]
                if model_choice == human_response:
                    stats['agreement_count'] += 1

            # Track confidence distribution and accuracy by confidence
            if 'confidence' in human_annotation:
                confidence = human_annotation['confidence']
                stats['confidence_distribution'][confidence] += 1
                stats['confidence_counts'][confidence] += 1
                if human_annotation['human_response'] == entry['correct_answer']:
                    stats['confidence_accuracy'][confidence] += 1

    # Calculate percentages
    if total_entries > 0:
        stats['model_accuracy'] = (
            stats['model_correct'] / total_entries) * 100

    if stats['human_annotated'] > 0:
        stats['human_accuracy'] = (
            stats['human_correct'] / stats['human_annotated']) * 100
        stats['agreement_rate'] = (
            stats['agreement_count'] / stats['human_annotated']) * 100

        # Calculate accuracy by confidence level
        for level in range(1, 6):
            if stats['confidence_counts'][level] > 0:
                stats['confidence_accuracy'][level] = (
                    stats['confidence_accuracy'][level] /
                    stats['confidence_counts'][level]
                ) * 100

    return jsonify(stats)


if __name__ == '__main__':
    # Make sure the template directory exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    init_human_annotations()

    # Create the HTML template
    with open('templates/review_interface.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Context-Reaction Experiment Review</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        .review-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .entry {
            border: 1px solid #ddd;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .entry-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            background-color: #eee;
            padding: 10px;
            border-radius: 5px;
        }
        .scenes {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .scene {
            flex: 1;
            min-width: 300px;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
            background-color: white;
        }
        .scene-image {
            width: 100%;
            height: auto;
            max-height: 300px;
            object-fit: contain;
            margin-bottom: 10px;
        }
        .transcript {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 3px;
            font-size: 14px;
        }
        .reaction-transcript {
            display: none; /* Hidden by default */
        }
        .controls {
            margin: 20px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        .model-info {
            background-color: #e6f7ff;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .correct {
            color: green;
            font-weight: bold;
        }
        .incorrect {
            color: red;
            font-weight: bold;
        }
        .flag-button {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
        }
        .flag-button.flagged {
            background-color: #ffcccc;
            border-color: #ff6666;
        }
        .pagination {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            gap: 10px;
        }
        .pagination button {
            padding: 8px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .pagination button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .page-info {
            padding: 8px 15px;
            background-color: #f1f1f1;
            border-radius: 4px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 18px;
        }
        .human-annotation {
            margin-top: 15px;
            padding: 10px;
            background-color: #f0f7ff;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }
        .menu-bar {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .menu-button {
            padding: 8px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="review-container">
        <div class="menu-bar">
            <h1>Context-Reaction Experiment Review</h1>
            <div>
                <a href="/annotate" class="menu-button">Annotation Interface</a>
                <a href="/stats" class="menu-button">View Statistics</a>
            </div>
        </div>
        
        <div class="controls">
            <label>
                <input type="checkbox" id="show-transcripts"> Show Reaction Transcripts
            </label>
            <p><small>Note: The model does not see reaction transcripts during evaluation.</small></p>
        </div>
        
        <div class="pagination">
            <button id="prev-page">Previous</button>
            <div class="page-info">Page <span id="current-page">1</span> of <span id="total-pages">1</span></div>
            <button id="next-page">Next</button>
        </div>
        
        <div id="entries-container">
            <div class="loading">Loading data...</div>
        </div>
        
        <div class="pagination">
            <button id="prev-page-bottom">Previous</button>
            <div class="page-info">Page <span id="current-page-bottom">1</span> of <span id="total-pages-bottom">1</span></div>
            <button id="next-page-bottom">Next</button>
        </div>
    </div>

    <script>
        let allEntries = [];
        let currentPage = 1;
        const entriesPerPage = 5;
        
        // Load data from the server
        async function loadData() {
            try {
                const response = await fetch('/api/review-data');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                allEntries = await response.json();
                renderPage();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('entries-container').innerHTML = 
                    `<div class="error">Error loading data: ${error.message}</div>`;
            }
        }
        
        // Render the current page of entries
        function renderPage() {
            const container = document.getElementById('entries-container');
            container.innerHTML = '';
            
            const startIndex = (currentPage - 1) * entriesPerPage;
            const endIndex = Math.min(startIndex + entriesPerPage, allEntries.length);
            
            if (allEntries.length === 0) {
                container.innerHTML = '<div class="no-data">No data available</div>';
                return;
            }
            
            for (let i = startIndex; i < endIndex; i++) {
                const entry = allEntries[i];
                container.appendChild(createEntryElement(entry));
            }
            
            // Update pagination info
            const totalPages = Math.ceil(allEntries.length / entriesPerPage);
            document.getElementById('current-page').textContent = currentPage;
            document.getElementById('total-pages').textContent = totalPages;
            document.getElementById('current-page-bottom').textContent = currentPage;
            document.getElementById('total-pages-bottom').textContent = totalPages;
            
            // Enable/disable pagination buttons
            document.getElementById('prev-page').disabled = currentPage === 1;
            document.getElementById('next-page').disabled = currentPage === totalPages;
            document.getElementById('prev-page-bottom').disabled = currentPage === 1;
            document.getElementById('next-page-bottom').disabled = currentPage === totalPages;
        }
        
        // Create an HTML element for an entry
        function createEntryElement(entry) {
            const entryElement = document.createElement('div');
            entryElement.className = 'entry';
            entryElement.dataset.entryId = entry.entry_id;
            
            // Determine if the model prediction is correct
            const isCorrect = entry.is_correct;
            const correctnessClass = isCorrect ? 'correct' : 'incorrect';
            const correctnessText = isCorrect ? 'CORRECT' : 'INCORRECT';
            
            // Human annotation display
            let humanAnnotationHtml = '<p>No human annotation yet</p>';
            if (entry.human_annotation && entry.human_annotation.human_response) {
                const humanIsCorrect = entry.human_annotation.human_response === entry.correct_answer;
                const humanCorrectnessClass = humanIsCorrect ? 'correct' : 'incorrect';
                const humanCorrectnessText = humanIsCorrect ? 'CORRECT' : 'INCORRECT';
                const confidenceText = entry.human_annotation.confidence ? 
                    `Confidence: ${entry.human_annotation.confidence}/5` : '';
                const notesText = entry.human_annotation.notes ? 
                    `<p><strong>Notes:</strong> ${entry.human_annotation.notes}</p>` : '';
                
                humanAnnotationHtml = `
                    <p><strong>Human Response:</strong> <span class="${humanCorrectnessClass}">
                        Option ${entry.human_annotation.human_response} (${humanCorrectnessText})
                    </span> ${confidenceText}</p>
                    ${notesText}
                `;
            }
            
            entryElement.innerHTML = `
                <div class="entry-header">
                    <div>
                        <strong>Entry ID:</strong> ${entry.entry_id} | 
                        <strong>Video ID:</strong> ${entry.video_id} |
                        <strong>Context Scene #:</strong> ${entry.context_scene_number}
                    </div>
                    <button class="flag-button" onclick="toggleFlag(this)">Flag for Review</button>
                </div>
                
                <div class="scenes">
                    <div class="scene">
                        <h3>Context Scene</h3>
                        <img class="scene-image" src="/${entry.context_image_path}" alt="Context Scene">
                        <div class="transcript">
                            <strong>Transcript:</strong> ${entry.context_transcript || 'No transcript available'}
                        </div>
                    </div>
                    
                    <div class="scene">
                        <h3>Reaction A</h3>
                        <img class="scene-image" src="/${entry.option_a_image_path}" alt="Reaction A">
                        <div class="transcript reaction-transcript">
                            <strong>Transcript:</strong> ${entry.option_a_transcript || 'No transcript available'}
                        </div>
                    </div>
                    
                    <div class="scene">
                        <h3>Reaction B</h3>
                        <img class="scene-image" src="/${entry.option_b_image_path}" alt="Reaction B">
                        <div class="transcript reaction-transcript">
                            <strong>Transcript:</strong> ${entry.option_b_transcript || 'No transcript available'}
                        </div>
                    </div>
                </div>
                
                <div class="model-info">
                    <p><strong>Ground Truth:</strong> Option ${entry.correct_answer}</p>
                    <p><strong>Model Prediction:</strong> <span class="${correctnessClass}">${entry.model_response} (${correctnessText})</span></p>
                    <p><strong>Model Reasoning:</strong> ${entry.model_reason}</p>
                    <div class="human-annotation">
                        <h4>Human Annotation:</h4>
                        ${humanAnnotationHtml}
                    </div>
                </div>
            `;
            
            return entryElement;
        }
        
        // Function to toggle flag on an entry
        function toggleFlag(button) {
            button.classList.toggle('flagged');
            const entryId = button.closest('.entry').dataset.entryId;
            
            // Here you could save the flagged state to localStorage or send to a server
            console.log(`Entry ${entryId} flag toggled`);
            
            // Optional: Save flagged entries to localStorage
            const flaggedEntries = JSON.parse(localStorage.getItem('flaggedEntries') || '[]');
            const index = flaggedEntries.indexOf(entryId);
            
            if (index === -1 && button.classList.contains('flagged')) {
                flaggedEntries.push(entryId);
            } else if (index !== -1 && !button.classList.contains('flagged')) {
                flaggedEntries.splice(index, 1);
            }
            
            localStorage.setItem('flaggedEntries', JSON.stringify(flaggedEntries));
        }
        
        // Toggle reaction transcripts visibility
        document.getElementById('show-transcripts').addEventListener('change', function() {
            const transcripts = document.querySelectorAll('.reaction-transcript');
            transcripts.forEach(transcript => {
                transcript.style.display = this.checked ? 'block' : 'none';
            });
        });
        
        // Set up navigation
        document.getElementById('prev-page').addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                renderPage();
                window.scrollTo(0, 0);
            }
        });
        
        document.getElementById('next-page').addEventListener('click', () => {
            const totalPages = Math.ceil(allEntries.length / entriesPerPage);
            if (currentPage < totalPages) {
                currentPage++;
                renderPage();
                window.scrollTo(0, 0);
            }
        });
        
        document.getElementById('prev-page-bottom').addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                renderPage();
                window.scrollTo(0, 0);
            }
        });
        
        document.getElementById('next-page-bottom').addEventListener('click', () => {
            const totalPages = Math.ceil(allEntries.length / entriesPerPage);
            if (currentPage < totalPages) {
                currentPage++;
                renderPage();
                window.scrollTo(0, 0);
            }
        });
        
        // Load data when page loads
        window.addEventListener('DOMContentLoaded', loadData);
    </script>
</body>
</html>''')

    # Create the annotation interface HTML template
    with open('templates/annotation_interface.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Annotation Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        .annotation-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .entry {
            border: 1px solid #ddd;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .entry-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            background-color: #eee;
            padding: 10px;
            border-radius: 5px;
        }
        .scenes {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .scene {
            flex: 1;
            min-width: 300px;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
            background-color: white;
        }
        .scene-image {
            width: 100%;
            height: auto;
            max-height: 300px;
            object-fit: contain;
            margin-bottom: 10px;
        }
        .transcript {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 3px;
            font-size: 14px;
        }
        .controls {
            margin: 20px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .option-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .option-button {
            padding: 15px 30px;
            font-size: 18px;
            background-color: #f0f0f0;
            border: 2px solid #ccc;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
            flex: 1;
        }
        .option-button:hover {
            background-color: #e0e0e0;
        }
        .option-button.selected {
            background-color: #4CAF50;
            color: white;
            border-color: #2E7D32;
        }
        .navigation {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        .nav-button {
            padding: 10px 20px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .nav-button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .progress {
            text-align: center;
            font-size: 16px;
            margin: 20px 0;
        }
        .confidence-selector {
            margin-top: 15px;
        }
        .notes-area {
            width: 100%;
            margin-top: 15px;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .keyboard-shortcuts {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            padding: 10px;
            margin-top: 20px;
            border-radius: 5px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 18px;
        }
        .completed {
            color: green;
            font-weight: bold;
        }
        .menu-bar {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .menu-button {
            padding: 8px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="annotation-container">
        <div class="menu-bar">
            <h1>Human Annotation Interface</h1>
            <div>
                <a href="/" class="menu-button">Review Interface</a>
                <a href="/stats" class="menu-button">View Statistics</a>
            </div>
        </div>
        
        <div class="controls">
            <div>
                <label>
                    <input type="checkbox" id="show-transcripts"> Show Reaction Transcripts
                </label>
                <p><small>Note: Seeing transcripts may bias your judgment.</small></p>
            </div>
            <div class="progress">
                <span id="progress-text">Entry 0 of 0</span>
                <div>Completed: <span id="completed-count">0</span></div>
            </div>
        </div>
        
        <div class="keyboard-shortcuts">
            <strong>Keyboard Shortcuts:</strong>
            <ul>
                <li><kbd>1</kbd> or <kbd>A</kbd>: Select Option A</li>
                <li><kbd>2</kbd> or <kbd>B</kbd>: Select Option B</li>
                <li><kbd>←</kbd>: Previous sample</li>
                <li><kbd>→</kbd> or <kbd>Enter</kbd>: Next sample</li>
            </ul>
        </div>
        
        <div id="annotation-container">
            <div class="loading">Loading data...</div>
        </div>
    </div>

    <script>
        let allEntries = [];
        let currentIndex = 0;
        let humanAnnotations = {};
        
        // Load data from the server
        async function loadData() {
            try {
                // Load entries
                const response = await fetch('/api/review-data');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                allEntries = await response.json();
                
                // Load existing human annotations
                const annotationsResponse = await fetch('/api/human-annotations');
                if (annotationsResponse.ok) {
                    humanAnnotations = await annotationsResponse.json();
                }
                
                renderCurrentEntry();
                updateProgress();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('annotation-container').innerHTML = 
                    `<div class="error">Error loading data: ${error.message}</div>`;
            }
        }
        
        // Render the current entry
        function renderCurrentEntry() {
            const container = document.getElementById('annotation-container');
            
            if (allEntries.length === 0) {
                container.innerHTML = '<div class="no-data">No data available</div>';
                return;
            }
            
            if (currentIndex >= allEntries.length) {
                container.innerHTML = '<div class="completed">All entries have been reviewed!</div>';
                return;
            }
            
            const entry = allEntries[currentIndex];
            const entryId = entry.entry_id.toString();
            const existingAnnotation = humanAnnotations[entryId];
            
            container.innerHTML = '';
            const entryElement = document.createElement('div');
            entryElement.className = 'entry';
            entryElement.dataset.entryId = entryId;
            
            entryElement.innerHTML = `
                <div class="entry-header">
                    <div>
                        <strong>Entry ID:</strong> ${entry.entry_id} | 
                        <strong>Video ID:</strong> ${entry.video_id} |
                        <strong>Context Scene #:</strong> ${entry.context_scene_number}
                    </div>
                </div>
                
                <div class="scenes">
                    <div class="scene">
                        <h3>Context Scene</h3>
                        <img class="scene-image" src="/${entry.context_image_path}" alt="Context Scene">
                        <div class="transcript">
                            <strong>Transcript:</strong> ${entry.context_transcript || 'No transcript available'}
                        </div>
                    </div>
                    
                    <div class="scene">
                        <h3>Reaction A</h3>
                        <img class="scene-image" src="/${entry.option_a_image_path}" alt="Reaction A">
                        <div class="transcript reaction-transcript" style="display: none;">
                            <strong>Transcript:</strong> ${entry.option_a_transcript || 'No transcript available'}
                        </div>
                    </div>
                    
                    <div class="scene">
                        <h3>Reaction B</h3>
                        <img class="scene-image" src="/${entry.option_b_image_path}" alt="Reaction B">
                        <div class="transcript reaction-transcript" style="display: none;">
                            <strong>Transcript:</strong> ${entry.option_b_transcript || 'No transcript available'}
                        </div>
                    </div>
                </div>
                
                <div class="option-buttons">
                    <button class="option-button" id="option-a" onclick="selectOption('A')">
                        Option A (1)
                    </button>
                    <button class="option-button" id="option-b" onclick="selectOption('B')">
                        Option B (2)
                    </button>
                </div>
                
                <div class="confidence-selector">
                    <label><strong>Confidence:</strong></label>
                    <select id="confidence-select">
                        <option value="1">1 - Very Uncertain</option>
                        <option value="2">2 - Somewhat Uncertain</option>
                        <option value="3" selected>3 - Moderate Confidence</option>
                        <option value="4">4 - Confident</option>
                        <option value="5">5 - Very Confident</option>
                    </select>
                </div>
                
                <div>
                    <label><strong>Notes:</strong></label>
                    <textarea id="notes-area" class="notes-area" rows="2" placeholder="Optional notes about this annotation"></textarea>
                </div>
                
                <div class="navigation">
                    <button class="nav-button" id="prev-button" onclick="navigateToPrevious()">
                        ← Previous
                    </button>
                    <button class="nav-button" id="next-button" onclick="navigateToNext()">
                        Next →
                    </button>
                </div>
            `;
            
            container.appendChild(entryElement);
            
            // If there's an existing annotation, select it
            if (existingAnnotation) {
                selectOption(existingAnnotation.human_response, false);
                
                // Set confidence and notes if available
                if (existingAnnotation.confidence) {
                    document.getElementById('confidence-select').value = existingAnnotation.confidence;
                }
                if (existingAnnotation.notes) {
                    document.getElementById('notes-area').value = existingAnnotation.notes;
                }
            }
            
            // Update button states
            document.getElementById('prev-button').disabled = currentIndex === 0;
        }
        
        // Select an option (A or B)
        function selectOption(option, saveAndAdvance = true) {
            const optionAButton = document.getElementById('option-a');
            const optionBButton = document.getElementById('option-b');
            
            optionAButton.classList.remove('selected');
            optionBButton.classList.remove('selected');
            
            if (option === 'A') {
                optionAButton.classList.add('selected');
            } else if (option === 'B') {
                optionBButton.classList.add('selected');
            }
            
            if (saveAndAdvance) {
                saveAnnotation(option);
            }
        }
        
        // Save the current annotation
        async function saveAnnotation(option) {
            const entryId = allEntries[currentIndex].entry_id.toString();
            const confidence = document.getElementById('confidence-select').value;
            const notes = document.getElementById('notes-area').value;
            
            try {
                const response = await fetch('/api/save-annotation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        entry_id: entryId,
                        human_response: option,
                        confidence: parseInt(confidence),
                        notes: notes
                    }),
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                // Update local cache
                humanAnnotations[entryId] = {
                    human_response: option,
                    confidence: parseInt(confidence),
                    notes: notes
                };
                
                // Update progress
                updateProgress();
                
                // Navigate to next entry
                navigateToNext();
            } catch (error) {
                console.error('Error saving annotation:', error);
                alert('Failed to save annotation. Please try again.');
            }
        }
        
        // Navigate to the previous entry
        function navigateToPrevious() {
            if (currentIndex > 0) {
                currentIndex--;
                renderCurrentEntry();
            }
        }
        
        // Navigate to the next entry
        function navigateToNext() {
            if (currentIndex < allEntries.length - 1) {
                currentIndex++;
                renderCurrentEntry();
            }
        }
        
        // Update progress information
        function updateProgress() {
            const completedCount = Object.keys(humanAnnotations).length;
            document.getElementById('progress-text').textContent = 
                `Entry ${currentIndex + 1} of ${allEntries.length}`;
            document.getElementById('completed-count').textContent = completedCount;
        }
        
        // Toggle reaction transcripts visibility
        document.getElementById('show-transcripts').addEventListener('change', function() {
            const transcripts = document.querySelectorAll('.reaction-transcript');
            transcripts.forEach(transcript => {
                transcript.style.display = this.checked ? 'block' : 'none';
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            // Ignore keyboard shortcuts when typing in textarea
            if (event.target.tagName === 'TEXTAREA') return;
            
            switch(event.key) {
                case '1':
                case 'a':
                case 'A':
                    selectOption('A');
                    break;
                case '2':
                case 'b':
                case 'B':
                    selectOption('B');
                    break;
                case 'ArrowLeft':
                    navigateToPrevious();
                    break;
                case 'ArrowRight':
                case 'Enter':
                    // Only navigate if an option is selected
                    const optionA = document.getElementById('option-a');
                    const optionB = document.getElementById('option-b');
                    if (optionA.classList.contains('selected') || optionB.classList.contains('selected')) {
                        navigateToNext();
                    }
                    break;
            }
        });
        
        // Load data when page loads
        window.addEventListener('DOMContentLoaded', loadData);
    </script>
</body>
</html>''')

    # Create the statistics page HTML template
    with open('templates/stats_page.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Context-Reaction Experiment Statistics</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        .stats-container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .stats-card {
            border: 1px solid #ddd;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .stats-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .menu-button {
            padding: 8px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .metric-card {
            background-color: white;
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        .chart-container {
            height: 300px;
            margin-top: 20px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 18px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="stats-container">
        <div class="stats-header">
            <h1>Context-Reaction Experiment Statistics</h1>
            <div>
                <a href="/" class="menu-button">Review Interface</a>
                <a href="/annotate" class="menu-button">Annotation Interface</a>
            </div>
        </div>
        
        <div id="stats-content">
            <div class="loading">Loading statistics...</div>
        </div>
    </div>

    <script>
        // Load statistics data
        async function loadStats() {
            try {
                const response = await fetch('/api/stats-data');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const stats = await response.json();
                renderStats(stats);
            } catch (error) {
                console.error('Error loading statistics:', error);
                document.getElementById('stats-content').innerHTML = 
                    `<div class="error">Error loading statistics: ${error.message}</div>`;
            }
        }
        
        // Render the statistics
        function renderStats(stats) {
            const container = document.getElementById('stats-content');
            container.innerHTML = '';
            
            // Overview section
            const overviewSection = document.createElement('div');
            overviewSection.className = 'stats-card';
            overviewSection.innerHTML = `
                <h2>Overview</h2>
                <div class="stats-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Entries</div>
                        <div class="metric-value">${stats.total}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Human Annotated</div>
                        <div class="metric-value">${stats.human_annotated}</div>
                        <div class="metric-label">${((stats.human_annotated / stats.total) * 100).toFixed(1)}% of total</div>
                    </div>
                </div>
            `;
            container.appendChild(overviewSection);
            
            // Accuracy comparison section
            const accuracySection = document.createElement('div');
            accuracySection.className = 'stats-card';
            accuracySection.innerHTML = `
                <h2>Accuracy Comparison</h2>
                <div class="stats-grid">
                    <div class="metric-card">
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value">${stats.model_accuracy.toFixed(1)}%</div>
                        <div class="metric-label">${stats.model_correct} / ${stats.total} correct</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Human Accuracy</div>
                        <div class="metric-value">${stats.human_accuracy.toFixed(1)}%</div>
                        <div class="metric-label">${stats.human_correct} / ${stats.human_annotated} correct</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Human-Model Agreement</div>
                        <div class="metric-value">${stats.agreement_rate.toFixed(1)}%</div>
                        <div class="metric-label">${stats.agreement_count} / ${stats.human_annotated} entries</div>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="accuracy-chart"></canvas>
                </div>
            `;
            container.appendChild(accuracySection);
            
            // Confidence analysis section
            const confidenceSection = document.createElement('div');
            confidenceSection.className = 'stats-card';
            confidenceSection.innerHTML = `
                <h2>Confidence Analysis</h2>
                <div class="chart-container">
                    <canvas id="confidence-chart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="confidence-accuracy-chart"></canvas>
                </div>
            `;
            container.appendChild(confidenceSection);
            
            // Create charts
            createAccuracyChart(stats);
            createConfidenceDistributionChart(stats);
            createConfidenceAccuracyChart(stats);
        }
        
        // Create accuracy comparison chart
        function createAccuracyChart(stats) {
            const ctx = document.getElementById('accuracy-chart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Model', 'Human'],
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: [stats.model_accuracy, stats.human_accuracy],
                        backgroundColor: [
                            'rgba(54, 162, 235, 0.6)',
                            'rgba(75, 192, 192, 0.6)'
                        ],
                        borderColor: [
                            'rgba(54, 162, 235, 1)',
                            'rgba(75, 192, 192, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Accuracy (%)'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Model vs Human Accuracy'
                        }
                    }
                }
            });
        }
        
        // Create confidence distribution chart
        function createConfidenceDistributionChart(stats) {
            const ctx = document.getElementById('confidence-chart').getContext('2d');
            const confidenceLevels = [1, 2, 3, 4, 5];
            const confidenceCounts = confidenceLevels.map(level => stats.confidence_distribution[level]);
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Very Uncertain (1)', 'Somewhat Uncertain (2)', 'Moderate (3)', 'Confident (4)', 'Very Confident (5)'],
                    datasets: [{
                        label: 'Number of Annotations',
                        data: confidenceCounts,
                        backgroundColor: 'rgba(153, 102, 255, 0.6)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Distribution of Confidence Levels'
                        }
                    }
                }
            });
        }
        
        // Create confidence vs accuracy chart
        function createConfidenceAccuracyChart(stats) {
            const ctx = document.getElementById('confidence-accuracy-chart').getContext('2d');
            const confidenceLevels = [1, 2, 3, 4, 5];
            const accuracyByConfidence = confidenceLevels.map(level => stats.confidence_accuracy[level]);
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Very Uncertain (1)', 'Somewhat Uncertain (2)', 'Moderate (3)', 'Confident (4)', 'Very Confident (5)'],
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: accuracyByConfidence,
                        backgroundColor: 'rgba(255, 159, 64, 0.6)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Accuracy (%)'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Accuracy by Confidence Level'
                        }
                    }
                }
            });
        }
        
        // Load data when page loads
        window.addEventListener('DOMContentLoaded', loadStats);
    </script>
</body>
</html>''')

    print("Starting server at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
