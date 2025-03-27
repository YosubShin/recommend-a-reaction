from flask import Flask, render_template, jsonify, send_from_directory
import json
import csv
import os
import pandas as pd
from pathlib import Path

app = Flask(__name__, template_folder='templates', static_folder='static')

# Define paths to your data files
RESULTS_PATH = 'output/vlm_experiment/context_reaction_results_20250327_105647.json'
CSV_PATH = 'output/context_reaction_pairs_20250327_105647.csv'
FRAMES_DIR = 'output/frames'


@app.route('/')
def index():
    """Serve the main review interface page"""
    return render_template('review_interface.html')


@app.route('/api/review-data')
def get_review_data():
    """Process and return the combined data for review"""
    # Load the model results
    with open(RESULTS_PATH, 'r') as f:
        model_results = json.load(f)

    # Load the CSV data using pandas for easier handling
    df = pd.read_csv(CSV_PATH)

    # Process and combine the data
    combined_data = []
    for entry in model_results:
        video_id = entry['video_id']
        context_scene_number = str(entry['context_scene_number']).zfill(3)
        true_reaction_scene_number = str(
            entry['true_reaction_scene_number']).zfill(3)
        random_reaction_scene_number = str(
            entry['random_reaction_scene_number']).zfill(3)

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
            context_image = f"/frames/{video_id}/{os.path.basename(context_image)}"

        option_a_image = ''
        if option_a_data:
            option_a_image = option_a_data.get('context_middle_frame', '')
            if option_a_image:
                option_a_image = f"/frames/{video_id}/{os.path.basename(option_a_image)}"

        option_b_image = ''
        if option_b_data:
            option_b_image = option_b_data.get('context_middle_frame', '')
            if option_b_image:
                option_b_image = f"/frames/{video_id}/{os.path.basename(option_b_image)}"

        # Create a combined entry
        combined_entry = {
            **entry,
            'context_transcript': context_data.get('context_transcript', ''),
            'context_image_path': context_image,
            'option_a_transcript': option_a_data.get('context_transcript', '') if option_a_data else '',
            'option_a_image_path': option_a_image,
            'option_b_transcript': option_b_data.get('context_transcript', '') if option_b_data else '',
            'option_b_image_path': option_b_image,
            'is_correct': entry['correct_answer'] == entry['model_response'] or
            (entry['correct_answer'] == 'A' and entry['model_response'] == 'Reaction A') or
            (entry['correct_answer'] ==
             'B' and entry['model_response'] == 'Reaction B')
        }

        combined_data.append(combined_entry)

    return jsonify(combined_data)


@app.route('/frames/<video_id>/<filename>')
def serve_frame(video_id, filename):
    """Serve frame images from the frames directory"""
    # Extract scene number from filename
    parts = filename.split('_')
    if len(parts) >= 2:
        scene_part = parts[1]
        scene_number = scene_part.replace('scene', '')

        # Construct the path to the image
        scene_dir = f"Scene-{scene_number}"
        image_path = os.path.join(FRAMES_DIR, video_id, scene_dir)

        return send_from_directory(image_path, filename)

    return "Image not found", 404


if __name__ == '__main__':
    # Make sure the template directory exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

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
    </style>
</head>
<body>
    <div class="review-container">
        <h1>Context-Reaction Experiment Review</h1>
        
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
            
            entryElement.innerHTML = `
                <div class="entry-header">
                    <div>
                        <strong>Entry ID:</strong> ${entry.entry_id} | 
                        <strong>Video ID:</strong> ${entry.video_id}
                    </div>
                    <button class="flag-button" onclick="toggleFlag(this)">Flag for Review</button>
                </div>
                
                <div class="scenes">
                    <div class="scene">
                        <h3>Context Scene</h3>
                        <img class="scene-image" src="${entry.context_image_path}" alt="Context Scene">
                        <div class="transcript">
                            <strong>Transcript:</strong> ${entry.context_transcript || 'No transcript available'}
                        </div>
                    </div>
                    
                    <div class="scene">
                        <h3>Option A</h3>
                        <img class="scene-image" src="${entry.option_a_image_path}" alt="Option A">
                        <div class="transcript">
                            <strong>Transcript:</strong> ${entry.option_a_transcript || 'No transcript available'}
                        </div>
                    </div>
                    
                    <div class="scene">
                        <h3>Option B</h3>
                        <img class="scene-image" src="${entry.option_b_image_path}" alt="Option B">
                        <div class="transcript">
                            <strong>Transcript:</strong> ${entry.option_b_transcript || 'No transcript available'}
                        </div>
                    </div>
                </div>
                
                <div class="model-info">
                    <p><strong>Ground Truth:</strong> Option ${entry.correct_answer}</p>
                    <p><strong>Model Prediction:</strong> <span class="${correctnessClass}">${entry.model_response} (${correctnessText})</span></p>
                    <p><strong>Model Reasoning:</strong> ${entry.model_reason}</p>
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

    print("Starting server at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
