// Get references to DOM elements
const videoInput = document.getElementById('videoInput');
const analyzeButton = document.getElementById('analyzeButton');
const statusArea = document.getElementById('statusArea');
const statusMessage = document.getElementById('statusMessage');
const resultsArea = document.getElementById('resultsArea');
const fileNameSpan = document.getElementById('fileName');
const verdictSpan = document.getElementById('verdict');
const totalFramesSpan = document.getElementById('totalFrames');
const realCountSpan = document.getElementById('realCount');
const deepfakeCountSpan = document.getElementById('deepfakeCount');
// const frameDetailsList = document.getElementById('frameDetailsList'); // Uncomment if using detailed list

// --- Configuration ---
const API_ENDPOINT = "http://127.0.0.1:8000/analyze/"; // Make sure this matches your FastAPI server address

// --- Event Listener ---
analyzeButton.addEventListener('click', handleAnalysisRequest);

// Add event listener to file input to update label or show filename (optional)
videoInput.addEventListener('change', () => {
    const fileLabel = document.querySelector('.file-label');
    if (videoInput.files.length > 0) {
        fileLabel.textContent = videoInput.files[0].name; // Show filename
    } else {
        fileLabel.textContent = 'Choose Video File'; // Reset label
    }
    // Clear previous results when a new file is selected
    hideResults();
    hideStatus();
});


// --- Functions ---

function showStatus(message, isError = false) {
    statusMessage.textContent = message;
    statusArea.className = 'status ' + (isError ? 'error' : 'loading'); // Apply CSS class
    statusArea.style.display = 'block';
}

function hideStatus() {
    statusArea.style.display = 'none';
    statusMessage.textContent = '';
}

function showResults(data) {
    fileNameSpan.textContent = data.filename || 'N/A';
    verdictSpan.textContent = data.verdict || 'Error';
    totalFramesSpan.textContent = data.total_frames_processed !== undefined ? data.total_frames_processed : 'N/A';
    realCountSpan.textContent = data.real_frames !== undefined ? data.real_frames : 'N/A';
    deepfakeCountSpan.textContent = data.deepfake_frames !== undefined ? data.deepfake_frames : 'N/A';

    // Apply verdict-specific styling
    verdictSpan.className = 'verdict'; // Reset classes first
    if (data.verdict) {
        // Replace spaces for CSS class compatibility if needed, e.g., "Inconclusive (Equal)" -> "Inconclusive"
        const verdictClass = data.verdict.split(' ')[0].replace(/[^a-zA-Z0-9]/g, '');
        verdictSpan.classList.add(`verdict-${verdictClass}`);
    }


    // Optional: Display detailed frame results
    /*
    frameDetailsList.innerHTML = ''; // Clear previous details
    if (data.frame_results && data.frame_results.length > 0) {
        data.frame_results.forEach((frameResult, index) => {
            const listItem = document.createElement('li');
            let detailText = `Frame ${frameResult.input_index}: Status - ${frameResult.status}`;
            if (frameResult.status === 'success' && frameResult.prediction) {
                detailText += `, Prediction - ${frameResult.prediction.label} (${(frameResult.prediction.confidence * 100).toFixed(1)}%)`;
            } else if (frameResult.status === 'error') {
                detailText += `, Error - ${frameResult.error_message}`;
            }
            listItem.textContent = detailText;
            frameDetailsList.appendChild(listItem);
        });
    } else {
         frameDetailsList.innerHTML = '<li>No detailed frame data available.</li>';
    }
    */

    resultsArea.style.display = 'block';
}

function hideResults() {
    resultsArea.style.display = 'none';
}

async function handleAnalysisRequest() {
    const file = videoInput.files[0];

    // 1. Basic Validation
    if (!file) {
        showStatus("Please select a video file first.", true); // Show error
        return;
    }

    // 2. Prepare UI for Upload/Analysis
    analyzeButton.disabled = true;
    hideResults();
    showStatus("Uploading and analyzing video... Please wait."); // Show loading

    // 3. Prepare FormData
    const formData = new FormData();
    formData.append("file", file); // Key must match FastAPI parameter name

    // 4. Make API Request
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData,
            // Headers are NOT needed here for FormData; browser sets Content-Type correctly
        });

        // 5. Handle Response
        const responseData = await response.json(); // Try to parse JSON regardless of status

        if (!response.ok) {
            // Handle HTTP errors (e.g., 4xx, 5xx)
            const errorMsg = responseData.detail || `Server error: ${response.status} ${response.statusText}`;
            console.error("API Error Response:", responseData);
            showStatus(`Analysis failed: ${errorMsg}`, true);
        } else {
            // Handle Success (2xx status)
            console.log("API Success Response:", responseData);
            hideStatus(); // Hide loading message
            showResults(responseData); // Display the results
        }

    } catch (error) {
        // Handle Network errors or other fetch issues
        console.error("Fetch Error:", error);
        showStatus(`An error occurred: ${error.message}. Check console and ensure the server is running.`, true);
    } finally {
        // 6. Re-enable button regardless of success/failure
        analyzeButton.disabled = false;
    }
}
