document.addEventListener('DOMContentLoaded', function() {
    // --- DOM Elements ---
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
    const confidenceMeter = document.getElementById('confidenceMeter');
    const confidenceValue = document.getElementById('confidenceValue');
    const mobileMenuButton = document.getElementById('mobileMenuButton');
    const mobileMenu = document.getElementById('mobileMenu');
    const fileUploadContainer = document.querySelector('.file-upload-container'); // Needed for Drag & Drop
    const fileLabelText = document.querySelector('.file-label-text'); // More specific selector
    const fileInfoText = document.querySelector('.file-info');       // More specific selector

    // --- Configuration ---
    const API_ENDPOINT = "/analyze"; // Ensure this matches your Flask/backend route
    const MAX_FILE_SIZE_MB = 50;
    const ALLOWED_FILE_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/avi']; // Added video/avi for consistency

    // --- Event Listeners ---
    analyzeButton.addEventListener('click', handleAnalysisRequest);
    videoInput.addEventListener('change', handleFileSelect); // Handles file selection via click
    mobileMenuButton.addEventListener('click', toggleMobileMenu);

    // Close mobile menu when clicking outside
    document.addEventListener('click', function(event) {
        // Close if click is outside the menu itself AND outside the button that opens it
        if (!mobileMenu.contains(event.target) && !mobileMenuButton.contains(event.target)) {
             if (mobileMenu.style.display === 'flex') { // Only close if it's open
                mobileMenu.style.display = 'none';
             }
        }
    });

    // --- Drag and Drop Event Listeners ---
    if (fileUploadContainer) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileUploadContainer.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false); // Prevent browser default behavior for whole page
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            fileUploadContainer.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            fileUploadContainer.addEventListener(eventName, unhighlight, false);
        });

        fileUploadContainer.addEventListener('drop', handleDrop, false);
    } else {
        console.warn("Drag and drop container (.file-upload-container) not found.");
    }

    // --- Functions ---

    function toggleMobileMenu() {
        // Use computed style for more reliable check if initially hidden via CSS
        const currentDisplay = window.getComputedStyle(mobileMenu).display;
        mobileMenu.style.display = (currentDisplay === 'none' || currentDisplay === '') ? 'flex' : 'none';
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        if (fileUploadContainer) fileUploadContainer.classList.add('highlight');
    }

    function unhighlight() {
        if (fileUploadContainer) fileUploadContainer.classList.remove('highlight');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            // Validate the first dropped file before assigning
            const file = files[0];
             if (!ALLOWED_FILE_TYPES.includes(file.type)) {
                showStatus(`Unsupported file type: ${file.type}. Please use MP4, MOV, or AVI.`, true);
                return;
            }
            if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
                 showStatus(`File "${file.name}" is too large (>${MAX_FILE_SIZE_MB}MB).`, true);
                return;
            }

            videoInput.files = files; // Assign the dropped files to the input
            updateFileInfoDisplay(file); // Update display manually as 'change' might not fire reliably
            hideResults();
            hideStatus();
        }
    }

    function handleFileSelect() {
        // This function is triggered when the file input changes (e.g., user clicks 'Choose File')
        if (videoInput.files.length > 0) {
            const file = videoInput.files[0];
             // Optional: Add validation here too, although it's also done before upload
            updateFileInfoDisplay(file);
        } else {
            resetFileInfoDisplay();
        }
        hideResults();
        hideStatus();
    }

    function updateFileInfoDisplay(file) {
        if (file && fileLabelText && fileInfoText) {
            fileLabelText.textContent = file.name;
            fileInfoText.textContent = `${formatFileSize(file.size)} • ${file.type || 'N/A'}`;
        } else {
             resetFileInfoDisplay(); // Reset if file is invalid or elements missing
        }
    }

     function resetFileInfoDisplay() {
         if (fileLabelText && fileInfoText) {
            fileLabelText.textContent = 'Choose Video File or Drag Here';
            fileInfoText.textContent = `MP4, MOV, AVI up to ${MAX_FILE_SIZE_MB}MB`;
         }
         // Clear the actual input value if needed
         videoInput.value = ''; // Clears the selected file from the input element itself
    }


    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']; // Added TB just in case
        // Handle potential edge case where bytes is negative or not a number
        if (bytes < 0 || isNaN(bytes)) return 'Invalid size';
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        // Ensure index i is within the bounds of the sizes array
        const sizeIndex = Math.min(i, sizes.length - 1);
        return parseFloat((bytes / Math.pow(k, sizeIndex)).toFixed(2)) + ' ' + sizes[sizeIndex];
    }

    function showStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusArea.className = 'status-area ' + (isError ? 'error' : 'info'); // Use info class for non-errors
        statusArea.style.display = 'block';
        resultsArea.style.display = 'none'; // Hide results when status is shown
    }

    function hideStatus() {
        statusArea.style.display = 'none';
        statusMessage.textContent = ''; // Clear message
    }

    function showResults(responseData) { // Expects the full {success: true, results: {...}} object
        hideStatus(); // Hide any previous status messages

        if (!responseData || typeof responseData !== 'object') {
             showStatus("Received invalid response from server.", true);
             console.error("Invalid response:", responseData);
             return;
        }

        if (!responseData.success || !responseData.results) {
            const errorMsg = responseData.error || responseData.message || "Analysis failed. Results key missing in response.";
            showStatus(`Analysis Error: ${errorMsg}`, true);
            console.error("API Error or missing results:", responseData);
            return;
        }

        const data = responseData.results; // Access the nested results object

        // Check for specific non-fatal messages or warnings within results
        if (data.error_message) {
            // Display as an informational status, not a full error, but don't show results grid
            showStatus(`Analysis Note: ${data.error_message}`, false); // Show as info, not error
            resultsArea.style.display = 'none'; // Ensure results grid remains hidden
            return;
        }

        // --- Populate Results Area ---
        fileNameSpan.textContent = data.file_name || 'N/A';
        verdictSpan.textContent = data.verdict || 'UNKNOWN'; // Handle missing verdict
        totalFramesSpan.textContent = data.frames_analyzed ?? 'N/A';
        realCountSpan.textContent = data.real_frames ?? 'N/A';
        deepfakeCountSpan.textContent = data.fake_frames ?? 'N/A';

        // Use the confidence provided by the backend
        let confidence = 0;
        if (data.confidence !== undefined && data.confidence !== null && !isNaN(data.confidence)) {
            confidence = Math.max(0, Math.min(100, Math.round(data.confidence))); // Clamp between 0-100
        } else {
             console.warn("Confidence value missing or invalid from backend:", data.confidence);
        }

        confidenceMeter.style.width = `${confidence}%`;
        confidenceValue.textContent = `${confidence}%`;

        // Update meter color based on confidence (adjust thresholds if needed)
        // Consider linking color more directly to verdict? Or keep confidence-based?
        // Let's stick to confidence-based colors for now.
        if (confidence >= 70) {
            confidenceMeter.className = 'confidence-meter-bar high'; // Use classes for styling
        } else if (confidence >= 40) {
            confidenceMeter.className = 'confidence-meter-bar medium';
        } else {
            confidenceMeter.className = 'confidence-meter-bar low';
        }
        // Note: You'll need corresponding CSS rules for .high, .medium, .low background colors

        // Apply verdict styling based on backend verdict
        verdictSpan.className = 'verdict-badge'; // Reset classes
        const verdictUpper = (data.verdict || 'UNKNOWN').toUpperCase(); // Normalize
        verdictSpan.classList.add(`verdict-${verdictUpper}`); // e.g., verdict-REAL, verdict-FAKE, verdict-UNKNOWN
        // Note: You'll need corresponding CSS rules for .verdict-REAL, .verdict-FAKE, etc.

        resultsArea.style.display = 'block'; // Show the results grid/block

        // Scroll to results smoothly after a short delay to allow rendering
        setTimeout(() => {
            resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' }); // scroll to top of results
        }, 150);
    }

    function hideResults() {
        resultsArea.style.display = 'none';
        // Reset results fields potentially? Or just hide the container. Hiding is usually fine.
        // Resetting confidence meter is good practice:
        confidenceMeter.style.width = '0%';
        confidenceMeter.className = 'confidence-meter-bar'; // Reset class
        confidenceValue.textContent = '0%';
    }

    async function handleAnalysisRequest() {
        const file = videoInput.files[0];

        // --- Validation ---
        if (!file) {
            showStatus("Please select or drop a video file first.", true);
            return;
        }

        if (!ALLOWED_FILE_TYPES.includes(file.type)) {
            // Try to provide the actual type found for better debugging
             const fileType = file.type || 'unknown';
             const allowedTypesStr = ALLOWED_FILE_TYPES.map(t => t.split('/')[1]).join(', ').toUpperCase(); // "MP4, QUICKTIME, X-MSVIDEO, AVI"
            showStatus(`Unsupported file type (${fileType}). Please upload ${allowedTypesStr}.`, true);
            return;
        }

        if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
            showStatus(`File size (${formatFileSize(file.size)}) exceeds ${MAX_FILE_SIZE_MB}MB limit.`, true);
            return;
        }

        // --- UI Preparation ---
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        hideResults();
        showStatus("Uploading and analyzing video. This may take a moment...", false); // Show as 'info'

        // --- Prepare FormData ---
        const formData = new FormData();
        formData.append("file", file); // Key "file" must match backend expectation (e.g., request.files['file'])

        // --- API Call ---
        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                body: formData,
                // Add headers if needed, e.g., for authentication
                // headers: { 'Authorization': 'Bearer YOUR_TOKEN' }
            });

            // Try to parse JSON regardless of status code, as errors might be in JSON body
            let responseData = null;
            try {
                 responseData = await response.json();
            } catch (jsonError) {
                 // Handle cases where response is not JSON (e.g., server HTML error page)
                 console.error("Failed to parse JSON response:", jsonError);
                 // If response is not ok AND not json, throw a generic error
                 if (!response.ok) {
                     throw new Error(`Server returned status ${response.status} ${response.statusText}, but response was not valid JSON.`);
                 }
                 // If response IS ok but not json, that's unexpected
                 responseData = { success: false, error: "Received non-JSON response from server." };
            }


            if (!response.ok) {
                // Use error message from JSON body if available, otherwise use status text
                const errorMsg = responseData?.error || responseData?.detail || responseData?.message || `Server error: ${response.status} ${response.statusText}`;
                throw new Error(errorMsg);
            }

            // Check for success flag from backend if it exists
            if (responseData && responseData.success === false) {
                 const errorMsg = responseData.error || responseData.message || "Analysis reported failure.";
                 throw new Error(errorMsg);
            }

            // If we reach here, response is OK and likely contains results
             showResults(responseData); // Pass the full parsed JSON


        } catch (error) {
            console.error("Analysis Error:", error);
            let userMessage = `Analysis failed: ${error.message}`;
            // Check for common network errors
            if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
                userMessage = "Could not connect to the analysis server. Please check the server status and your network connection.";
            } else if (error.message.includes('Server error: 5')) {
                 userMessage = "A server error occurred during analysis. Please try again later.";
            }
            showStatus(userMessage, true); // Show the error message in the status area
            hideResults(); // Ensure results are hidden on error

        } finally {
            // --- Reset UI ---
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = '<i class="fas fa-search"></i> Analyze Video';
        }
    }

    // --- Initial Setup ---
    resetFileInfoDisplay(); // Set initial placeholder text
    hideStatus();
    hideResults();

}); // End DOMContentLoaded
