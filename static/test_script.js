document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
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
  
    // Configuration
    const API_ENDPOINT = "/analyze";
    const MAX_FILE_SIZE_MB = 50;
    const ALLOWED_FILE_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
  
    // Event Listeners
    analyzeButton.addEventListener('click', handleAnalysisRequest);
    videoInput.addEventListener('change', handleFileSelect);
    mobileMenuButton.addEventListener('click', toggleMobileMenu);
  
    // Close mobile menu when clicking outside
    document.addEventListener('click', function(event) {
      if (!event.target.closest('.navbar-mobile-menu') && !event.target.closest('.mobile-menu')) {
        mobileMenu.style.display = 'none';
      }
    });
  
    // Functions
    function toggleMobileMenu() {
      if (mobileMenu.style.display === 'flex') {
        mobileMenu.style.display = 'none';
      } else {
        mobileMenu.style.display = 'flex';
      }
    }
  
    function handleFileSelect() {
      const fileLabel = document.querySelector('.file-label-text');
      const fileInfo = document.querySelector('.file-info');
      
      if (videoInput.files.length > 0) {
        const file = videoInput.files[0];
        fileLabel.textContent = file.name;
        fileInfo.textContent = `${formatFileSize(file.size)} • ${file.type}`;
      } else {
        fileLabel.textContent = 'Choose Video File';
        fileInfo.textContent = 'MP4, MOV, AVI up to 50MB';
      }
      
      hideResults();
      hideStatus();
    }
  
    function formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
  
    function showStatus(message, isError = false) {
      statusMessage.textContent = message;
      statusArea.className = 'status-area ' + (isError ? 'error' : '');
      statusArea.style.display = 'block';
    }
  
    function hideStatus() {
      statusArea.style.display = 'none';
    }
  
    function showResults(responseData) { // responseData is the full {success: true, results: {...}} object
    const data = responseData.results; // <-- Access the nested results object

    if (!data) {
        showStatus("Received success response, but results data is missing.", true);
        console.error("Missing 'results' key in response:", responseData);
        return;
    }

    // Check for specific error messages returned within results
    if (data.error_message) {
         showStatus(`Analysis Info: ${data.error_message}`); // Display info/warning but don't show full results
         resultsArea.style.display = 'none'; // Hide the results grid
         return;
    }


    // Update basic info using keys from main.py
    fileNameSpan.textContent = data.file_name || 'N/A';
    verdictSpan.textContent = data.verdict || 'Error'; // Backend provides 'REAL' or 'FAKE'
    totalFramesSpan.textContent = data.frames_analyzed ?? 'N/A'; // Use frames_analyzed
    realCountSpan.textContent = data.real_frames ?? 'N/A';
    deepfakeCountSpan.textContent = data.fake_frames ?? 'N/A'; // Use fake_frames

    // Use the confidence provided by the backend
    const confidence = data.confidence !== undefined ? Math.round(data.confidence) : 0; // Use backend's verdict confidence
    confidenceMeter.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}%`;

    // Update meter color based on confidence (adjust thresholds if needed)
    if (confidence >= 70) {
        confidenceMeter.style.backgroundColor = '#2b8a3e'; // Green
    } else if (confidence >= 40) {
        confidenceMeter.style.backgroundColor = '#e67700'; // Orange
    } else {
        confidenceMeter.style.backgroundColor = '#c92a2a'; // Red
    }

    // Apply verdict styling based on backend verdict
    verdictSpan.className = 'verdict-badge'; // Reset classes
    if (data.verdict === 'REAL') {
         verdictSpan.classList.add(`verdict-REAL`);
    } else if (data.verdict === 'FAKE') {
         verdictSpan.classList.add(`verdict-FAKE`);
    } // Add handling for 'UNKNOWN' if necessary

    resultsArea.style.display = 'block';

    // Scroll to results
    setTimeout(() => {
        resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}
  
    function calculateConfidence(data) {
      if (data.total_frames_processed && data.real_frames !== undefined && data.deepfake_frames !== undefined) {
        const total = data.total_frames_processed;
        const real = data.real_frames;
        const fake = data.deepfake_frames;
        
        const ratio = Math.max(real, fake) / total;
        return Math.round(ratio * 100);
      }
      return 0;
    }
  
    function hideResults() {
      resultsArea.style.display = 'none';
      confidenceMeter.style.width = '0%';
      confidenceValue.textContent = '0%';
    }
  
    async function handleAnalysisRequest() {
      const file = videoInput.files[0];
  
      // Validation
      if (!file) {
        showStatus("Please select a video file first.", true);
        return;
      }
  
      if (!ALLOWED_FILE_TYPES.includes(file.type)) {
        showStatus("Unsupported file type. Please upload an MP4, MOV, or AVI file.", true);
        return;
      }
  
      if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        showStatus(`File size exceeds ${MAX_FILE_SIZE_MB}MB limit. Please choose a smaller file.`, true);
        return;
      }
  
      // UI Preparation
      analyzeButton.disabled = true;
      analyzeButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
      hideResults();
      showStatus("Uploading and analyzing video. This may take a few moments...");
  
      // Prepare FormData
      const formData = new FormData();
      formData.append("file", file);
  
      try {
        const response = await fetch(API_ENDPOINT, {
          method: 'POST',
          body: formData,
        });
  
        const responseData = await response.json();
  
        if (!response.ok) {
          const errorMsg = responseData.detail || 
                          responseData.message || 
                          `Server error: ${response.status} ${response.statusText}`;
          throw new Error(errorMsg);
        }
  
        hideStatus();
        showResults(responseData);
  
      } catch (error) {
        console.error("Analysis Error:", error);
        showStatus(`Analysis failed: ${error.message}`, true);
        
        if (error.message.toLowerCase().includes('failed to fetch')) {
          showStatus("Could not connect to the analysis server. Please check your internet connection and try again.", true);
        }
      } finally {
        analyzeButton.disabled = false;
        analyzeButton.innerHTML = '<i class="fas fa-search"></i> Analyze Video';
      }
    }
  });
  // test_script.js
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobileMenuButton');
    const mobileMenu = document.getElementById('mobileMenu');
    
    mobileMenuButton.addEventListener('click', function() {
        mobileMenu.style.display = mobileMenu.style.display === 'block' ? 'none' : 'block';
    });
    
    // Video analysis functionality
    const videoInput = document.getElementById('videoInput');
    const analyzeButton = document.getElementById('analyzeButton');
    const statusArea = document.getElementById('statusArea');
    const statusMessage = document.getElementById('statusMessage');
    const spinner = document.getElementById('spinner');
    const resultsArea = document.getElementById('resultsArea');
    
    analyzeButton.addEventListener('click', async function() {
        const file = videoInput.files[0];
        
        if (!file) {
            alert('Please select a video file first');
            return;
        }
        
        // Check file size (50MB limit)
        if (file.size > 50 * 1024 * 1024) {
            alert('File size exceeds 50MB limit');
            return;
        }
        
        // Show loading state
        analyzeButton.disabled = true;
        statusArea.style.display = 'block';
        statusMessage.textContent = 'Analyzing video...';
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Analysis failed');
            }
            
            // Display results
            displayResults(data.results);
            
        } catch (error) {
            console.error('Error:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            setTimeout(() => {
                statusArea.style.display = 'none';
            }, 3000);
        } finally {
            analyzeButton.disabled = false;
        }
    });
    
    function displayResults(results) {
        // Update UI with results
        document.getElementById('fileName').textContent = results.file_name;
        document.getElementById('totalFrames').textContent = results.total_frames;
        document.getElementById('realCount').textContent = results.real_frames;
        document.getElementById('deepfakeCount').textContent = results.fake_frames;
        
        // Update verdict
        const verdictElement = document.getElementById('verdict');
        verdictElement.textContent = results.verdict;
        verdictElement.className = 'verdict-badge ' + (results.verdict === 'REAL' ? 'verdict-real' : 'verdict-fake');
        
        // Update confidence meter
        const confidenceMeter = document.getElementById('confidenceMeter');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidence = Math.round(results.confidence);
        
        confidenceMeter.style.width = `${confidence}%`;
        confidenceMeter.style.backgroundColor = results.verdict === 'REAL' ? '#4CAF50' : '#F44336';
        confidenceValue.textContent = `${confidence}%`;
        
        // Hide spinner and show results
        statusArea.style.display = 'none';
        resultsArea.style.display = 'block';
        
        // Scroll to results
        resultsArea.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Drag and drop functionality
    const fileUploadContainer = document.querySelector('.file-upload-container');
    const fileLabel = document.querySelector('.file-label');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileUploadContainer.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        fileUploadContainer.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        fileUploadContainer.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        fileUploadContainer.classList.add('highlight');
    }
    
    function unhighlight() {
        fileUploadContainer.classList.remove('highlight');
    }
    
    fileUploadContainer.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            videoInput.files = files;
            
            // Update file info display
            const fileInfo = fileLabel.querySelector('.file-info');
            fileInfo.textContent = `${files[0].name} (${formatFileSize(files[0].size)})`;
        }
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
});
