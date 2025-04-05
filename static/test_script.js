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
    const API_ENDPOINT = "http://127.0.0.1:8000/analyze/";
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
  
    function showResults(data) {
      // Update basic info
      fileNameSpan.textContent = data.filename || 'N/A';
      verdictSpan.textContent = data.verdict || 'Error';
      totalFramesSpan.textContent = data.total_frames_processed ?? 'N/A';
      realCountSpan.textContent = data.real_frames ?? 'N/A';
      deepfakeCountSpan.textContent = data.deepfake_frames ?? 'N/A';
  
      // Calculate confidence percentage
      const confidence = calculateConfidence(data);
      confidenceMeter.style.width = `${confidence}%`;
      confidenceValue.textContent = `${confidence}%`;
      
      // Update meter color based on confidence
      if (confidence >= 70) {
        confidenceMeter.style.backgroundColor = '#2b8a3e';
      } else if (confidence >= 40) {
        confidenceMeter.style.backgroundColor = '#e67700';
      } else {
        confidenceMeter.style.backgroundColor = '#c92a2a';
      }
  
      // Apply verdict styling
      verdictSpan.className = 'verdict-badge';
      if (data.verdict) {
        const verdictClass = data.verdict.split(' ')[0].replace(/[^a-zA-Z0-9]/g, '');
        verdictSpan.classList.add(`verdict-${verdictClass}`);
      }
  
      resultsArea.style.display = 'block';
      
      // Smooth scroll to results
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
