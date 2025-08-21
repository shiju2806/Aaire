/**
 * AAIRE Browser Extension - Background Service Worker
 * Handles communication between content scripts and AAIRE backend
 */

// Configuration
const AAIRE_API_BASE = 'https://aaire.xyz/api/v1/extension';
const EXTENSION_VERSION = '1.0.0';

// Track upload jobs
let activeJobs = new Map();

// Listen for messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received message:', request.type);
  
  switch (request.type) {
    case 'UPLOAD_DOCUMENT':
      handleDocumentUpload(request.data)
        .then(response => sendResponse({ success: true, data: response }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true; // Keep message channel open for async response
      
    case 'CHECK_JOB_STATUS':
      checkJobStatus(request.jobId)
        .then(response => sendResponse({ success: true, data: response }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
      
    case 'QUERY_DOCUMENT':
      queryDocument(request.jobId, request.query)
        .then(response => sendResponse({ success: true, data: response }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
      
    case 'GET_STORED_DOCUMENTS':
      getStoredDocuments()
        .then(response => sendResponse({ success: true, data: response }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
      
    case 'OPEN_AAIRE':
      // Open AAIRE website in new tab
      chrome.tabs.create({ url: 'https://aaire.xyz' });
      sendResponse({ success: true });
      return true;
      
    default:
      console.warn('Unknown message type:', request.type);
  }
});

/**
 * Upload document to AAIRE backend
 */
async function handleDocumentUpload(uploadData) {
  try {
    console.log('Starting document upload:', uploadData.fileData.name);
    
    // Reconstruct File object from transferred data
    const fileContent = new Uint8Array(uploadData.fileData.content);
    const file = new File([fileContent], uploadData.fileData.name, { 
      type: uploadData.fileData.type 
    });
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('source_url', uploadData.sourceUrl);
    formData.append('page_title', uploadData.pageTitle);
    formData.append('extension_version', EXTENSION_VERSION);
    
    console.log('Uploading to AAIRE:', {
      filename: file.name,
      fileSize: file.size,
      fileType: file.type,
      sourceUrl: uploadData.sourceUrl,
      pageTitle: uploadData.pageTitle
    });
    
    // Upload to AAIRE
    const response = await fetch(`${AAIRE_API_BASE}/upload`, {
      method: 'POST',
      body: formData
    });
    
    console.log('Upload response status:', response.status);
    console.log('Upload response headers:', response.headers);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Upload error response:', errorText);
      throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('Upload successful:', result);
    
    // Store job info
    activeJobs.set(result.job_id, {
      ...result,
      uploadTime: Date.now(),
      sourceUrl: uploadData.sourceUrl,
      pageTitle: uploadData.pageTitle
    });
    
    // Store in browser storage for persistence
    await storeJobInfo(result.job_id, activeJobs.get(result.job_id));
    
    return result;
    
  } catch (error) {
    console.error('Document upload failed:', error);
    throw error;
  }
}

/**
 * Check job processing status
 */
async function checkJobStatus(jobId) {
  try {
    const response = await fetch(`${AAIRE_API_BASE}/status/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Update stored job info
    if (activeJobs.has(jobId)) {
      activeJobs.set(jobId, { ...activeJobs.get(jobId), ...result });
      await storeJobInfo(jobId, activeJobs.get(jobId));
    }
    
    return result;
    
  } catch (error) {
    console.error('Status check failed:', error);
    throw error;
  }
}

/**
 * Query processed document
 */
async function queryDocument(jobId, query) {
  try {
    const formData = new FormData();
    formData.append('job_id', jobId);
    formData.append('query', query);
    
    const response = await fetch(`${AAIRE_API_BASE}/query`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Query failed: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('Query successful for job:', jobId);
    
    return result;
    
  } catch (error) {
    console.error('Document query failed:', error);
    throw error;
  }
}

/**
 * Get list of stored documents
 */
async function getStoredDocuments() {
  try {
    // Get from local storage
    const result = await chrome.storage.local.get('aaire_documents');
    const documents = result.aaire_documents || {};
    
    // Also fetch fresh list from server
    const response = await fetch(`${AAIRE_API_BASE}/documents`);
    if (response.ok) {
      const serverDocs = await response.json();
      return {
        local: Object.values(documents),
        server: serverDocs.documents || []
      };
    }
    
    return {
      local: Object.values(documents),
      server: []
    };
    
  } catch (error) {
    console.error('Failed to get stored documents:', error);
    throw error;
  }
}

/**
 * Store job info in browser storage
 */
async function storeJobInfo(jobId, jobInfo) {
  try {
    const result = await chrome.storage.local.get('aaire_documents');
    const documents = result.aaire_documents || {};
    
    documents[jobId] = jobInfo;
    
    await chrome.storage.local.set({ aaire_documents: documents });
    console.log('Job info stored:', jobId);
    
  } catch (error) {
    console.error('Failed to store job info:', error);
  }
}

/**
 * Handle extension installation
 */
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('AAIRE Extension installed');
    
    // Initialize storage
    chrome.storage.local.set({
      aaire_documents: {},
      extension_settings: {
        auto_detect: true,
        notifications: true
      }
    });
  }
});

/**
 * Health check for AAIRE backend
 */
async function checkBackendHealth() {
  try {
    const response = await fetch(`${AAIRE_API_BASE}/health`);
    return response.ok;
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
}

// Periodic health check
setInterval(checkBackendHealth, 300000); // Every 5 minutes

console.log('AAIRE Extension background script loaded');