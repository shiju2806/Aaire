/**
 * AAIRE Browser Extension - Popup Script
 */

// DOM elements
let currentTab = 'detect';
let currentPageInfo = null;
let storedDocuments = [];
let selectedDocument = null;

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
  console.log('AAIRE Popup initializing...');
  
  // Setup tab navigation
  setupTabs();
  
  // Setup event listeners
  setupEventListeners();
  
  // Load initial data
  await loadInitialData();
  
  console.log('AAIRE Popup initialized');
});

/**
 * Setup tab navigation
 */
function setupTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const tabName = button.getAttribute('data-tab');
      
      // Update active tab button
      tabButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
      
      // Update active tab content
      tabContents.forEach(content => content.classList.remove('active'));
      document.getElementById(`${tabName}-tab`).classList.add('active');
      
      currentTab = tabName;
      
      // Load tab-specific data
      loadTabData(tabName);
    });
  });
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Detect tab
  document.getElementById('scan-page').addEventListener('click', scanCurrentPage);
  document.getElementById('send-all').addEventListener('click', sendAllDocuments);
  
  // Documents tab
  document.getElementById('refresh-documents').addEventListener('click', refreshDocuments);
  document.getElementById('open-aaire').addEventListener('click', openAAIRE);
  document.getElementById('document-search').addEventListener('input', filterDocuments);
  
  // Query tab
  document.getElementById('query-document').addEventListener('change', selectDocument);
  document.getElementById('submit-query').addEventListener('click', submitQuery);
  document.getElementById('query-text').addEventListener('input', updateQueryButton);
  
  // Footer
  document.getElementById('open-settings').addEventListener('click', openSettings);
}

/**
 * Load initial data
 */
async function loadInitialData() {
  try {
    showLoading(true);
    
    // Get current page info
    await getCurrentPageInfo();
    
    // Load stored documents
    await loadStoredDocuments();
    
    // Update UI
    updateDetectTab();
    updateDocumentsTab();
    updateQueryTab();
    
  } catch (error) {
    console.error('Failed to load initial data:', error);
    showError('Failed to load extension data');
  } finally {
    showLoading(false);
  }
}

/**
 * Get current page information
 */
async function getCurrentPageInfo() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab) {
      throw new Error('No active tab found');
    }
    
    // Send message to content script
    const response = await chrome.tabs.sendMessage(tab.id, {
      type: 'GET_PAGE_INFO'
    });
    
    if (response.success) {
      currentPageInfo = response.data;
      console.log('Current page info:', currentPageInfo);
    } else {
      throw new Error('Failed to get page info');
    }
    
  } catch (error) {
    console.log('Content script not available, page not supported');
    currentPageInfo = {
      url: 'about:blank',
      title: 'Unsupported Page',
      documents: []
    };
  }
}

/**
 * Load stored documents
 */
async function loadStoredDocuments() {
  try {
    const response = await chrome.runtime.sendMessage({
      type: 'GET_STORED_DOCUMENTS'
    });
    
    if (response.success) {
      storedDocuments = response.data.local || [];
      console.log('Loaded stored documents:', storedDocuments);
    } else {
      throw new Error(response.error);
    }
    
  } catch (error) {
    console.error('Failed to load stored documents:', error);
    storedDocuments = [];
  }
}

/**
 * Load tab-specific data
 */
async function loadTabData(tabName) {
  switch (tabName) {
    case 'detect':
      await scanCurrentPage();
      break;
    case 'documents':
      await refreshDocuments();
      break;
    case 'query':
      updateQueryTab();
      break;
  }
}

/**
 * Update detect tab
 */
function updateDetectTab() {
  const statusElement = document.getElementById('page-status');
  const documentsElement = document.getElementById('detected-documents');
  const sendButton = document.getElementById('send-all');
  
  if (!currentPageInfo) {
    statusElement.className = 'status-indicator error';
    statusElement.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Unable to access page</span>
    `;
    documentsElement.innerHTML = '';
    sendButton.disabled = true;
    return;
  }
  
  const isSOAPage = currentPageInfo.url.includes('soa.org');
  const documentCount = currentPageInfo.documents ? currentPageInfo.documents.length : 0;
  
  if (!isSOAPage) {
    statusElement.className = 'status-indicator warning';
    statusElement.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Not on SOA website</span>
    `;
    documentsElement.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🌐</div>
        <div class="empty-state-title">Navigate to SOA</div>
        <div class="empty-state-description">
          Visit publications.soa.org to detect and capture documents
        </div>
      </div>
    `;
    sendButton.disabled = true;
    return;
  }
  
  if (documentCount === 0) {
    statusElement.className = 'status-indicator warning';
    statusElement.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">No documents detected</span>
    `;
    documentsElement.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <div class="empty-state-title">No Documents Found</div>
        <div class="empty-state-description">
          Try scanning the page or navigate to a publication page
        </div>
      </div>
    `;
    sendButton.disabled = true;
  } else {
    statusElement.className = 'status-indicator success';
    statusElement.innerHTML = `
      <span class="status-dot"></span>
      <span class="status-text">Found ${documentCount} document(s)</span>
    `;
    
    documentsElement.innerHTML = currentPageInfo.documents.map((doc, index) => `
      <div class="document-item">
        <input type="checkbox" class="document-checkbox" id="doc-${index}" checked>
        <div class="document-info">
          <div class="document-title">${doc.title}</div>
          <div class="document-meta">${doc.type} • ${getFilenameFromUrl(doc.url)}</div>
        </div>
      </div>
    `).join('');
    
    sendButton.disabled = false;
  }
}

/**
 * Update documents tab
 */
function updateDocumentsTab() {
  const documentsElement = document.getElementById('stored-documents');
  
  if (storedDocuments.length === 0) {
    documentsElement.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📚</div>
        <div class="empty-state-title">No Documents</div>
        <div class="empty-state-description">
          Upload documents from SOA pages to get started
        </div>
      </div>
    `;
    return;
  }
  
  documentsElement.innerHTML = storedDocuments.map(doc => `
    <div class="stored-document" data-job-id="${doc.job_id}">
      <div class="document-title">${doc.filename || 'Untitled Document'}</div>
      <div class="document-meta">
        ${doc.source_url ? `From: ${new URL(doc.source_url).hostname}` : ''}
        ${doc.upload_time ? ` • ${formatDate(doc.upload_time)}` : ''}
      </div>
      <div class="document-status ${getStatusClass(doc.status)}">${doc.status || 'unknown'}</div>
    </div>
  `).join('');
  
  // Add click listeners
  document.querySelectorAll('.stored-document').forEach(element => {
    element.addEventListener('click', () => {
      // Remove previous selection
      document.querySelectorAll('.stored-document').forEach(el => el.classList.remove('selected'));
      
      // Select this document
      element.classList.add('selected');
      const jobId = element.getAttribute('data-job-id');
      selectedDocument = storedDocuments.find(doc => doc.job_id === jobId);
      
      // Update query tab
      updateQueryTab();
    });
  });
}

/**
 * Update query tab
 */
function updateQueryTab() {
  const selectElement = document.getElementById('query-document');
  const submitButton = document.getElementById('submit-query');
  
  // Populate document select
  selectElement.innerHTML = '<option value="">Select a document...</option>' +
    storedDocuments
      .filter(doc => doc.status === 'completed')
      .map(doc => `
        <option value="${doc.job_id}">${doc.filename || 'Untitled Document'}</option>
      `).join('');
  
  // Pre-select if we have a selected document
  if (selectedDocument && selectedDocument.status === 'completed') {
    selectElement.value = selectedDocument.job_id;
  }
  
  updateQueryButton();
}

/**
 * Scan current page for documents
 */
async function scanCurrentPage() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab) {
      throw new Error('No active tab found');
    }
    
    // Send message to content script
    const response = await chrome.tabs.sendMessage(tab.id, {
      type: 'DETECT_DOCUMENTS'
    });
    
    if (response.success) {
      currentPageInfo.documents = response.documents;
      updateDetectTab();
    } else {
      throw new Error('Failed to scan page');
    }
    
  } catch (error) {
    console.error('Page scan failed:', error);
    showError('Failed to scan page. Make sure you\'re on a SOA website.');
  }
}

/**
 * Send all selected documents to AAIRE
 */
async function sendAllDocuments() {
  try {
    showLoading(true);
    
    const checkboxes = document.querySelectorAll('.document-checkbox:checked');
    const selectedDocs = Array.from(checkboxes).map(checkbox => {
      const index = parseInt(checkbox.id.split('-')[1]);
      return currentPageInfo.documents[index];
    });
    
    if (selectedDocs.length === 0) {
      throw new Error('No documents selected');
    }
    
    for (const doc of selectedDocs) {
      await sendDocumentToAAIRE(doc);
    }
    
    // Refresh stored documents
    await loadStoredDocuments();
    updateDocumentsTab();
    
    showSuccess(`Successfully sent ${selectedDocs.length} document(s) to AAIRE`);
    
  } catch (error) {
    console.error('Failed to send documents:', error);
    showError(`Failed to send documents: ${error.message}`);
  } finally {
    showLoading(false);
  }
}

/**
 * Send individual document to AAIRE
 */
async function sendDocumentToAAIRE(document) {
  // Download the document
  const response = await fetch(document.url);
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.status}`);
  }
  
  const blob = await response.blob();
  const filename = getFilenameFromUrl(document.url);
  const file = new File([blob], filename, { type: blob.type });
  
  // Send to background script
  const result = await new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({
      type: 'UPLOAD_DOCUMENT',
      data: {
        file: file,
        filename: filename,
        sourceUrl: currentPageInfo.url,
        pageTitle: currentPageInfo.title
      }
    }, response => {
      if (response.success) {
        resolve(response.data);
      } else {
        reject(new Error(response.error));
      }
    });
  });
  
  console.log('Document sent successfully:', result);
}

/**
 * Refresh stored documents
 */
async function refreshDocuments() {
  try {
    showLoading(true);
    await loadStoredDocuments();
    updateDocumentsTab();
  } catch (error) {
    console.error('Failed to refresh documents:', error);
    showError('Failed to refresh documents');
  } finally {
    showLoading(false);
  }
}

/**
 * Filter documents based on search
 */
function filterDocuments() {
  const searchTerm = document.getElementById('document-search').value.toLowerCase();
  const documents = document.querySelectorAll('.stored-document');
  
  documents.forEach(doc => {
    const title = doc.querySelector('.document-title').textContent.toLowerCase();
    const meta = doc.querySelector('.document-meta').textContent.toLowerCase();
    
    if (title.includes(searchTerm) || meta.includes(searchTerm)) {
      doc.style.display = 'block';
    } else {
      doc.style.display = 'none';
    }
  });
}

/**
 * Select document for querying
 */
function selectDocument() {
  const selectElement = document.getElementById('query-document');
  const jobId = selectElement.value;
  
  selectedDocument = storedDocuments.find(doc => doc.job_id === jobId);
  updateQueryButton();
}

/**
 * Update query button state
 */
function updateQueryButton() {
  const submitButton = document.getElementById('submit-query');
  const queryText = document.getElementById('query-text').value.trim();
  const documentSelected = document.getElementById('query-document').value;
  
  submitButton.disabled = !queryText || !documentSelected;
}

/**
 * Submit query to AAIRE
 */
async function submitQuery() {
  try {
    const jobId = document.getElementById('query-document').value;
    const query = document.getElementById('query-text').value.trim();
    
    if (!jobId || !query) {
      throw new Error('Please select a document and enter a query');
    }
    
    showLoading(true);
    
    const response = await chrome.runtime.sendMessage({
      type: 'QUERY_DOCUMENT',
      jobId: jobId,
      query: query
    });
    
    if (!response.success) {
      throw new Error(response.error);
    }
    
    // Display response
    const responseElement = document.getElementById('query-response');
    responseElement.innerHTML = `
      <div class="response-text">${response.data.response}</div>
    `;
    
  } catch (error) {
    console.error('Query failed:', error);
    showError(`Query failed: ${error.message}`);
  } finally {
    showLoading(false);
  }
}

/**
 * Open AAIRE web app
 */
function openAAIRE() {
  chrome.tabs.create({ url: 'https://aaire.xyz' });
}

/**
 * Open extension settings
 */
function openSettings() {
  chrome.runtime.openOptionsPage();
}

/**
 * Utility functions
 */
function getFilenameFromUrl(url) {
  try {
    return url.split('/').pop() || 'document.pdf';
  } catch {
    return 'document.pdf';
  }
}

function getStatusClass(status) {
  switch (status) {
    case 'completed': return 'ready';
    case 'processing': return 'processing';
    case 'failed': return 'failed';
    default: return 'processing';
  }
}

function formatDate(dateString) {
  try {
    return new Date(dateString).toLocaleDateString();
  } catch {
    return 'Unknown date';
  }
}

function showLoading(show) {
  const overlay = document.getElementById('loading-overlay');
  overlay.style.display = show ? 'flex' : 'none';
}

function showError(message) {
  // Simple error display - could be enhanced with better UI
  console.error(message);
  alert(`Error: ${message}`);
}

function showSuccess(message) {
  // Simple success display - could be enhanced with better UI
  console.log(message);
}

console.log('AAIRE Popup script loaded');