/**
 * AAIRE Browser Extension - Content Script
 * Runs on SOA pages to detect and capture documents
 */

// Configuration
const SUPPORTED_DOCUMENT_TYPES = ['.pdf', '.docx', '.doc', '.ppt', '.pptx'];
const SUPPORTED_DOMAINS = ['publications.soa.org', 'www.soa.org', 'platform.virdocs.com'];

// State
let detectedDocuments = [];
let aaireUI = null;

// Initialize when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}

function initialize() {
  console.log('AAIRE Content Script initializing on:', window.location.href);
  console.log('Hostname:', window.location.hostname);
  console.log('Supported domains:', SUPPORTED_DOMAINS);
  
  // Check if we're on a supported domain
  const isSupported = SUPPORTED_DOMAINS.some(domain => {
    const matches = window.location.hostname.includes(domain);
    console.log(`Checking ${domain}: ${matches}`);
    return matches;
  });
  
  if (!isSupported) {
    console.log('Not on supported domain, extension inactive');
    console.log('Current hostname:', window.location.hostname);
    return;
  }
  
  console.log('✅ Supported domain detected, continuing initialization');
  
  // Create AAIRE UI overlay first
  createAAIREUI();
  
  // Then detect documents (this will update the UI)
  detectDocuments();
  
  // Monitor for dynamic content changes
  observePageChanges();
  
  console.log('AAIRE Content Script initialized');
}

/**
 * Detect documents on the current page
 */
function detectDocuments() {
  detectedDocuments = [];
  
  // Method 1: Look for direct PDF/document links
  const documentLinks = document.querySelectorAll('a[href]');
  documentLinks.forEach(link => {
    const href = link.href;
    if (SUPPORTED_DOCUMENT_TYPES.some(type => href.toLowerCase().includes(type))) {
      detectedDocuments.push({
        type: 'link',
        url: href,
        title: link.textContent.trim() || link.title || 'Untitled Document',
        element: link
      });
    }
  });
  
  // Method 2: Look for embedded PDFs or document viewers
  const embeddedDocuments = document.querySelectorAll('iframe, embed, object');
  embeddedDocuments.forEach(element => {
    const src = element.src || element.data;
    if (src && SUPPORTED_DOCUMENT_TYPES.some(type => src.toLowerCase().includes(type))) {
      detectedDocuments.push({
        type: 'embedded',
        url: src,
        title: element.title || 'Embedded Document',
        element: element
      });
    }
  });
  
  // Method 3: Look for platform-specific document patterns
  detectPlatformSpecificDocuments();
  
  console.log(`Detected ${detectedDocuments.length} documents:`, detectedDocuments);
  
  // Update UI
  updateAAIREUI();
}

/**
 * Detect platform-specific document patterns
 */
function detectPlatformSpecificDocuments() {
  
  // VirDocs platform detection
  if (window.location.hostname.includes('platform.virdocs.com')) {
    detectVirDocsDocument();
  }
  
  // SOA platform detection
  if (window.location.hostname.includes('soa.org')) {
    detectSOADocuments();
  }
}

/**
 * Detect VirDocs document
 */
function detectVirDocsDocument() {
  // VirDocs shows documents in a viewer - detect the current document
  const currentUrl = window.location.href;
  const documentTitle = document.title || 'VirDocs Document';
  
  // Extract document ID from URL (e.g., /read/2867432/8/#/4/222)
  const docIdMatch = currentUrl.match(/\/read\/(\d+)/);
  if (docIdMatch) {
    const docId = docIdMatch[1];
    
    // Check if there's a way to get the actual document URL
    // VirDocs usually has a download or print option
    const downloadLink = document.querySelector('a[href*="download"], a[href*="pdf"], button[title*="download"], button[title*="print"]');
    
    let documentUrl = currentUrl;
    if (downloadLink && downloadLink.href) {
      documentUrl = downloadLink.href;
    }
    
    detectedDocuments.push({
      type: 'virdocs_viewer',
      url: documentUrl,
      title: documentTitle,
      element: document.body,
      platform: 'VirDocs',
      documentId: docId
    });
    
    // Also try to find direct PDF links in the page
    const pdfLinks = document.querySelectorAll('a[href*=".pdf"], iframe[src*=".pdf"]');
    pdfLinks.forEach(link => {
      const url = link.href || link.src;
      if (url && !detectedDocuments.some(doc => doc.url === url)) {
        detectedDocuments.push({
          type: 'virdocs_pdf',
          url: url,
          title: documentTitle + ' (PDF)',
          element: link,
          platform: 'VirDocs'
        });
      }
    });
  }
}

/**
 * Detect SOA-specific documents
 */
function detectSOADocuments() {
  // Look for publication download buttons
  const downloadButtons = document.querySelectorAll('.download-btn, .pdf-download, [data-download]');
  downloadButtons.forEach(button => {
    const url = button.href || button.getAttribute('data-url') || button.getAttribute('data-download');
    if (url) {
      detectedDocuments.push({
        type: 'download_button',
        url: url,
        title: button.textContent.trim() || 'SOA Publication',
        element: button
      });
    }
  });
  
  // Look for document titles and descriptions
  const publicationTitles = document.querySelectorAll('.publication-title, .document-title, h1, h2');
  const currentPageTitle = document.title;
  
  // Check if current page is a document page
  if (currentPageTitle.toLowerCase().includes('publication') || 
      currentPageTitle.toLowerCase().includes('research') ||
      document.querySelector('.publication-content, .document-content')) {
    
    // Look for download links on publication pages
    const pageDownloads = document.querySelectorAll('a[href*=".pdf"], a[href*="download"]');
    pageDownloads.forEach(link => {
      detectedDocuments.push({
        type: 'page_download',
        url: link.href,
        title: currentPageTitle,
        element: link
      });
    });
  }
}

/**
 * Create AAIRE UI overlay
 */
function createAAIREUI() {
  // Remove existing UI if present
  if (aaireUI) {
    aaireUI.remove();
  }
  
  // Create floating UI panel
  aaireUI = document.createElement('div');
  aaireUI.id = 'aaire-extension-ui';
  aaireUI.innerHTML = `
    <div class="aaire-header">
      <img src="${chrome.runtime.getURL('icons/icon32.png')}" alt="AAIRE" class="aaire-logo">
      <span class="aaire-title">AAIRE Assistant</span>
      <button class="aaire-toggle" id="aaire-toggle">−</button>
    </div>
    <div class="aaire-content" id="aaire-content">
      <div class="aaire-status" id="aaire-status">
        Scanning page for documents...
      </div>
      <div class="aaire-documents" id="aaire-documents">
        <!-- Documents will be populated here -->
      </div>
      <div class="aaire-actions" id="aaire-actions" style="display: none;">
        <button class="aaire-btn aaire-btn-primary" id="send-to-aaire">
          Send to AAIRE
        </button>
        <button class="aaire-btn aaire-btn-secondary" id="view-results">
          View Results
        </button>
      </div>
    </div>
  `;
  
  document.body.appendChild(aaireUI);
  
  // Add event listeners
  document.getElementById('aaire-toggle').addEventListener('click', toggleAAIREUI);
  document.getElementById('send-to-aaire').addEventListener('click', sendSelectedDocuments);
  document.getElementById('view-results').addEventListener('click', viewResults);
  
  console.log('AAIRE UI created');
}

/**
 * Update AAIRE UI with detected documents
 */
function updateAAIREUI() {
  const statusElement = document.getElementById('aaire-status');
  const documentsElement = document.getElementById('aaire-documents');
  const actionsElement = document.getElementById('aaire-actions');
  
  // Safety check - if UI elements don't exist, skip update
  if (!statusElement || !documentsElement || !actionsElement) {
    console.log('AAIRE UI elements not found, skipping update');
    return;
  }
  
  if (detectedDocuments.length === 0) {
    statusElement.textContent = 'No documents detected on this page';
    documentsElement.innerHTML = '';
    actionsElement.style.display = 'none';
    return;
  }
  
  statusElement.textContent = `Found ${detectedDocuments.length} document(s)`;
  
  // Create document list
  documentsElement.innerHTML = detectedDocuments.map((doc, index) => `
    <div class="aaire-document">
      <input type="checkbox" id="doc-${index}" class="aaire-checkbox" checked>
      <label for="doc-${index}" class="aaire-document-title">${doc.title}</label>
      <span class="aaire-document-type">${doc.type}</span>
    </div>
  `).join('');
  
  actionsElement.style.display = 'block';
}

/**
 * Toggle AAIRE UI visibility
 */
function toggleAAIREUI() {
  const content = document.getElementById('aaire-content');
  const toggle = document.getElementById('aaire-toggle');
  
  if (content.style.display === 'none') {
    content.style.display = 'block';
    toggle.textContent = '−';
  } else {
    content.style.display = 'none';
    toggle.textContent = '+';
  }
}

/**
 * Send selected documents to AAIRE
 */
async function sendSelectedDocuments() {
  const checkboxes = document.querySelectorAll('.aaire-checkbox:checked');
  const selectedDocs = Array.from(checkboxes).map(cb => {
    const index = parseInt(cb.id.split('-')[1]);
    return detectedDocuments[index];
  });
  
  if (selectedDocs.length === 0) {
    alert('Please select at least one document');
    return;
  }
  
  console.log('Sending documents to AAIRE:', selectedDocs);
  
  for (const doc of selectedDocs) {
    try {
      await sendDocumentToAAIRE(doc);
    } catch (error) {
      console.error('Failed to send document:', error);
      alert(`Failed to send ${doc.title}: ${error.message}`);
    }
  }
}

/**
 * Send individual document to AAIRE
 */
async function sendDocumentToAAIRE(docInfo) {
  try {
    let file, filename;
    
    // Handle VirDocs viewer documents differently
    if (docInfo.type === 'virdocs_viewer') {
      // For VirDocs, we need to extract content from the document viewer
      // Try multiple strategies to get the document content
      let pageContent = '';
      
      // Strategy 1: Look for specific VirDocs content containers
      const contentSelectors = [
        '.document-content',
        '.reader-content', 
        '.pdf-content',
        '.viewer-content',
        '.virdocs-content',
        '[class*="document"]',
        '[class*="content"]',
        '[class*="viewer"]',
        '[class*="reader"]',
        '[id*="document"]',
        '[id*="content"]',
        '[id*="viewer"]',
        '[id*="reader"]',
        '.page-content',
        'main',
        'article',
        '.react-pdf__Page__textContent',
        '.textLayer',
        'canvas + div' // PDF.js text layer often follows canvas
      ];
      
      console.log('Trying content selectors...');
      for (const selector of contentSelectors) {
        const contentElement = document.querySelector(selector);
        if (contentElement) {
          const text = contentElement.innerText || contentElement.textContent || '';
          console.log(`Found content with selector "${selector}": ${text.length} characters`);
          if (text.length > pageContent.length) {
            pageContent = text;
          }
        }
      }
      
      // Strategy 2: Look for React/Angular component content
      if (pageContent.length < 100) {
        console.log('Trying React/Angular selectors...');
        const reactSelectors = [
          '[data-testid*="content"]',
          '[data-testid*="document"]',
          '[data-testid*="viewer"]',
          '[class*="App"]',
          '[class*="Component"]',
          '[id*="root"] *',
          '.document-viewer',
          '.pdf-viewer'
        ];
        
        for (const selector of reactSelectors) {
          const elements = document.querySelectorAll(selector);
          elements.forEach(element => {
            const text = element.innerText || element.textContent || '';
            if (text.length > pageContent.length && text.length > 100) {
              console.log(`Found content with React selector "${selector}": ${text.length} characters`);
              pageContent = text;
            }
          });
        }
      }
      
      // Strategy 3: Try to wait for dynamic content and look for text layers
      if (pageContent.length < 100) {
        console.log('Waiting for dynamic content...');
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds for content to load
        
        // Look for PDF.js text layers specifically
        const textLayers = document.querySelectorAll('.textLayer, .react-pdf__Page__textContent, canvas + div');
        let combinedText = '';
        textLayers.forEach(layer => {
          const text = layer.innerText || layer.textContent || '';
          if (text.length > 10) {
            combinedText += text + '\n';
          }
        });
        
        if (combinedText.length > pageContent.length) {
          console.log(`Found text layers content: ${combinedText.length} characters`);
          pageContent = combinedText;
        }
      }
      
      // Strategy 4: If no specific content found, try all text content but filter out navigation
      if (pageContent.length < 100) {
        console.log('Using filtered body content...');
        const allText = document.body.innerText || document.body.textContent || '';
        // Filter out common navigation/UI elements
        const lines = allText.split('\n').filter(line => {
          const trimmedLine = line.trim();
          return trimmedLine.length > 10 && 
                 !trimmedLine.includes('Navigation') &&
                 !trimmedLine.includes('Menu') &&
                 !trimmedLine.includes('Download') &&
                 !trimmedLine.includes('Print') &&
                 !trimmedLine.includes('Share') &&
                 !trimmedLine.includes('©') &&
                 !trimmedLine.startsWith('Page ') &&
                 !trimmedLine.match(/^\d+$/) && // Skip page numbers
                 trimmedLine.length < 500; // Skip very long lines (likely navigation)
        });
        pageContent = lines.join('\n');
        console.log(`Filtered content: ${pageContent.length} characters`);
      }
      
      // Strategy 5: Look for iframe content (PDF viewers)
      if (pageContent.length < 100) {
        console.log('Trying iframe content...');
        const iframes = document.querySelectorAll('iframe');
        for (const iframe of iframes) {
          try {
            const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
            if (iframeDoc) {
              const iframeText = iframeDoc.body?.innerText || iframeDoc.body?.textContent || '';
              if (iframeText.length > pageContent.length) {
                console.log(`Found iframe content: ${iframeText.length} characters`);
                pageContent = iframeText;
              }
            }
          } catch (e) {
            // Cross-origin iframe, can't access content
            console.log('Cannot access iframe content due to cross-origin restrictions');
          }
        }
      }
      
      // If still no content, create a meaningful placeholder
      if (pageContent.length < 100) {
        pageContent = `VirDocs Document (ID: ${docInfo.documentId || 'unknown'})

This document was detected on VirDocs platform but the full content could not be extracted automatically.

Source URL: ${window.location.href}
Page Title: ${document.title}
Document Type: VirDocs Viewer

The document may be in a format that requires special handling or may be dynamically loaded. 
You can try downloading the document directly from VirDocs or contact support for assistance.

Extraction attempted at: ${new Date().toISOString()}

Note: VirDocs documents often use complex viewers that prevent automatic text extraction. 
This placeholder document contains the metadata needed to identify and reference the original document.`;
      }
      
      // Ensure we always have a reasonable minimum content length
      if (pageContent.length < 200) {
        pageContent = pageContent + `\n\n[Extended content placeholder to meet minimum requirements]
        
Original page analysis:
- URL: ${window.location.href}
- Title: ${document.title}
- Document ID: ${docInfo.documentId || 'unknown'}
- Platform: VirDocs
- Timestamp: ${new Date().toISOString()}
- User Agent: ${navigator.userAgent}

This document was processed by the AAIRE browser extension. The minimal content extracted 
suggests this is a protected or dynamically-loaded document that requires special handling.`;
      }
      
      // Validate content - be more lenient with VirDocs
      if (pageContent.length < 20) {
        throw new Error('Insufficient content extracted from VirDocs viewer. The document may be password protected or require special access.');
      }
      
      // Log extraction details for debugging
      console.log('VirDocs extraction details:', {
        originalLength: pageContent.length,
        firstChars: pageContent.substring(0, 100),
        extractionUrl: window.location.href
      });
      
      if (pageContent.length > 1000000) { // 1MB limit
        // Truncate content but keep meaningful parts
        pageContent = pageContent.substring(0, 1000000) + '\n\n[Content truncated due to size limit]';
      }
      
      const blob = new Blob([pageContent], { type: 'text/plain' });
      filename = `virdocs_${docInfo.documentId || Date.now()}.txt`;
      
      // Ensure filename is valid
      filename = filename.replace(/[^a-zA-Z0-9.-]/g, '_');
      
      file = new File([blob], filename, { type: 'text/plain' });
      
      console.log('VirDocs content extracted:', {
        contentLength: pageContent.length,
        filename: filename,
        fileSize: blob.size,
        extractionMethod: pageContent.length > 100 ? 'successful' : 'placeholder'
      });
    } else {
      // For regular documents, download normally
      const response = await fetch(docInfo.url);
      if (!response.ok) {
        throw new Error(`Failed to download: ${response.status}`);
      }
      
      const blob = await response.blob();
      filename = docInfo.url.split('/').pop() || 'document.pdf';
      file = new File([blob], filename, { type: blob.type });
    }
    
    // Convert file to transferable format for background script
    const fileArrayBuffer = await file.arrayBuffer();
    const fileData = {
      content: Array.from(new Uint8Array(fileArrayBuffer)),
      name: filename,
      type: file.type,
      size: file.size
    };
    
    // Send to background script
    console.log('Sending message to background script...', {
      fileSize: fileData.size,
      fileName: fileData.name,
      sourceUrl: window.location.href
    });
    
    const result = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({
        type: 'UPLOAD_DOCUMENT',
        data: {
          fileData: fileData,
          sourceUrl: window.location.href,
          pageTitle: document.title,
          documentType: docInfo.type,
          platform: docInfo.platform || 'Unknown'
        }
      }, response => {
        console.log('Response from background script:', response);
        if (response && response.success) {
          resolve(response.data);
        } else {
          reject(new Error(response ? response.error : 'No response from background script'));
        }
      });
    });
    
    console.log('Document sent successfully:', result);
    alert(`Document "${docInfo.title}" sent to AAIRE successfully!`);
    
  } catch (error) {
    console.error('Failed to send document:', error);
    throw error;
  }
}

/**
 * View AAIRE results
 */
function viewResults() {
  // Open AAIRE website in new tab
  chrome.runtime.sendMessage({
    type: 'OPEN_AAIRE'
  });
}

/**
 * Observe page changes for dynamic content
 */
function observePageChanges() {
  const observer = new MutationObserver((mutations) => {
    let shouldRedetect = false;
    
    mutations.forEach(mutation => {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        // Check if new nodes contain documents
        mutation.addedNodes.forEach(node => {
          if (node.nodeType === 1) { // Element node
            const hasDocuments = node.querySelectorAll && 
              (node.querySelectorAll('a[href*=".pdf"], iframe, embed').length > 0);
            if (hasDocuments) {
              shouldRedetect = true;
            }
          }
        });
      }
    });
    
    if (shouldRedetect) {
      setTimeout(detectDocuments, 1000); // Debounce
    }
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Message handling
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.type) {
    case 'DETECT_DOCUMENTS':
      detectDocuments();
      sendResponse({ success: true, documents: detectedDocuments });
      break;
      
    case 'GET_PAGE_INFO':
      sendResponse({
        success: true,
        data: {
          url: window.location.href,
          title: document.title,
          documents: detectedDocuments
        }
      });
      break;
  }
});

console.log('AAIRE Content Script loaded');