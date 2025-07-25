/**
 * AAIRE Frontend JavaScript
 * Handles WebSocket communication, file uploads, and UI interactions
 */

class AAIREApp {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.sessionStart = Date.now();
        this.queryCount = 0;
        this.messages = [];
        this.uploadedFiles = [];
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.checkAPIHealth();
        this.updateSessionTimer();
        this.loadChatHistory();
        this.loadUploadedFiles();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.switchSection(e.currentTarget.dataset.section);
            });
        });

        // Chat input
        const chatInput = document.getElementById('chat-input');
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        chatInput.addEventListener('input', this.autoResizeTextarea);

        // File upload
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.getElementById('upload-area');
        const chooseFilesBtn = document.getElementById('choose-files-btn');
        
        fileInput.addEventListener('change', (e) => {
            console.log('File input change detected:', e.target.files.length, 'files');
            this.handleFileUpload(e.target.files);
        });
        
        // Choose files button
        chooseFilesBtn.addEventListener('click', () => {
            console.log('Choose files button clicked');
            console.log('File input element:', fileInput);
            console.log('File input exists:', !!fileInput);
            fileInput.click();
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });

        uploadArea.addEventListener('click', (e) => {
            // Don't trigger if clicking on the button inside the upload area
            if (e.target.closest('#choose-files-btn')) {
                return;
            }
            console.log('Upload area clicked - triggering file input');
            console.log('File input element:', fileInput);
            fileInput.click();
        });
    }

    switchSection(section) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`).classList.add('active');

        // Hide all sections
        document.getElementById('chat-section').style.display = 'none';
        document.getElementById('upload-section').style.display = 'none';
        document.getElementById('workflows-section').style.display = 'none';
        document.getElementById('dashboard-section').style.display = 'none';

        // Show selected section
        document.getElementById(`${section}-section`).style.display = 
            section === 'chat' ? 'flex' : 'block';

        // Update header title
        const titles = {
            'chat': 'Welcome to AAIRE',
            'upload': 'Document Upload',
            'workflows': 'Accounting Workflows',
            'dashboard': 'System Dashboard'
        };
        document.getElementById('section-title').textContent = titles[section];

        // Load section-specific data
        if (section === 'dashboard') {
            this.loadDashboardData();
        } else if (section === 'workflows') {
            this.loadWorkflows();
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/chat/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.connected = true;
                this.updateConnectionStatus(true);
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.connected = false;
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    if (!this.connected) {
                        this.connectWebSocket();
                    }
                }, 5000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.connected = false;
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    handleWebSocketMessage(data) {
        if (data.type === 'response') {
            this.hideTypingIndicator();
            this.addMessage('assistant', data.message, data.sources);
        } else if (data.type === 'error') {
            this.hideTypingIndicator();
            this.addMessage('error', `Error: ${data.message}`);
        } else if (data.type === 'status') {
            console.log('Status update:', data);
        }
    }

    sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message || !this.connected) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        
        // Clear input
        input.value = '';
        input.style.height = 'auto';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Send via WebSocket
        this.ws.send(JSON.stringify({
            type: 'query',
            message: message,
            session_id: this.getSessionId()
        }));
        
        this.queryCount++;
        this.updateQueryCount();
    }

    formatMessageContent(content) {
        // Format the content for better readability
        let formatted = content;
        
        // Convert numbered lists (e.g., "1. ", "2. ", etc.)
        formatted = formatted.replace(/^(\d+)\.\s+(.+)$/gm, '<strong>$1.</strong> $2');
        
        // Convert bullet points at start of line
        formatted = formatted.replace(/^[-•]\s+(.+)$/gm, '• $1');
        
        // Convert inline numbered lists in paragraphs
        formatted = formatted.replace(/(\d+)\.\s+([A-Z][^:]+):/g, '<br><br><strong>$1. $2:</strong>');
        
        // Add line breaks before section headers (text followed by colon at end of line)
        formatted = formatted.replace(/([A-Z][^:]+):$/gm, '<br><strong>$1:</strong>');
        
        // Convert double line breaks to paragraphs
        formatted = formatted.replace(/\n\n/g, '</p><p>');
        
        // Convert single line breaks to <br>
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Wrap in paragraph tags
        formatted = '<p>' + formatted + '</p>';
        
        // Clean up empty paragraphs
        formatted = formatted.replace(/<p><\/p>/g, '');
        formatted = formatted.replace(/<p><br>/g, '<p>');
        
        return formatted;
    }
    
    addMessage(sender, content, sources = null) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        let messageContent = `<div class="message-content">`;
        
        if (sender === 'error') {
            messageContent += `<div style="color: #e74c3c;">${content}</div>`;
        } else if (sender === 'assistant') {
            // Format assistant messages for better readability
            messageContent += this.formatMessageContent(content);
        } else {
            // User messages remain plain
            messageContent += content;
        }
        
        // Add sources if available
        if (sources && sources.length > 0) {
            // Deduplicate sources
            const uniqueSources = [...new Set(sources)];
            messageContent += '<br><br><small><strong>Source:</strong><br>';
            uniqueSources.forEach(source => {
                messageContent += `• ${source}<br>`;
            });
            messageContent += '</small>';
        }
        
        // Add copy button for assistant messages (inside the message content)
        if (sender === 'assistant') {
            messageContent += `
                <button class="copy-btn" title="Copy response">
                    <i class="fas fa-copy"></i>
                </button>
            `;
        }
        
        messageContent += `<div class="message-meta">${new Date().toLocaleTimeString()}</div>`;
        messageContent += '</div>';
        
        messageDiv.innerHTML = messageContent;
        
        // Add copy functionality if it's an assistant message
        if (sender === 'assistant') {
            const copyBtn = messageDiv.querySelector('.copy-btn');
            copyBtn.addEventListener('click', () => {
                this.copyToClipboard(content, copyBtn);
            });
        }
        messagesContainer.appendChild(messageDiv);
        
        // Force scroll to bottom with smooth behavior
        setTimeout(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            // Also trigger a scroll event to ensure it works
            messagesContainer.dispatchEvent(new Event('scroll'));
        }, 100);
        
        // Store message
        this.messages.push({
            sender,
            content,
            sources,
            timestamp: Date.now()
        });
        
        this.saveChatHistory();
    }

    copyToClipboard(text, button) {
        const icon = button.querySelector('i');
        
        // Try modern clipboard API first
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(() => {
                this.showCopySuccess(icon, button);
            }).catch(err => {
                console.error('Clipboard API failed:', err);
                this.fallbackCopyText(text, icon, button);
            });
        } else {
            // Fallback for older browsers or non-HTTPS
            this.fallbackCopyText(text, icon, button);
        }
    }
    
    fallbackCopyText(text, icon, button) {
        try {
            // Create a temporary textarea element
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            // Try to copy using execCommand
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (successful) {
                this.showCopySuccess(icon, button);
            } else {
                this.showCopyError(icon, button);
            }
        } catch (err) {
            console.error('Fallback copy failed:', err);
            this.showCopyError(icon, button);
        }
    }
    
    showCopySuccess(icon, button) {
        icon.className = 'fas fa-check';
        button.style.color = '#2ecc71';
        button.title = 'Copied!';
        
        setTimeout(() => {
            icon.className = 'fas fa-copy';
            button.style.color = '';
            button.title = 'Copy response';
        }, 2000);
    }
    
    showCopyError(icon, button) {
        icon.className = 'fas fa-times';
        button.style.color = '#e74c3c';
        button.title = 'Copy failed';
        
        setTimeout(() => {
            icon.className = 'fas fa-copy';
            button.style.color = '';
            button.title = 'Copy response';
        }, 2000);
    }

    showTypingIndicator() {
        document.getElementById('typing-indicator').style.display = 'flex';
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTypingIndicator() {
        document.getElementById('typing-indicator').style.display = 'none';
    }

    autoResizeTextarea(event) {
        const textarea = event.target;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;
        
        const progressContainer = document.getElementById('upload-progress');
        const progressFill = document.getElementById('progress-fill');
        const statusDiv = document.getElementById('upload-status');
        
        // Reset status color and text
        statusDiv.style.color = '#333';
        progressContainer.style.display = 'block';
        
        console.log('Starting file upload for', files.length, 'files');
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append('file', file);
            
            // Add required metadata (must match backend validation)
            const metadata = {
                title: file.name.replace(/\.[^/.]+$/, ""), // Remove extension
                source_type: "COMPANY", // Must be uppercase - valid options: US_GAAP, IFRS, COMPANY, ACTUARIAL
                effective_date: new Date().toISOString().split('T')[0], // Today's date
                document_type: "uploaded_document",
                uploaded_by: "web_interface"
            };
            formData.append('metadata', JSON.stringify(metadata));
            
            statusDiv.textContent = `Uploading ${file.name} (${i + 1}/${files.length})...`;
            
            console.log('Uploading file:', file.name, 'with metadata:', metadata);
            
            try {
                const response = await fetch('/api/v1/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('Upload success:', result);
                    progressFill.style.width = `${((i + 1) / files.length) * 100}%`;
                    
                    // Add to uploaded files list
                    this.addUploadedFile({
                        name: file.name,
                        size: file.size,
                        type: file.type,
                        job_id: result.job_id,
                        status: result.status,
                        message: result.message,
                        uploaded_at: new Date().toISOString()
                    });
                    
                    if (i === files.length - 1) {
                        statusDiv.textContent = 'Upload completed successfully!';
                        statusDiv.style.color = '#2ecc71';
                        setTimeout(() => {
                            progressContainer.style.display = 'none';
                            progressFill.style.width = '0%';
                        }, 3000);
                    }
                } else {
                    const errorText = await response.text();
                    console.error('Upload failed:', response.status, errorText);
                    let errorMsg = `Upload failed: ${response.status} ${response.statusText}`;
                    try {
                        const errorJson = JSON.parse(errorText);
                        if (errorJson.detail) {
                            errorMsg = errorJson.detail;
                        }
                    } catch (e) {
                        // Use default error message if response isn't JSON
                    }
                    throw new Error(errorMsg);
                }
            } catch (error) {
                console.error('Upload error:', error);
                statusDiv.textContent = `Upload failed: ${error.message}`;
                statusDiv.style.color = '#e74c3c';
            }
        }
    }

    addUploadedFile(fileInfo) {
        this.uploadedFiles.unshift(fileInfo); // Add to beginning of array
        this.updateUploadedFilesList();
        this.saveUploadedFiles();
    }

    updateUploadedFilesList() {
        const filesList = document.getElementById('files-list');
        
        if (this.uploadedFiles.length === 0) {
            filesList.innerHTML = '<div class="no-files">No files uploaded yet</div>';
            return;
        }

        filesList.innerHTML = this.uploadedFiles.map(file => {
            const fileSize = this.formatFileSize(file.size);
            const uploadTime = new Date(file.uploaded_at).toLocaleString();
            const statusClass = file.status === 'completed' ? 'success' : 
                               file.status === 'accepted' ? 'success' :
                               file.status === 'processing' ? 'processing' : 'error';
            
            return `
                <div class="file-item" data-job-id="${file.job_id}">
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-meta">${fileSize} • Uploaded ${uploadTime}</div>
                        <div class="file-meta">Job ID: ${file.job_id}</div>
                    </div>
                    <div class="file-actions">
                        <div class="file-status ${statusClass}">${file.status}</div>
                        ${file.status === 'completed' ? `
                            <button class="summary-btn" onclick="window.app.viewSummary('${file.job_id}', '${file.name}')" title="View AI Summary">
                                <i class="fas fa-file-alt"></i>
                            </button>
                        ` : ''}
                        <button class="delete-btn" onclick="window.app.deleteFile('${file.job_id}', '${file.name}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    saveUploadedFiles() {
        try {
            localStorage.setItem('aaire_uploaded_files', JSON.stringify(this.uploadedFiles));
        } catch (e) {
            console.warn('Could not save uploaded files:', e);
        }
    }

    loadUploadedFiles() {
        try {
            const saved = localStorage.getItem('aaire_uploaded_files');
            if (saved) {
                this.uploadedFiles = JSON.parse(saved);
                this.updateUploadedFilesList();
            }
        } catch (e) {
            console.warn('Could not load uploaded files:', e);
        }
    }

    async viewSummary(jobId, fileName) {
        try {
            const response = await fetch(`/api/v1/documents/${jobId}/summary`);
            
            if (response.ok) {
                const summaryData = await response.json();
                this.displaySummaryModal(summaryData);
            } else {
                alert('Summary not available for this document yet. Please try again later.');
            }
        } catch (error) {
            console.error('Error fetching summary:', error);
            alert('Unable to load document summary. Please try again.');
        }
    }

    displaySummaryModal(summaryData) {
        // Create modal overlay
        const overlay = document.createElement('div');
        overlay.className = 'summary-modal-overlay';
        overlay.innerHTML = `
            <div class="summary-modal">
                <div class="summary-header">
                    <h2><i class="fas fa-file-alt"></i> Document Summary</h2>
                    <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="summary-content">
                    <div class="document-info">
                        <h3>${summaryData.document_info.filename}</h3>
                        <p><strong>Status:</strong> ${summaryData.document_info.status}</p>
                        <p><strong>Processed:</strong> ${new Date(summaryData.document_info.created_at).toLocaleString()}</p>
                    </div>
                    
                    <div class="ai-summary">
                        <h4><i class="fas fa-brain"></i> AI-Generated Executive Summary</h4>
                        <div class="summary-text">${this.formatMessageContent(summaryData.summary.summary)}</div>
                    </div>
                    
                    ${summaryData.summary.key_insights && summaryData.summary.key_insights.length > 0 ? `
                        <div class="key-insights">
                            <h4><i class="fas fa-lightbulb"></i> Key Insights</h4>
                            <div class="insights-grid">
                                ${summaryData.summary.key_insights.map(insight => `
                                    <div class="insight-card">
                                        <h5>${insight.description}</h5>
                                        <p>${Array.isArray(insight.value) ? insight.value.join(', ') : insight.value}</p>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class="summary-actions">
                        <button class="btn" onclick="navigator.clipboard.writeText('${summaryData.summary.summary.replace(/'/g, "\\'")}')">
                            <i class="fas fa-copy"></i> Copy Summary
                        </button>
                        <button class="btn btn-secondary" onclick="this.parentElement.parentElement.parentElement.parentElement.remove()">
                            Close
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }

    async deleteFile(jobId, fileName) {
        if (!confirm(`Are you sure you want to delete "${fileName}"?`)) {
            return;
        }

        try {
            // Remove from local list immediately for better UX
            this.uploadedFiles = this.uploadedFiles.filter(f => f.job_id !== jobId);
            this.updateUploadedFilesList();
            this.saveUploadedFiles();

            // Call backend API to delete from vector store
            const response = await fetch(`/api/v1/documents/${jobId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                console.log(`File ${fileName} deleted successfully`);
            } else {
                // If backend fails, add it back to the list
                console.error('Failed to delete file from backend');
                this.loadUploadedFiles(); // Reload from localStorage
            }
        } catch (error) {
            console.error('Error deleting file:', error);
            this.loadUploadedFiles(); // Reload from localStorage
        }
    }

    async checkAPIHealth() {
        try {
            const start = Date.now();
            const response = await fetch('/health');
            const responseTime = Date.now() - start;
            
            if (response.ok) {
                document.getElementById('api-status').textContent = 'Online';
                document.getElementById('api-status').className = 'status-online';
                document.getElementById('dash-api-status').textContent = 'Healthy';
                document.getElementById('response-time').textContent = `${responseTime}ms`;
            } else {
                throw new Error('API unhealthy');
            }
        } catch (error) {
            document.getElementById('api-status').textContent = 'Offline';
            document.getElementById('api-status').className = 'status-offline';
            document.getElementById('dash-api-status').textContent = 'Unhealthy';
        }
        
        // Check again in 30 seconds
        setTimeout(() => this.checkAPIHealth(), 30000);
    }

    updateConnectionStatus(connected) {
        const statusDiv = document.getElementById('connection-status');
        const wsStatus = document.getElementById('ws-status');
        
        if (connected) {
            statusDiv.className = 'connection-status connected';
            statusDiv.innerHTML = '<i class="fas fa-circle"></i> Connected';
            wsStatus.textContent = 'Connected';
            wsStatus.className = 'status-online';
        } else {
            statusDiv.className = 'connection-status disconnected';
            statusDiv.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
            wsStatus.textContent = 'Disconnected';
            wsStatus.className = 'status-offline';
        }
    }

    updateSessionTimer() {
        const sessionTime = Math.floor((Date.now() - this.sessionStart) / 1000 / 60);
        document.getElementById('session-time').textContent = `${sessionTime}m`;
        
        setTimeout(() => this.updateSessionTimer(), 60000);
    }

    updateQueryCount() {
        document.getElementById('total-queries').textContent = this.queryCount;
    }

    async loadDashboardData() {
        try {
            // Load knowledge base stats
            const response = await fetch('/api/v1/knowledge/stats');
            if (response.ok) {
                const stats = await response.json();
                document.getElementById('gaap-docs').textContent = stats.us_gaap || 0;
                document.getElementById('ifrs-docs').textContent = stats.ifrs || 0;
                document.getElementById('custom-docs').textContent = stats.custom || 0;
                document.getElementById('docs-processed').textContent = 
                    (stats.us_gaap || 0) + (stats.ifrs || 0) + (stats.custom || 0);
            }
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }

    saveChatHistory() {
        try {
            localStorage.setItem('aaire_chat_history', JSON.stringify(this.messages));
        } catch (e) {
            if (e.name === 'QuotaExceededError') {
                // Clear old data if storage is full
                console.warn('LocalStorage full, clearing old chat history');
                localStorage.removeItem('aaire_chat_history');
                localStorage.removeItem('aaire_session_id');
                
                // Keep only last 10 messages
                if (this.messages.length > 10) {
                    this.messages = this.messages.slice(-10);
                }
                
                // Try saving again with reduced data
                try {
                    localStorage.setItem('aaire_chat_history', JSON.stringify(this.messages));
                } catch (e2) {
                    console.warn('Could not save chat history:', e2);
                }
            }
        }
    }

    loadChatHistory() {
        const history = localStorage.getItem('aaire_chat_history');
        if (history) {
            this.messages = JSON.parse(history);
            const messagesContainer = document.getElementById('chat-messages');
            
            // Clear existing messages except welcome
            const welcomeMsg = messagesContainer.querySelector('.message');
            messagesContainer.innerHTML = '';
            messagesContainer.appendChild(welcomeMsg);
            
            // Restore messages without adding to this.messages again
            this.messages.forEach(msg => {
                if (msg.sender !== 'assistant' || Date.now() - msg.timestamp < 24 * 60 * 60 * 1000) {
                    this.displayMessage(msg.sender, msg.content, msg.sources);
                }
            });
        }
    }

    displayMessage(sender, content, sources = null) {
        // Similar to addMessage but doesn't store in this.messages or save to localStorage
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        let messageContent = `<div class="message-content">`;
        
        if (sender === 'error') {
            messageContent += `<div style="color: #e74c3c;">${content}</div>`;
        } else if (sender === 'assistant') {
            messageContent += this.formatMessageContent(content);
        } else {
            messageContent += content;
        }
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const uniqueSources = [...new Set(sources)];
            messageContent += '<br><br><small><strong>Source:</strong><br>';
            uniqueSources.forEach(source => {
                messageContent += `• ${source}<br>`;
            });
            messageContent += '</small>';
        }
        
        // Add copy button for assistant messages (inside the message content)
        if (sender === 'assistant') {
            messageContent += `
                <button class="copy-btn" title="Copy response">
                    <i class="fas fa-copy"></i>
                </button>
            `;
        }
        
        messageContent += `<div class="message-meta">${new Date().toLocaleTimeString()}</div>`;
        messageContent += '</div>';
        
        messageDiv.innerHTML = messageContent;
        
        // Add copy functionality if it's an assistant message
        if (sender === 'assistant') {
            const copyBtn = messageDiv.querySelector('.copy-btn');
            copyBtn.addEventListener('click', () => {
                this.copyToClipboard(content, copyBtn);
            });
        }
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        setTimeout(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 100);
    }

    getSessionId() {
        let sessionId = localStorage.getItem('aaire_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('aaire_session_id', sessionId);
        }
        return sessionId;
    }

    // Workflow Management
    async loadWorkflows() {
        try {
            const response = await fetch('/api/v1/workflows');
            if (response.ok) {
                const data = await response.json();
                this.displayWorkflows(data.workflows);
            } else {
                document.getElementById('workflows-grid').innerHTML = 
                    '<div class="loading-workflows">Failed to load workflows</div>';
            }
        } catch (error) {
            console.error('Error loading workflows:', error);
            document.getElementById('workflows-grid').innerHTML = 
                '<div class="loading-workflows">Error loading workflows</div>';
        }
    }

    displayWorkflows(workflows) {
        const grid = document.getElementById('workflows-grid');
        
        if (workflows.length === 0) {
            grid.innerHTML = '<div class="loading-workflows">No workflows available</div>';
            return;
        }

        grid.innerHTML = workflows.map(workflow => `
            <div class="workflow-card" onclick="window.app.startWorkflow('${workflow.id}')">
                <div class="workflow-card-header">
                    <div class="workflow-icon">
                        <i class="fas fa-route"></i>
                    </div>
                    <h4>${workflow.name}</h4>
                </div>
                <p>${workflow.description}</p>
                <div class="workflow-meta">
                    <span><i class="fas fa-clock"></i> ${workflow.estimated_time}</span>
                    <span class="difficulty-badge difficulty-${workflow.difficulty}">${workflow.difficulty}</span>
                </div>
            </div>
        `).join('');
    }

    async startWorkflow(templateId) {
        try {
            const sessionId = `workflow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            const response = await fetch(`/api/v1/workflows/${templateId}/start?session_id=${sessionId}`, {
                method: 'POST'
            });

            if (response.ok) {
                const data = await response.json();
                this.currentWorkflowSession = sessionId;
                this.showActiveWorkflow(data);
            } else {
                alert('Failed to start workflow. Please try again.');
            }
        } catch (error) {
            console.error('Error starting workflow:', error);
            alert('Error starting workflow. Please try again.');
        }
    }

    showActiveWorkflow(workflowData) {
        // Hide workflow library, show active workflow
        document.querySelector('.workflow-library').style.display = 'none';
        document.getElementById('active-workflow').style.display = 'block';

        // Update workflow header
        document.getElementById('workflow-title').textContent = workflowData.workflow_name;
        document.getElementById('workflow-progress').style.width = `${workflowData.progress.percentage}%`;
        document.getElementById('workflow-step-counter').textContent = 
            `Step ${workflowData.progress.current} of ${workflowData.progress.total}`;

        // Update step content
        this.displayWorkflowStep(workflowData.current_step);
    }

    displayWorkflowStep(step) {
        document.getElementById('step-title').textContent = step.title;
        document.getElementById('step-description').textContent = step.description;
        document.getElementById('step-instruction').textContent = step.instruction;
        document.getElementById('step-help-text').textContent = step.help_text || '';

        // Create appropriate input based on step type
        const inputContainer = document.getElementById('step-input-container');
        inputContainer.innerHTML = '';

        if (step.input_type === 'choice') {
            const choicesDiv = document.createElement('div');
            choicesDiv.className = 'workflow-choices';
            
            step.choices.forEach(choice => {
                const option = document.createElement('div');
                option.className = 'choice-option';
                option.dataset.value = choice.value;
                option.innerHTML = `
                    <input type="radio" name="workflow-choice" value="${choice.value}">
                    <span>${choice.label}</span>
                `;
                
                option.addEventListener('click', () => {
                    document.querySelectorAll('.choice-option').forEach(opt => opt.classList.remove('selected'));
                    option.classList.add('selected');
                    option.querySelector('input').checked = true;
                });
                
                choicesDiv.appendChild(option);
            });
            
            inputContainer.appendChild(choicesDiv);
        } else {
            const input = document.createElement(step.input_type === 'text' && step.instruction.toLowerCase().includes('list') ? 'textarea' : 'input');
            input.className = 'workflow-input';
            input.id = 'workflow-step-input';
            
            if (input.tagName === 'TEXTAREA') {
                input.className += ' workflow-textarea';
                input.rows = 4;
            } else {
                input.type = 'text';
            }
            
            if (step.validation && step.validation.type === 'number') {
                input.type = 'number';
                if (step.validation.min !== undefined) {
                    input.min = step.validation.min;
                }
            }
            
            input.placeholder = step.help_text || 'Enter your response...';
            inputContainer.appendChild(input);
        }

        // Update next button
        const nextBtn = document.getElementById('step-next-btn');
        nextBtn.textContent = step.required ? 'Next Step' : 'Skip Step';
        nextBtn.innerHTML = step.required ? 
            '<i class="fas fa-arrow-right"></i> Next Step' : 
            '<i class="fas fa-arrow-right"></i> Skip Step';
    }

    async submitWorkflowStep() {
        try {
            let response = '';
            
            // Get response based on input type
            const selectedChoice = document.querySelector('input[name="workflow-choice"]:checked');
            if (selectedChoice) {
                response = selectedChoice.value;
            } else {
                const input = document.getElementById('workflow-step-input');
                if (input) {
                    response = input.value.trim();
                }
            }

            // Submit response
            const apiResponse = await fetch(`/api/v1/workflows/${this.currentWorkflowSession}/step`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ response })
            });

            if (apiResponse.ok) {
                const data = await apiResponse.json();
                
                if (data.status === 'continue') {
                    // Move to next step
                    this.updateWorkflowProgress(data.progress);
                    this.displayWorkflowStep(data.current_step);
                } else if (data.status === 'completed') {
                    // Workflow completed
                    this.showWorkflowCompletion(data);
                }
            } else {
                const error = await apiResponse.json();
                alert(error.detail || 'Error processing step. Please try again.');
            }
        } catch (error) {
            console.error('Error submitting workflow step:', error);
            alert('Error submitting step. Please try again.');
        }
    }

    updateWorkflowProgress(progress) {
        document.getElementById('workflow-progress').style.width = `${progress.percentage}%`;
        document.getElementById('workflow-step-counter').textContent = 
            `Step ${progress.current} of ${progress.total}`;
    }

    showWorkflowCompletion(data) {
        const container = document.getElementById('active-workflow');
        container.innerHTML = `
            <div class="workflow-header">
                <h3><i class="fas fa-check-circle" style="color: #2ecc71;"></i> Workflow Completed!</h3>
                <button class="btn btn-secondary" onclick="window.app.exitWorkflow()">
                    <i class="fas fa-times"></i> Close
                </button>
            </div>
            <div class="workflow-content">
                <div class="completion-summary">
                    <h4>Summary</h4>
                    <p><strong>Completed at:</strong> ${new Date(data.completed_at).toLocaleString()}</p>
                    <p><strong>Duration:</strong> ${data.summary.duration_minutes} minutes</p>
                    <p><strong>Steps completed:</strong> ${data.summary.steps_completed}</p>
                    
                    ${data.summary.recommendations ? `
                        <h4>Recommendations</h4>
                        <ul>
                            ${data.summary.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    ` : ''}
                    
                    <div class="step-actions">
                        <button class="btn" onclick="navigator.clipboard.writeText('${JSON.stringify(data.all_responses)}')">
                            <i class="fas fa-copy"></i> Copy Responses
                        </button>
                        <button class="btn btn-secondary" onclick="window.app.exitWorkflow()">
                            <i class="fas fa-arrow-left"></i> Back to Workflows
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    exitWorkflow() {
        // Reset workflow state
        this.currentWorkflowSession = null;
        
        // Show workflow library, hide active workflow
        document.querySelector('.workflow-library').style.display = 'block';
        document.getElementById('active-workflow').style.display = 'none';
        
        // Reload workflows
        this.loadWorkflows();
    }
}

// Global functions
function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = `
            <div class="message">
                <div class="message-content">
                    <strong>Chat cleared.</strong><br>
                    How can I assist you today?
                </div>
            </div>
        `;
        
        if (window.app) {
            app.messages = [];
            app.saveChatHistory();
        }
        
        // Also clear localStorage directly
        localStorage.removeItem('aaire_chat_history');
    }
}

function exportChat() {
    const chatData = {
        messages: app.messages,
        exported_at: new Date().toISOString(),
        session_id: app.getSessionId()
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aaire_chat_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Test function to add multiple messages for scroll testing
function testScrolling() {
    if (!window.app) return;
    
    for (let i = 1; i <= 20; i++) {
        app.addMessage('user', `Test message ${i} - This is a longer message to test scrolling functionality in the chat interface.`);
        app.addMessage('assistant', `Response ${i} - This is AAIRE's response to test message ${i}. The chat should automatically scroll to show the latest messages at the bottom.`);
    }
}

// Test function to simulate file upload
function testUpload() {
    if (!window.app) return;
    
    console.log('Testing upload functionality...');
    
    // Create a test CSV file (supported format)
    const testContent = "Account Type,Premium,Reserves\nLife Insurance,1000000,800000\nProperty Insurance,500000,400000";
    const testFile = new File([testContent], "test_upload.csv", { type: "text/csv" });
    
    // Simulate upload
    app.handleFileUpload([testFile]);
}

// Test function to check file input
function testFileInput() {
    const fileInput = document.getElementById('file-input');
    console.log('File input test:');
    console.log('Element exists:', !!fileInput);
    console.log('Element:', fileInput);
    console.log('Clicking file input...');
    if (fileInput) {
        fileInput.click();
    }
}

// Function to clear uploaded files list
function clearUploadedFiles() {
    if (window.app) {
        app.uploadedFiles = [];
        app.updateUploadedFilesList();
        app.saveUploadedFiles();
        console.log('Uploaded files list cleared');
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new AAIREApp();
    // Make app globally accessible for debugging
    window.app = app;
});