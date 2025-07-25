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
        document.getElementById('dashboard-section').style.display = 'none';

        // Show selected section
        document.getElementById(`${section}-section`).style.display = 
            section === 'chat' ? 'flex' : 'block';

        // Update header title
        const titles = {
            'chat': 'Chat Assistant',
            'upload': 'Document Upload',
            'dashboard': 'System Dashboard'
        };
        document.getElementById('section-title').textContent = titles[section];

        // Load section-specific data
        if (section === 'dashboard') {
            this.loadDashboardData();
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
        
        messageContent += `<div class="message-meta">${new Date().toLocaleTimeString()}</div>`;
        messageContent += '</div>';
        
        // Add copy button for assistant messages
        if (sender === 'assistant') {
            const copyBtnId = `copy-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            messageContent += `
                <button class="copy-btn" id="${copyBtnId}" title="Copy response">
                    <i class="fas fa-copy"></i>
                </button>
            `;
        }
        
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
        // Copy text to clipboard
        navigator.clipboard.writeText(text).then(() => {
            // Change icon to checkmark
            const icon = button.querySelector('i');
            icon.className = 'fas fa-check';
            button.style.color = '#2ecc71';
            
            // Reset after 2 seconds
            setTimeout(() => {
                icon.className = 'fas fa-copy';
                button.style.color = '';
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text:', err);
            // Show error state
            const icon = button.querySelector('i');
            icon.className = 'fas fa-times';
            button.style.color = '#e74c3c';
            
            setTimeout(() => {
                icon.className = 'fas fa-copy';
                button.style.color = '';
            }, 2000);
        });
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
            const statusClass = file.status === 'accepted' ? 'success' : 
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
            
            // Restore messages
            this.messages.forEach(msg => {
                if (msg.sender !== 'assistant' || Date.now() - msg.timestamp < 24 * 60 * 60 * 1000) {
                    this.addMessage(msg.sender, msg.content, msg.sources);
                }
            });
        }
    }

    getSessionId() {
        let sessionId = localStorage.getItem('aaire_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('aaire_session_id', sessionId);
        }
        return sessionId;
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