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
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.checkAPIHealth();
        this.updateSessionTimer();
        this.loadChatHistory();
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
        
        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
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

        uploadArea.addEventListener('click', () => {
            document.getElementById('file-input').click();
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

    addMessage(sender, content, sources = null) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        let messageContent = `<div class="message-content">`;
        
        if (sender === 'error') {
            messageContent += `<div style="color: #e74c3c;">${content}</div>`;
        } else {
            messageContent += content;
        }
        
        // Add sources if available
        if (sources && sources.length > 0) {
            messageContent += '<br><br><small><strong>Sources:</strong><br>';
            sources.forEach(source => {
                messageContent += `• ${source}<br>`;
            });
            messageContent += '</small>';
        }
        
        messageContent += `<div class="message-meta">${new Date().toLocaleTimeString()}</div>`;
        messageContent += '</div>';
        
        messageDiv.innerHTML = messageContent;
        messagesContainer.appendChild(messageDiv);
        
        // Auto-scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Store message
        this.messages.push({
            sender,
            content,
            sources,
            timestamp: Date.now()
        });
        
        this.saveChatHistory();
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
        
        progressContainer.style.display = 'block';
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append('file', file);
            
            statusDiv.textContent = `Uploading ${file.name} (${i + 1}/${files.length})...`;
            
            try {
                const response = await fetch('/api/v1/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    progressFill.style.width = `${((i + 1) / files.length) * 100}%`;
                    
                    if (i === files.length - 1) {
                        statusDiv.textContent = 'Upload completed successfully!';
                        setTimeout(() => {
                            progressContainer.style.display = 'none';
                            progressFill.style.width = '0%';
                        }, 2000);
                    }
                } else {
                    throw new Error(`Upload failed: ${response.statusText}`);
                }
            } catch (error) {
                console.error('Upload error:', error);
                statusDiv.textContent = `Upload failed: ${error.message}`;
                statusDiv.style.color = '#e74c3c';
            }
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
        localStorage.setItem('aaire_chat_history', JSON.stringify(this.messages));
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
        
        app.messages = [];
        app.saveChatHistory();
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

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new AAIREApp();
});