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
        this.currentUser = null;
        this.updateTimeout = null;
        
        // CRITICAL: Clean up all stale localStorage data on app start
        this.performOneTimeCleanup();
        
        this.init();
    }
    
    performOneTimeCleanup() {
        // FORCE CACHE REFRESH - Check if user has old cached version
        const currentVersion = '17-cleanup-fix';
        const lastVersion = localStorage.getItem('aaire_app_version');
        
        if (lastVersion !== currentVersion) {
            console.log(`🔄 Version mismatch detected: ${lastVersion} → ${currentVersion}`);
            console.log('🧹 Forcing cache refresh and performing comprehensive cleanup...');
            
            // Clear ALL localStorage to ensure clean state
            const allKeys = Object.keys(localStorage);
            allKeys.forEach(key => {
                if (key.startsWith('aaire_')) {
                    localStorage.removeItem(key);
                    console.log(`🗑️ Removed: ${key}`);
                }
            });
            
            // Set new version
            localStorage.setItem('aaire_app_version', currentVersion);
            
            console.log('✅ Cache cleared and version updated. App will use fresh code.');
            
            // Force a hard refresh to ensure no cached JavaScript
            if (lastVersion && lastVersion !== currentVersion) {
                console.log('🔄 Performing hard refresh to load latest code...');
                window.location.reload(true);
                return;
            }
        }
        
        console.log('🧹 Performing one-time cleanup of stale localStorage data');
        
        // List of all possible stale keys that could cause issues
        const staleKeys = [
            // Old chat history format (causes restoration issues)
            'aaire_chat_history',
            'aaire_chat_history_Court_Williams',
            'aaire_chat_history_Bill_Smith',
            'aaire_chat_history_Sarah_Chen',
            'aaire_chat_history_Bob_Johnson',
            'aaire_chat_history_Bill_Johnson',
            'aaire_chat_history_Sarah_Davis', // Current user having issues
            
            // Old session data that might contain false citations
            'aaire_session_id',
            
            // Uploaded files data that might reference false documents
            'aaire_uploaded_files'
        ];
        
        let removedCount = 0;
        staleKeys.forEach(key => {
            if (localStorage.getItem(key)) {
                localStorage.removeItem(key);
                removedCount++;
                console.log(`✅ Removed stale data: ${key}`);
            }
        });
        
        // Also clear any session data with false citations
        this.clearFalseCitationSessions();
        
        console.log(`🎉 Cleanup complete: ${removedCount} stale keys removed`);
        
        // Also clear server-side caches to eliminate false citations
        this.clearServerCaches();
    }
    
    clearFalseCitationSessions() {
        console.log('🔍 Scanning session data for false citations...');
        
        const users = ['Court_Williams', 'Bill_Smith', 'Sarah_Chen', 'Bob_Johnson', 'Bill_Johnson', 'Sarah_Davis'];
        let cleanedSessions = 0;
        
        users.forEach(username => {
            const sessionsKey = `aaire_chat_sessions_${username}`;
            const sessionsData = localStorage.getItem(sessionsKey);
            
            if (sessionsData) {
                try {
                    const sessions = JSON.parse(sessionsData);
                    const originalLength = sessions.length;
                    
                    // Filter out sessions containing false citations
                    const cleanedSessions = sessions.filter(session => {
                        // Check if session contains false citations
                        const hasFalseCitation = session.messages?.some(msg => {
                            if (msg.sources) {
                                return msg.sources.some(source => 
                                    source.toLowerCase().includes('canadian tax book') ||
                                    source.toLowerCase().includes('.pdf') && msg.content.toLowerCase().includes('how can i assist')
                                );
                            }
                            return false;
                        });
                        
                        return !hasFalseCitation;
                    });
                    
                    if (cleanedSessions.length !== originalLength) {
                        localStorage.setItem(sessionsKey, JSON.stringify(cleanedSessions));
                        console.log(`🧹 Cleaned ${username}: ${originalLength} → ${cleanedSessions.length} sessions`);
                        cleanedSessions++;
                    }
                    
                } catch (error) {
                    console.error(`Error cleaning sessions for ${username}:`, error);
                }
            }
        });
        
        console.log(`✅ Session cleanup complete: ${cleanedSessions} users cleaned`);
    }
    
    async clearServerCaches() {
        console.log('🧹 Clearing server-side caches to eliminate false citations...');
        
        try {
            // Clear server cache
            const cacheResponse = await fetch('/api/v1/debug/clear-cache', { method: 'POST' });
            if (cacheResponse.ok) {
                console.log('✅ Server cache cleared');
            }
            
            // Clear application state
            const stateResponse = await fetch('/api/v1/debug/restart-state', { method: 'POST' });
            if (stateResponse.ok) {
                console.log('✅ Server application state cleared');
            }
            
        } catch (error) {
            console.warn('⚠️ Could not clear server caches (server may not be ready yet):', error.message);
        }
    }

    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.checkAPIHealth();
        this.updateSessionTimer();
        // Load chat history for current user (will be set after user selection)
        this.loadUploadedFiles();
        this.initializeUser();
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

        // Event delegation for file action buttons (summary and delete)
        document.addEventListener('click', (e) => {
            // Handle summary button clicks
            if (e.target.closest('.summary-btn')) {
                const button = e.target.closest('.summary-btn');
                const jobId = button.dataset.jobId;
                const filename = button.dataset.filename;
                if (jobId && filename) {
                    e.preventDefault();
                    this.viewSummary(jobId, filename);
                }
            }
            
            // Handle delete button clicks
            if (e.target.closest('.delete-btn')) {
                const button = e.target.closest('.delete-btn');
                const jobId = button.dataset.jobId;
                const filename = button.dataset.filename;
                if (jobId && filename) {
                    e.preventDefault();
                    this.deleteFile(jobId, filename);
                }
            }
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
            this.addMessage('assistant', data.message, data.sources, data.followUpQuestions);
        } else if (data.type === 'error') {
            this.hideTypingIndicator();
            this.addMessage('error', `Error: ${data.message}`);
        } else if (data.type === 'status') {
            console.log('Status update:', data);
        }
    }

    sendMessage() {
        console.log('📨 App.sendMessage() called');
        
        const input = document.getElementById('chat-input');
        console.log('📝 Input element:', input ? 'FOUND' : 'NOT FOUND');
        
        if (!input) {
            console.error('❌ Chat input element not found');
            alert('Error: Chat input not found');
            return;
        }
        
        const message = input.value.trim();
        console.log('💬 Message content:', message || 'EMPTY');
        
        if (!message) {
            console.log('⚠️ Empty message, returning');
            return;
        }
        
        console.log('✅ Adding user message to chat');
        // Add user message to chat
        this.addMessage('user', message);
        
        // Clear input
        input.value = '';
        input.style.height = 'auto';
        
        // Show typing indicator
        console.log('⏳ Showing typing indicator');
        this.showTypingIndicator();
        
        // Send via WebSocket if connected, otherwise use HTTP fallback
        console.log('🌐 Connection status - connected:', this.connected, 'ws state:', this.ws ? this.ws.readyState : 'NULL');
        
        if (this.connected && this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('📡 Sending via WebSocket');
            this.ws.send(JSON.stringify({
                type: 'query',
                message: message,
                session_id: this.getSessionId(),
                conversation_history: this.getConversationHistory(),
                user_context: this.getCurrentUserContext()
            }));
        } else {
            console.log('🔄 Using HTTP fallback');
            // HTTP fallback when WebSocket is not available
            this.sendMessageHTTP(message);
        }
        
        this.queryCount++;
        this.updateQueryCount();
        console.log('✅ App.sendMessage() completed');
    }

    async sendMessageHTTP(message) {
        try {
            const response = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: message,
                    session_id: this.getSessionId(),
                    conversation_history: this.getConversationHistory(),
                    user_context: this.getCurrentUserContext()
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.hideTypingIndicator();
                this.addMessage('assistant', data.response, data.citations, data.follow_up_questions);
            } else {
                this.hideTypingIndicator();
                this.addMessage('assistant', 'Sorry, I encountered an error processing your request. Please try again.');
            }
        } catch (error) {
            console.error('HTTP fallback error:', error);
            this.hideTypingIndicator();
            this.addMessage('assistant', 'Sorry, I encountered a connection error. Please check your internet connection and try again.');
        }
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
    
    addMessage(sender, content, sources = null, followUpQuestions = null) {
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
            // Handle both string array (WebSocket) and object array (HTTP) formats
            const sourceList = sources.map(source => {
                if (typeof source === 'string') {
                    return source;
                } else if (source && source.source) {
                    return source.source;
                } else {
                    return 'Unknown Source';
                }
            });
            
            // Deduplicate sources
            const uniqueSources = [...new Set(sourceList)];
            messageContent += '<br><br><small><strong>Source:</strong><br>';
            uniqueSources.forEach(source => {
                messageContent += `• ${source}<br>`;
            });
            messageContent += '</small>';
        }
        
        // Add follow-up questions for assistant messages
        if (sender === 'assistant' && followUpQuestions && followUpQuestions.length > 0) {
            messageContent += '<div class="follow-up-questions">';
            messageContent += '<div class="follow-up-title">💡 Suggested follow-up questions:</div>';
            followUpQuestions.forEach((question, index) => {
                messageContent += `
                    <button class="follow-up-btn" onclick="askFollowUpQuestion('${question.replace(/'/g, "\\'")}')">
                        ${question}
                    </button>
                `;
            });
            messageContent += '</div>';
        }
        
        // Add feedback buttons for assistant messages
        if (sender === 'assistant') {
            const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            messageContent += `
                <div class="response-feedback">
                    <span class="feedback-label">Was this helpful?</span>
                    <button class="feedback-btn thumbs-up" onclick="submitFeedback('${messageId}', 'thumbs_up')" title="Helpful">
                        👍
                    </button>
                    <button class="feedback-btn thumbs-down" onclick="submitFeedback('${messageId}', 'thumbs_down')" title="Not helpful">
                        👎
                    </button>
                    <button class="feedback-btn report-issue" onclick="reportIssue('${messageId}')" title="Report issue">
                        ⚠️
                    </button>
                </div>
            `;
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
        
        // Store message with all necessary data for reconstruction
        this.messages.push({
            sender,
            content,
            sources,
            followUpQuestions,
            timestamp: Date.now()
        });
        
        // Auto-save disabled - users now manage sessions manually via Clear Chat
        // this.saveUserChatHistory(this.currentUser.name);
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
                    console.log('Initial status received:', result.status);
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
                    
                    // Check for status updates with retries
                    this.pollDocumentStatus(result.job_id, 0);
                    
                    if (i === files.length - 1) {
                        statusDiv.innerHTML = '<i class="fas fa-check"></i> Upload completed! Processing documents & generating summaries...';
                        statusDiv.style.color = '#2ecc71';
                        setTimeout(() => {
                            statusDiv.innerHTML = '<i class="fas fa-cog fa-spin"></i> AI summaries being generated...';
                            statusDiv.style.color = '#f39c12';
                        }, 2000);
                        setTimeout(() => {
                            progressContainer.style.display = 'none';
                            progressFill.style.width = '0%';
                        }, 8000); // Longer delay to show summary generation
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
        // Debounce rapid updates during uploads
        if (this.updateTimeout) {
            clearTimeout(this.updateTimeout);
        }
        
        this.updateTimeout = setTimeout(() => {
            this.performFileListUpdate();
        }, 100);
    }

    performFileListUpdate() {
        const filesList = document.getElementById('files-list');
        
        // Update global repository count
        const repoCount = document.getElementById('doc-count-repo');
        if (repoCount) {
            repoCount.textContent = this.uploadedFiles.length;
        }
        
        if (this.uploadedFiles.length === 0) {
            filesList.innerHTML = '<div class="no-files">No files uploaded yet</div>';
            return;
        }

        filesList.innerHTML = this.uploadedFiles.map(file => {
            const fileSize = this.formatFileSize(file.size);
            const uploadTime = new Date(file.uploaded_at).toLocaleString();
            const statusClass = file.status === 'completed' ? 'success' : 
                               file.status === 'accepted' ? 'processing' :
                               file.status === 'processing' ? 'processing' : 'error';
            
            // Determine status text and icon
            let statusText, statusIcon, showSummaryButton;
            switch(file.status) {
                case 'completed':
                    statusText = 'Processing complete, summary ready';
                    statusIcon = 'fas fa-check-circle';
                    showSummaryButton = true;
                    break;
                case 'accepted':
                    statusText = 'Processing document & generating summary...';
                    statusIcon = 'fas fa-cog fa-spin';
                    showSummaryButton = false;
                    break;
                case 'processing':
                    statusText = 'Processing document...';
                    statusIcon = 'fas fa-cog fa-spin';
                    showSummaryButton = false;
                    break;
                case 'queued':
                    statusText = 'Queued for processing...';
                    statusIcon = 'fas fa-clock';
                    showSummaryButton = false;
                    break;
                default:
                    statusText = file.status || 'Upload failed';
                    statusIcon = 'fas fa-exclamation-triangle';
                    showSummaryButton = false;
            }
            
            console.log(`File ${file.name} status: "${file.status}", will show summary button: ${file.status === 'completed'}`);
            
            return `
                <div class="file-item" data-job-id="${file.job_id}">
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-meta">${fileSize} • Uploaded ${uploadTime}</div>
                        <div class="file-meta">Job ID: ${file.job_id}</div>
                    </div>
                    <div class="file-actions">
                        <div class="file-status ${statusClass}">
                            <i class="${statusIcon}"></i> ${statusText}
                        </div>
                        ${showSummaryButton ? `
                            <button class="summary-btn" data-job-id="${file.job_id}" data-filename="${file.name}" title="View AI Summary">
                                <i class="fas fa-file-alt"></i>
                            </button>
                        ` : ''}
                        <button class="delete-btn" data-job-id="${file.job_id}" data-filename="${file.name}" title="Delete Document">
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

    async pollDocumentStatus(jobId, attempt) {
        const maxAttempts = 6; // Try for ~30 seconds
        const delays = [2000, 3000, 5000, 5000, 10000, 10000]; // Increasing delays
        
        try {
            const response = await fetch(`/api/v1/documents/${jobId}/status`);
            if (response.ok) {
                const status = await response.json();
                console.log(`Document status check (attempt ${attempt + 1}):`, status);
                
                // Find and update the file in our list
                const fileIndex = this.uploadedFiles.findIndex(file => file.job_id === jobId);
                if (fileIndex !== -1) {
                    const oldStatus = this.uploadedFiles[fileIndex].status;
                    this.uploadedFiles[fileIndex].status = status.status;
                    
                    // If status changed, update the display
                    if (oldStatus !== status.status) {
                        console.log(`Status updated for ${jobId}: ${oldStatus} -> ${status.status}`);
                        this.updateUploadedFilesList();
                        this.saveUploadedFiles();
                        
                        // Check if this was the last document to complete
                        if (status.status === 'completed') {
                            this.checkAllDocumentsCompleted();
                        }
                    }
                    
                    // If still processing and haven't exceeded max attempts, check again
                    if (status.status === 'processing' || status.status === 'queued') {
                        if (attempt < maxAttempts - 1) {
                            setTimeout(() => {
                                this.pollDocumentStatus(jobId, attempt + 1);
                            }, delays[attempt] || 10000);
                        }
                    }
                    // If accepted, also retry (might change to completed)
                    else if (status.status === 'accepted' && attempt < 3) {
                        setTimeout(() => {
                            this.pollDocumentStatus(jobId, attempt + 1);
                        }, delays[attempt] || 5000);
                    }
                }
            }
        } catch (error) {
            console.warn(`Could not check document status (attempt ${attempt + 1}):`, error);
            // Retry on error too
            if (attempt < 3) {
                setTimeout(() => {
                    this.pollDocumentStatus(jobId, attempt + 1);
                }, delays[attempt] || 5000);
            }
        }
    }

    async checkDocumentStatus(jobId) {
        // Legacy method for compatibility - just do a single check
        this.pollDocumentStatus(jobId, 0);
    }

    checkAllDocumentsCompleted() {
        // Check if all recent documents are completed
        const recentDocuments = this.uploadedFiles.filter(file => {
            const uploadTime = new Date(file.uploaded_at);
            const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
            return uploadTime > fiveMinutesAgo;
        });

        const allCompleted = recentDocuments.length > 0 && 
                           recentDocuments.every(file => file.status === 'completed');

        if (allCompleted && recentDocuments.length > 0) {
            this.showCompletionNotification(recentDocuments.length);
        }
    }

    showCompletionNotification(documentCount) {
        // Show a toast notification
        const notification = document.createElement('div');
        notification.className = 'completion-notification';
        notification.innerHTML = `
            <div class="completion-content">
                <i class="fas fa-check-circle"></i>
                <strong>Processing Complete!</strong>
                <p>${documentCount} document${documentCount > 1 ? 's' : ''} processed and ${documentCount > 1 ? 'summaries' : 'summary'} generated.</p>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);

        // Scroll to documents section if not visible
        const filesSection = document.getElementById('upload-section');
        if (filesSection && filesSection.style.display !== 'none') {
            const filesList = document.getElementById('files-list');
            if (filesList) {
                filesList.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
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
        const wsStatus = document.getElementById('ws-status');
        
        if (connected) {
            if (wsStatus) {
                wsStatus.textContent = 'Connected';
                wsStatus.className = 'status-online';
            }
        } else {
            if (wsStatus) {
                wsStatus.textContent = 'Disconnected';
                wsStatus.className = 'status-offline';
            }
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

    initializeUser() {
        // Load saved user or default to first option
        const savedUser = localStorage.getItem('aaire_current_user');
        const userSelector = document.getElementById('user-selector');
        
        if (savedUser && userSelector) {
            userSelector.value = savedUser;
            this.switchUser(savedUser);
        } else if (userSelector) {
            this.switchUser(userSelector.value);
        }
    }

    switchUser(userId = null) {
        const userSelector = document.getElementById('user-selector');
        if (!userId && userSelector) {
            userId = userSelector.value;
        }
        
        if (!userId) return;
        
        const previousUser = this.currentUser?.name;
        
        // Update current user
        this.currentUser = this.getUserInfo(userId);
        
        // Update selector to match
        if (userSelector) {
            userSelector.value = userId;
        }
        
        // Update current user name display
        const currentUserNameElement = document.getElementById('current-user-name');
        if (currentUserNameElement) {
            currentUserNameElement.textContent = this.currentUser.name;
        }
        
        // Save to localStorage
        localStorage.setItem('aaire_current_user', userId);
        
        // Auto-save current chat as session when switching users
        if (previousUser && this.messages.length > 0) {
            console.log(`💾 Auto-saving current chat session for ${previousUser}`);
            this.saveCurrentChatAsSession();
        }
        
        // Always start with empty chat for new sessions
        // Users can load previous sessions from Chat History if needed
        this.initializeEmptyChat(this.currentUser);
        
        // Generate new session ID for this user
        this.sessionId = `${userId}_${Date.now()}`;
        
        // Update welcome message
        this.updateWelcomeMessage();
        
        // Show user switch notification
        if (previousUser && previousUser !== this.currentUser.name) {
            this.showUserSwitchNotification(previousUser, this.currentUser.name);
        }
        
        console.log('Switched to user:', this.currentUser);
    }

    getUserInfo(userId) {
        const users = {
            'bill-accounting': {
                name: 'Bill Johnson',
                department: 'Accounting',
                role: 'Senior Accountant',
                expertise: 'GAAP, Financial Reporting'
            },
            'court-accounting': {
                name: 'Court Williams', 
                department: 'Accounting',
                role: 'Staff Accountant',
                expertise: 'General Ledger, Month-end Close'
            },
            'sarah-actuarial': {
                name: 'Sarah Davis',
                department: 'Actuarial',
                role: 'Actuarial Analyst',
                expertise: 'Reserve Calculations, Risk Assessment'
            },
            'bob-actuarial': {
                name: 'Bob Miller',
                department: 'Actuarial',
                role: 'Senior Actuary',
                expertise: 'Pricing Models, Capital Analysis'
            }
        };
        
        return users[userId] || users['bill-accounting'];
    }

    updateWelcomeMessage() {
        if (!this.currentUser) return;
        
        const messagesContainer = document.getElementById('chat-messages');
        const welcomeMessage = messagesContainer.querySelector('.message');
        
        if (welcomeMessage) {
            const messageContent = welcomeMessage.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = `
                    <strong>Welcome to AAIRE Enterprise, ${this.currentUser.name}!</strong><br>
                    <em>Department: ${this.currentUser.department} | Role: ${this.currentUser.role}</em>
                    <br><br>
                    Your intelligent assistant for insurance accounting and actuarial guidance. I can help you with:
                    <br><br>
                    • US GAAP and IFRS accounting standards<br>
                    • Insurance reserve calculations<br>
                    • Actuarial analysis and modeling<br>
                    • Regulatory compliance questions<br>
                    • Document analysis and insights
                    <br><br>
                    <strong>📚 Accessing Global Repository:</strong> All departments can ask questions about any topic - accounting, actuarial, compliance, or audit matters.
                    <br><br>
                    How can I assist you today?
                `;
            }
        }
    }

    getCurrentUserContext() {
        if (!this.currentUser) return {};
        
        return {
            name: this.currentUser.name,
            department: this.currentUser.department,
            role: this.currentUser.role,
            expertise: this.currentUser.expertise
        };
    }

    saveUserChatHistory(username) {
        // OLD SAVE SYSTEM DISABLED
        // Users now use session-based system via Clear Chat button
        console.log(`⚠️ saveUserChatHistory called but disabled - use session system instead`);
        console.log(`💡 Users should use "Clear Chat" to save sessions manually`);
    }
    
    loadUserChatHistory(userId) {
        // NO LONGER LOAD OLD CHAT HISTORY AUTOMATICALLY
        // Users now start with empty chat and can access sessions from Chat History panel
        const user = this.getUserInfo(userId);
        
        console.log(`🆕 Starting fresh chat session for ${user.name} (old auto-loading disabled)`);
        console.log(`💡 Access previous conversations via Chat History panel`);
        
        // Always start with empty chat
        this.initializeEmptyChat(user);
    }
    
    initializeEmptyChat(user) {
        this.messages = [];
        this.createWelcomeMessage();
        console.log(`🌟 Initialized empty chat for ${user.name}`);
    }
    
    clearProblematicChatHistory() {
        // Clear any saved chat history that might contain false citations
        const users = ['Court_Williams', 'Bill_Smith', 'Sarah_Chen', 'Bob_Johnson', 'Bill_Johnson'];
        users.forEach(username => {
            const storageKey = `aaire_chat_history_${username}`;
            const savedHistory = localStorage.getItem(storageKey);
            if (savedHistory) {
                try {
                    const chatHistory = JSON.parse(savedHistory);
                    const originalLength = chatHistory.messages?.length || 0;
                    
                    if (chatHistory.messages) {
                        // Filter out problematic messages
                        chatHistory.messages = chatHistory.messages.filter(msg => {
                            if (msg.sender === 'assistant' && msg.sources && msg.sources.length > 0) {
                                const content = msg.content.toLowerCase();
                                if (content.includes('how can i assist you today') || 
                                    content.includes('feel free to share') ||
                                    content.includes('hello! how can i assist') ||
                                    content.includes('canadian tax book')) {
                                    return false;
                                }
                            }
                            return true;
                        });
                        
                        if (chatHistory.messages.length !== originalLength) {
                            localStorage.setItem(storageKey, JSON.stringify(chatHistory));
                            console.log(`🧹 Cleaned chat history for ${username}: ${originalLength} → ${chatHistory.messages.length} messages`);
                        }
                    }
                } catch (error) {
                    console.error(`Error cleaning chat history for ${username}:`, error);
                }
            }
        });
    }
    
    // Specifically clear Bill Johnson's problematic data
    clearBillJohnsonData() {
        const storageKey = 'aaire_chat_history_Bill_Johnson';
        console.log('🚮 Completely clearing Bill Johnson chat history due to persistent false citations');
        localStorage.removeItem(storageKey);
        console.log('✅ Bill Johnson chat history cleared');
    }
    
    // Save current chat as a session and clear the active chat
    saveCurrentChatAsSession() {
        if (!this.currentUser || this.messages.length === 0) return;
        
        console.log(`💾 Saving current chat session for ${this.currentUser.name}`);
        
        // Create session data
        const session = {
            id: `session_${Date.now()}`,
            timestamp: new Date().toISOString(),
            messageCount: this.messages.length,
            messages: [...this.messages],
            user: this.currentUser.name
        };
        
        // Get existing chat sessions for this user
        const sessionsKey = `aaire_chat_sessions_${this.currentUser.name.replace(' ', '_')}`;
        const existingSessions = JSON.parse(localStorage.getItem(sessionsKey) || '[]');
        
        // Add new session to the beginning
        existingSessions.unshift(session);
        
        // Keep only last 10 sessions to prevent storage bloat
        if (existingSessions.length > 10) {
            existingSessions.splice(10);
        }
        
        // Save sessions
        localStorage.setItem(sessionsKey, JSON.stringify(existingSessions));
        
        // Clear current chat
        this.messages = [];
        this.initializeEmptyChat(this.currentUser);
        
        console.log(`✅ Chat session saved and current chat cleared for ${this.currentUser.name}`);
        return session.id;
    }
    
    // Load a specific chat session
    loadChatSession(sessionId) {
        if (!this.currentUser) return;
        
        const sessionsKey = `aaire_chat_sessions_${this.currentUser.name.replace(' ', '_')}`;
        const sessions = JSON.parse(localStorage.getItem(sessionsKey) || '[]');
        
        const session = sessions.find(s => s.id === sessionId);
        if (!session) return;
        
        // Save current chat if it has messages
        if (this.messages.length > 0) {
            this.saveCurrentChatAsSession();
        }
        
        // Load session messages
        this.messages = [...session.messages];
        
        // Clear and rebuild UI
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = '';
        
        // Create welcome message first
        this.createWelcomeMessage();
        
        // Add session messages
        this.messages.forEach(msg => {
            this.addMessageToUI(msg.sender, msg.content, msg.sources, msg.followUpQuestions);
        });
        
        console.log(`📖 Loaded chat session: ${session.id} (${session.messageCount} messages)`);
    }
    
    createWelcomeMessage() {
        if (!this.currentUser) return;
        
        const messagesContainer = document.getElementById('chat-messages');
        
        // Clear any existing messages first to prevent duplicates
        messagesContainer.innerHTML = '';
        
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message assistant welcome-message';
        
        // Use proper HTML structure with paragraph tags for better formatting
        welcomeDiv.innerHTML = `
            <div class="message-content">
                <p><strong>Welcome to AAIRE Enterprise, ${this.currentUser.name}!</strong><br>
                <em>Department: ${this.currentUser.department} | Role: ${this.currentUser.role}</em></p>
                
                <p>Your intelligent assistant for insurance accounting and actuarial guidance. I can help you with:</p>
                
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>US GAAP and IFRS accounting standards</li>
                    <li>Insurance reserve calculations</li>
                    <li>Actuarial analysis and modeling</li>
                    <li>Regulatory compliance questions</li>
                    <li>Document analysis and insights</li>
                </ul>
                
                <p><strong>📚 Accessing Global Repository:</strong> All departments can ask questions about any topic - accounting, actuarial, compliance, or audit matters.</p>
                
                <p><strong>How can I assist you today?</strong></p>
                
                <div class="message-meta">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        messagesContainer.appendChild(welcomeDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        console.log(`✅ Created clean welcome message for ${this.currentUser.name}`);
    }
    
    addMessageToUI(sender, content, sources = null, followUpQuestions = null) {
        // This is the UI-only version of addMessage that doesn't modify this.messages
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
            const sourceList = sources.map(source => {
                if (typeof source === 'string') {
                    return source;
                } else if (source && source.source) {
                    return source.source;
                } else {
                    return 'Unknown Source';
                }
            });
            
            const uniqueSources = [...new Set(sourceList)];
            messageContent += '<br><br><small><strong>Source:</strong><br>';
            uniqueSources.forEach(source => {
                messageContent += `• ${source}<br>`;
            });
            messageContent += '</small>';
        }
        
        // Add follow-up questions for assistant messages
        if (sender === 'assistant' && followUpQuestions && followUpQuestions.length > 0) {
            messageContent += '<div class="follow-up-questions">';
            messageContent += '<div class="follow-up-title">💡 Suggested follow-up questions:</div>';
            followUpQuestions.forEach((question, index) => {
                messageContent += `
                    <button class="follow-up-btn" onclick="askFollowUpQuestion('${question.replace(/'/g, "\\'")}')">
                        ${question}
                    </button>
                `;
            });
            messageContent += '</div>';
        }
        
        // Add feedback buttons for assistant messages
        if (sender === 'assistant') {
            const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            messageContent += `
                <div class="response-feedback">
                    <span class="feedback-label">Was this helpful?</span>
                    <button class="feedback-btn thumbs-up" onclick="submitFeedback('${messageId}', 'thumbs_up')" title="Helpful">
                        👍
                    </button>
                    <button class="feedback-btn thumbs-down" onclick="submitFeedback('${messageId}', 'thumbs_down')" title="Not helpful">
                        👎
                    </button>
                    <button class="feedback-btn report-issue" onclick="reportIssue('${messageId}')" title="Report issue">
                        ⚠️
                    </button>
                </div>
            `;
        }

        // Add copy button for assistant messages
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
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Setup copy functionality
        if (sender === 'assistant') {
            const copyBtn = messageDiv.querySelector('.copy-btn');
            if (copyBtn) {
                copyBtn.addEventListener('click', () => {
                    navigator.clipboard.writeText(content).then(() => {
                        copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                        setTimeout(() => {
                            copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                        }, 2000);
                    });
                });
            }
        }
    }

    showUserSwitchNotification(previousUser, newUser) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;
        
        // Create a system notification message
        const notificationDiv = document.createElement('div');
        notificationDiv.className = 'message system-message';
        notificationDiv.innerHTML = `
            <div class="message-content system-notification">
                <i class="fas fa-user-friends"></i>
                <strong>User switched:</strong> ${previousUser} → ${newUser}
                <br><small>Chat history cleared. New session started.</small>
            </div>
        `;
        
        // Insert after welcome message
        const welcomeMessage = messagesContainer.querySelector('.message');
        if (welcomeMessage && welcomeMessage.nextSibling) {
            messagesContainer.insertBefore(notificationDiv, welcomeMessage.nextSibling);
        } else {
            messagesContainer.appendChild(notificationDiv);
        }
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notificationDiv.parentNode) {
                notificationDiv.remove();
            }
        }, 5000);
        
        // Scroll to show notification
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    getConversationHistory() {
        // Return last 20 messages (10 exchanges) for better memory retention
        const recentMessages = this.messages.slice(-20);
        return recentMessages.map(msg => ({
            sender: msg.sender,
            content: msg.content,
            timestamp: msg.timestamp
        }));
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
    if (!window.app || !window.app.currentUser) return;
    
    if (window.app.messages.length === 0) {
        alert('No messages to clear in current chat.');
        return;
    }
    
    if (confirm('Clear current chat? This will save the conversation to your chat history.')) {
        window.app.saveCurrentChatAsSession();
        console.log('✅ Chat cleared and saved to history');
    }
}

// Chat History Management Functions
function toggleChatHistoryPanel() {
    const panel = document.getElementById('chat-history-panel');
    if (panel.style.display === 'none' || panel.style.display === '') {
        panel.style.display = 'block';
        updateChatHistoryPanel();
    } else {
        panel.style.display = 'none';
    }
}

function updateChatHistoryPanel() {
    if (!window.app || !window.app.currentUser) return;
    
    // Update current user info
    document.getElementById('history-current-user').textContent = window.app.currentUser.name;
    
    // Update current user's sessions
    updateCurrentUserSessions();
    
    // Update all users summary
    updateAllUsersSummary();
}

function updateCurrentUserSessions() {
    const sessionsContainer = document.getElementById('current-user-sessions');
    const username = window.app.currentUser.name.replace(' ', '_');
    const sessionsKey = `aaire_chat_sessions_${username}`;
    const sessions = JSON.parse(localStorage.getItem(sessionsKey) || '[]');
    
    sessionsContainer.innerHTML = '';
    
    if (sessions.length === 0) {
        sessionsContainer.innerHTML = '<div class="text-muted">No chat sessions saved yet</div>';
        return;
    }
    
    sessions.forEach(session => {
        const date = new Date(session.timestamp);
        const preview = getSessionPreview(session.messages);
        
        const tile = document.createElement('div');
        tile.className = 'session-tile';
        tile.innerHTML = `
            <div class="session-header">
                <div class="session-date">${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                <div class="session-stats">${session.messageCount} messages</div>
            </div>
            <div class="session-preview">${preview}</div>
            <div class="session-actions">
                <button class="btn btn-sm btn-primary" onclick="loadSession('${session.id}')" title="Load this session">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn btn-sm btn-secondary" onclick="exportSession('${session.id}')" title="Export this session">
                    <i class="fas fa-download"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="deleteSession('${session.id}')" title="Delete this session">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        sessionsContainer.appendChild(tile);
    });
}

function getSessionPreview(messages) {
    if (!messages || messages.length === 0) return 'Empty session';
    
    // Find first user message
    const userMessage = messages.find(msg => msg.sender === 'user');
    if (userMessage) {
        return userMessage.content.substring(0, 80) + (userMessage.content.length > 80 ? '...' : '');
    }
    
    return 'Chat session';
}

function updateAllUsersSummary() {
    const summaryContainer = document.getElementById('all-users-summary');
    const users = ['Court_Williams', 'Bill_Smith', 'Sarah_Chen', 'Bob_Johnson', 'Bill_Johnson'];
    
    summaryContainer.innerHTML = '';
    
    users.forEach(username => {
        const sessionsKey = `aaire_chat_sessions_${username}`;
        const sessions = JSON.parse(localStorage.getItem(sessionsKey) || '[]');
        
        if (sessions.length > 0) {
            const totalMessages = sessions.reduce((sum, session) => sum + session.messageCount, 0);
            const storageSize = localStorage.getItem(sessionsKey)?.length || 0;
            
            const userTile = document.createElement('div');
            userTile.className = 'user-summary-tile';
            userTile.innerHTML = `
                <div class="user-summary-info">
                    <div class="user-summary-name">${username.replace('_', ' ')}</div>
                    <div class="user-summary-stats">${sessions.length} sessions | ${totalMessages} total messages | ${Math.round(storageSize / 1024)} KB</div>
                </div>
                <div class="user-summary-actions">
                    <button class="btn btn-sm btn-primary" onclick="exportAllUserSessions('${username}')" title="Export all sessions">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="deleteAllUserSessions('${username}')" title="Delete all sessions">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            summaryContainer.appendChild(userTile);
        }
    });
    
    if (summaryContainer.innerHTML === '') {
        summaryContainer.innerHTML = '<div class="text-muted">No chat sessions found for any users</div>';
    }
}

// Session management functions
function loadSession(sessionId) {
    if (window.app) {
        window.app.loadChatSession(sessionId);
        toggleChatHistoryPanel(); // Close panel after loading
    }
}

function exportSession(sessionId) {
    const username = window.app.currentUser.name.replace(' ', '_');
    const sessionsKey = `aaire_chat_sessions_${username}`;
    const sessions = JSON.parse(localStorage.getItem(sessionsKey) || '[]');
    const session = sessions.find(s => s.id === sessionId);
    
    if (!session) {
        alert('Session not found');
        return;
    }
    
    const exportData = {
        session_id: session.id,
        user: session.user,
        timestamp: session.timestamp,
        message_count: session.messageCount,
        messages: session.messages
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aaire_session_${session.user.replace(' ', '_')}_${new Date(session.timestamp).toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function deleteSession(sessionId) {
    if (!confirm('Delete this chat session? This cannot be undone.')) return;
    
    const username = window.app.currentUser.name.replace(' ', '_');
    const sessionsKey = `aaire_chat_sessions_${username}`;
    const sessions = JSON.parse(localStorage.getItem(sessionsKey) || '[]');
    
    const updatedSessions = sessions.filter(s => s.id !== sessionId);
    localStorage.setItem(sessionsKey, JSON.stringify(updatedSessions));
    
    updateChatHistoryPanel();
    console.log(`✅ Deleted session: ${sessionId}`);
}

function exportAllUserSessions(username) {
    const sessionsKey = `aaire_chat_sessions_${username}`;
    const sessions = JSON.parse(localStorage.getItem(sessionsKey) || '[]');
    
    if (sessions.length === 0) {
        alert('No chat sessions found for this user');
        return;
    }
    
    const exportData = {
        user: username.replace('_', ' '),
        exported_at: new Date().toISOString(),
        total_sessions: sessions.length,
        sessions: sessions
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aaire_all_sessions_${username}_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function deleteAllUserSessions(username) {
    if (!confirm(`Delete ALL chat sessions for ${username.replace('_', ' ')}? This cannot be undone!`)) return;
    
    const sessionsKey = `aaire_chat_sessions_${username}`;
    localStorage.removeItem(sessionsKey);
    
    updateChatHistoryPanel();
    console.log(`✅ Deleted all sessions for ${username}`);
}

function clearAllChatHistory() {
    if (!confirm('Delete ALL chat sessions for ALL users? This cannot be undone!')) return;
    
    const users = ['Court_Williams', 'Bill_Smith', 'Sarah_Chen', 'Bob_Johnson', 'Bill_Johnson'];
    users.forEach(username => {
        const sessionsKey = `aaire_chat_sessions_${username}`;
        localStorage.removeItem(sessionsKey);
    });
    
    // Clear current user's in-memory messages too
    if (window.app) {
        window.app.messages = [];
        window.app.initializeEmptyChat(window.app.currentUser);
    }
    
    updateChatHistoryPanel();
    console.log('✅ Deleted all chat sessions for all users');
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

// Global functions for HTML onclick handlers
function sendMessage() {
    console.log('🔥 Global sendMessage called');
    console.log('🔍 Checking window.app:', window.app ? 'EXISTS' : 'UNDEFINED');
    
    if (window.app) {
        console.log('✅ App exists, calling app.sendMessage()');
        try {
            window.app.sendMessage();
            console.log('✅ app.sendMessage() completed');
        } catch (error) {
            console.error('❌ Error in app.sendMessage():', error);
            alert('Send button error: ' + error.message);
        }
    } else {
        console.error('❌ App not initialized - DOM may not be ready');
        alert('App not ready yet - please wait a moment and try again');
        
        // Try to initialize if not done yet
        if (typeof AAIREApp !== 'undefined') {
            console.log('🔄 Attempting to initialize app...');
            try {
                window.app = new AAIREApp();
                console.log('✅ App initialized, retrying sendMessage');
                window.app.sendMessage();
            } catch (error) {
                console.error('❌ Failed to initialize app:', error);
            }
        }
    }
}

function askFollowUpQuestion(question) {
    if (window.app) {
        // Set the question in the input field
        const input = document.getElementById('chat-input');
        if (input) {
            input.value = question;
            // Trigger the send
            window.app.sendMessage();
        }
    }
}

function switchUser() {
    if (window.app) {
        window.app.switchUser();
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

// Feedback functions
function submitFeedback(messageId, feedbackType) {
    console.log('Feedback submitted:', messageId, feedbackType);
    
    // Visual feedback - highlight the selected button
    const messageElement = document.querySelector(`[onclick*="${messageId}"]`).closest('.message');
    const buttons = messageElement.querySelectorAll('.feedback-btn');
    buttons.forEach(btn => btn.classList.remove('selected'));
    
    const clickedButton = event.target;
    clickedButton.classList.add('selected');
    
    // Send feedback to backend
    fetch('/api/v1/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message_id: messageId,
            feedback_type: feedbackType,
            timestamp: new Date().toISOString(),
            user_id: window.app ? window.app.currentUser.id : 'demo-user',
            session_id: window.app ? window.app.sessionId : 'unknown'
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Feedback recorded:', data);
        // Show brief confirmation
        clickedButton.style.transform = 'scale(1.2)';
        setTimeout(() => {
            clickedButton.style.transform = 'scale(1)';
        }, 200);
    })
    .catch(error => {
        console.error('Error submitting feedback:', error);
    });
}

function reportIssue(messageId) {
    const issueType = prompt('What type of issue would you like to report?\n\n1. Incorrect information\n2. Missing information\n3. Irrelevant response\n4. Other\n\nEnter 1-4 or describe the issue:');
    
    if (issueType) {
        const issueMapping = {
            '1': 'incorrect_information',
            '2': 'missing_information', 
            '3': 'irrelevant_response',
            '4': 'other'
        };
        
        const issueTypeCode = issueMapping[issueType] || 'other';
        const issueDescription = issueType.length > 1 ? issueType : '';
        
        fetch('/api/v1/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message_id: messageId,
                feedback_type: 'issue_report',
                issue_type: issueTypeCode,
                issue_description: issueDescription,
                timestamp: new Date().toISOString(),
                user_id: window.app ? window.app.currentUser.id : 'demo-user',
                session_id: window.app ? window.app.sessionId : 'unknown'
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Issue reported:', data);
            alert('Thank you for reporting this issue. We will review it and improve our responses.');
        })
        .catch(error => {
            console.error('Error reporting issue:', error);
            alert('Error submitting issue report. Please try again.');
        });
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new AAIREApp();
    // Make app globally accessible for debugging
    window.app = app;
});