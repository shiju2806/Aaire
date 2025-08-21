# AAIRE Browser Extension

A Chrome extension that seamlessly captures documents from SOA (Society of Actuaries) publications and sends them to AAIRE for AI-powered analysis and Q&A.

## Features

### 🔍 Document Detection
- Automatically detects PDF and document links on SOA websites
- Identifies embedded documents and publication downloads
- Real-time scanning as you browse SOA content

### 📤 One-Click Upload
- Send documents directly to AAIRE from your browser
- Maintains context about source URL and page title
- Background processing with status tracking

### ❓ Integrated Q&A
- Query uploaded documents directly from the extension
- Full AAIRE AI capabilities in a convenient popup
- Access to all your uploaded documents

### 🔒 Privacy & Security
- Documents are processed securely through AAIRE's existing infrastructure
- No credentials stored in the extension
- Works within your existing SOA session

## Installation

### Development Installation
1. Clone the AAIRE repository
2. Navigate to the `browser-extension` directory
3. Open Chrome and go to `chrome://extensions/`
4. Enable "Developer mode"
5. Click "Load unpacked" and select the `browser-extension` folder

### Production Installation
*(Coming soon - will be available in Chrome Web Store)*

## Usage

### Step 1: Navigate to SOA
Visit any SOA publication page:
- https://publications.soa.org
- https://www.soa.org

### Step 2: Detect Documents
The extension automatically scans the page for documents. You can also:
- Click the AAIRE extension icon in your browser
- Use the "Scan Page" button to refresh detection

### Step 3: Send to AAIRE
- Select the documents you want to analyze
- Click "Send to AAIRE"
- Documents are uploaded and processed automatically

### Step 4: Ask Questions
- Switch to the "Query" tab in the extension popup
- Select a processed document
- Ask questions about the content
- Get instant AI-powered answers

## Supported Websites

Currently supports:
- publications.soa.org (SOA Publications)
- www.soa.org (Main SOA website)

Additional sites can be added by updating the manifest permissions.

## Architecture

### Components
- **Content Script**: Runs on SOA pages, detects documents
- **Background Script**: Handles communication with AAIRE backend
- **Popup Interface**: User interface for managing documents and queries
- **Backend API**: New extension-specific endpoints in AAIRE

### API Integration
The extension uses dedicated API endpoints:
- `POST /api/v1/extension/upload` - Upload documents
- `GET /api/v1/extension/status/{job_id}` - Check processing status
- `POST /api/v1/extension/query` - Query documents
- `GET /api/v1/extension/documents` - List uploaded documents

### Security Features
- Isolated from main AAIRE functionality (no regression risk)
- Secure file upload with metadata tracking
- Extension-specific job tracking and storage

## Development

### File Structure
```
browser-extension/
├── manifest.json          # Extension configuration
├── background.js          # Service worker
├── content.js            # Page content detection
├── content.css           # Content script styles
├── popup.html            # Extension popup interface
├── popup.js              # Popup functionality
├── popup.css             # Popup styles
├── icons/                # Extension icons
└── README.md             # This file
```

### Key Technologies
- **Manifest V3**: Latest Chrome extension format
- **Content Scripts**: For document detection on SOA pages
- **Service Worker**: Background processing and API communication
- **Chrome Storage API**: Persistent document tracking
- **Fetch API**: Communication with AAIRE backend

### Testing
1. Load the extension in development mode
2. Navigate to a SOA publication page
3. Verify document detection in the extension popup
4. Test document upload and processing
5. Test querying functionality

## Troubleshooting

### Extension Not Working
- Ensure you're on a supported SOA website
- Check that AAIRE backend is running at https://aaire.xyz
- Verify extension permissions in Chrome settings

### Documents Not Detected
- Try the "Scan Page" button to refresh detection
- Some documents may be behind authentication
- Check browser console for error messages

### Upload Failures
- Verify AAIRE backend is accessible
- Check network connectivity
- Large documents may take time to process

## Contributing

This extension is part of the AAIRE project. To contribute:
1. Follow the main AAIRE development guidelines
2. Test changes thoroughly on various SOA pages
3. Ensure no regression to main AAIRE functionality
4. Update documentation as needed

## License

Part of the AAIRE project - see main project license.