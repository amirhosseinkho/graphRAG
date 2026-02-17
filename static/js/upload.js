// JavaScript for Upload Graph Page

// تابع بررسی معتبر بودن URL
function isValidUrl(string) {
    try {
        const url = new URL(string);
        return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (_) {
        return false;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('graph-file');
    const fileInfo = document.getElementById('file-info');
    const uploadBtn = document.getElementById('upload-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const progressSection = document.getElementById('progress-section');
    const resultSection = document.getElementById('result-section');
    const successMessage = document.getElementById('success-message');
    const errorMessage = document.getElementById('error-message');

    let selectedFile = null;

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Upload button click
    uploadBtn.addEventListener('click', function() {
        if (selectedFile) {
            uploadFile(selectedFile);
        }
    });

    // Cancel button click
    cancelBtn.addEventListener('click', function() {
        resetUpload();
    });

    function handleFileSelect(file) {
        // Check file size (100MB limit)
        if (file.size > 100 * 1024 * 1024) {
            showError('اندازه فایل بیش از 100 مگابایت است');
            return;
        }

        // Check file type
        const allowedTypes = ['.pkl', '.sif', '.tsv', '.csv', '.txt', '.gz'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            showError('نوع فایل مجاز نیست. فایل‌های مجاز: pkl, sif, tsv, csv, txt, gz');
            return;
        }

        selectedFile = file;
        displayFileInfo(file);
        showUploadActions();
    }

    function displayFileInfo(file) {
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        const fileType = document.getElementById('file-type');

        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileType.textContent = file.type || 'نوع نامشخص';

        fileInfo.style.display = 'block';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function showUploadActions() {
        uploadBtn.style.display = 'inline-block';
        cancelBtn.style.display = 'inline-block';
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('graph_file', file);

        // Show progress
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        uploadBtn.disabled = true;

        // Simulate progress (since we can't track actual upload progress easily)
        let progress = 0;
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
            progressText.textContent = `آپلود فایل... ${Math.round(progress)}%`;
        }, 200);

        fetch('/api/upload_graph', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(progressInterval);
            progressFill.style.width = '100%';
            progressText.textContent = 'تکمیل شد!';

            setTimeout(() => {
                progressSection.style.display = 'none';
                
                if (data.success) {
                    showSuccess(data.message);
                } else {
                    showError(data.error);
                }
            }, 500);
        })
        .catch(error => {
            clearInterval(progressInterval);
            progressSection.style.display = 'none';
            showError('خطا در آپلود فایل: ' + error.message);
        });
    }

    function showSuccess(message) {
        document.getElementById('success-text').textContent = message;
        successMessage.style.display = 'block';
        errorMessage.style.display = 'none';
        resultSection.style.display = 'block';
    }

    function showError(message) {
        document.getElementById('error-text').textContent = message;
        errorMessage.style.display = 'block';
        successMessage.style.display = 'none';
        resultSection.style.display = 'block';
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
        uploadBtn.style.display = 'none';
        cancelBtn.style.display = 'none';
        progressSection.style.display = 'none';
        resultSection.style.display = 'none';
        uploadBtn.disabled = false;
    }

    // Global function for reset
    window.resetUpload = resetUpload;

    // ========== Text to Graph Functionality ==========
    
    // Tab switching
    window.switchTab = function(tab) {
        const fileTab = document.getElementById('tab-file');
        const textTab = document.getElementById('tab-text');
        const fileSection = document.getElementById('section-file');
        const textSection = document.getElementById('section-text');
        
        if (tab === 'file') {
            fileTab.classList.add('active');
            textTab.classList.remove('active');
            fileSection.style.display = 'block';
            textSection.style.display = 'none';
        } else {
            textTab.classList.add('active');
            fileTab.classList.remove('active');
            textSection.style.display = 'block';
            fileSection.style.display = 'none';
        }
    };

    // Text input elements
    const textInput = document.getElementById('text-input');
    const textLength = document.getElementById('text-length');
    const extractionMethod = document.getElementById('extraction-method');
    const llmSettings = document.getElementById('llm-settings');
    const confidenceSetting = document.getElementById('confidence-setting');
    const llmModel = document.getElementById('llm-model');
    const maxEntities = document.getElementById('max-entities');
    const maxRelationships = document.getElementById('max-relationships');
    const confidenceThreshold = document.getElementById('confidence-threshold');
    const buildGraphBtn = document.getElementById('build-graph-btn');
    const clearTextBtn = document.getElementById('clear-text-btn');
    const textProgressSection = document.getElementById('text-progress-section');
    const textResultSection = document.getElementById('text-result-section');
    const textSuccessMessage = document.getElementById('text-success-message');
    const textErrorMessage = document.getElementById('text-error-message');
    const textProgressFill = document.getElementById('text-progress-fill');
    const textProgressText = document.getElementById('text-progress-text');

    // Update text length counter
    if (textInput && textLength) {
        textInput.addEventListener('input', function() {
            textLength.textContent = this.value.length;
        });
    }

    // Get additional UI elements
    const multipassSettings = document.getElementById('multipass-settings');
    const hybridSettings = document.getElementById('hybrid-settings');
    
    // Show/hide settings based on extraction method
    if (extractionMethod) {
        extractionMethod.addEventListener('change', function() {
            const method = this.value;
            
            // Show/hide method-specific settings
            const llmSettings = document.getElementById('llm-settings');
            const multipassSettings = document.getElementById('multipass-settings');
            const hybridSettings = document.getElementById('hybrid-settings');
            const persianSettings = document.getElementById('persian-settings');
            const spanBasedSettings = document.getElementById('span-based-settings');
            const longTextSettings = document.getElementById('long-text-settings');
            const confidenceSetting = document.getElementById('confidence-setting');
            const jointErSettings = document.getElementById('joint-er-settings');
            const autoregressiveSettings = document.getElementById('autoregressive-settings');
            const edcSettings = document.getElementById('edc-settings');
            const incrementalSettings = document.getElementById('incremental-settings');
            
            // Hide all settings first
            if (llmSettings) llmSettings.style.display = 'none';
            if (multipassSettings) multipassSettings.style.display = 'none';
            if (hybridSettings) hybridSettings.style.display = 'none';
            if (persianSettings) persianSettings.style.display = 'none';
            if (spanBasedSettings) spanBasedSettings.style.display = 'none';
            if (longTextSettings) longTextSettings.style.display = 'none';
            if (confidenceSetting) confidenceSetting.style.display = 'none';
            if (jointErSettings) jointErSettings.style.display = 'none';
            if (autoregressiveSettings) autoregressiveSettings.style.display = 'none';
            if (edcSettings) edcSettings.style.display = 'none';
            if (incrementalSettings) incrementalSettings.style.display = 'none';
            
            // Show relevant settings based on method
            if (method === 'llm' || method === 'llm_multipass' || method === 'autoregressive' || method === 'edc') {
                if (llmSettings) llmSettings.style.display = 'block';
                if (confidenceSetting) confidenceSetting.style.display = 'block';
            }
            
            if (method === 'llm_multipass') {
                if (multipassSettings) multipassSettings.style.display = 'block';
            }
            
            if (method === 'hybrid') {
                if (hybridSettings) hybridSettings.style.display = 'block';
                if (confidenceSetting) confidenceSetting.style.display = 'block';
            }
            
            if (method === 'persian') {
                if (persianSettings) persianSettings.style.display = 'block';
            }
            
            if (method === 'span_based') {
                if (spanBasedSettings) spanBasedSettings.style.display = 'block';
            }
            
            if (method === 'long_text') {
                if (longTextSettings) longTextSettings.style.display = 'block';
            }
            
            if (method === 'joint_er') {
                if (jointErSettings) jointErSettings.style.display = 'block';
            }
            
            if (method === 'autoregressive') {
                if (autoregressiveSettings) autoregressiveSettings.style.display = 'block';
            }
            
            if (method === 'edc') {
                if (edcSettings) edcSettings.style.display = 'block';
            }
            
            if (method === 'incremental') {
                if (incrementalSettings) incrementalSettings.style.display = 'block';
            }
        });
        
        // Trigger change event to set initial state
        extractionMethod.dispatchEvent(new Event('change'));
    }
    
    // Toggle advanced settings
    window.toggleAdvancedSettings = function() {
        const content = document.getElementById('advanced-settings-content');
        const btn = event.target.closest('button');
        if (content) {
            if (content.style.display === 'none') {
                content.style.display = 'block';
                if (btn) btn.querySelector('i').classList.replace('fa-chevron-down', 'fa-chevron-up');
            } else {
                content.style.display = 'none';
                if (btn) btn.querySelector('i').classList.replace('fa-chevron-up', 'fa-chevron-down');
            }
        }
    };

    // Clear text button
    if (clearTextBtn) {
        clearTextBtn.addEventListener('click', function() {
            if (textInput) {
                textInput.value = '';
                textLength.textContent = '0';
            }
            const urlInput = document.getElementById('url-input');
            if (urlInput) {
                urlInput.value = '';
            }
        });
    }
    
    // URL input change handler
    const urlInput = document.getElementById('url-input');
    if (urlInput) {
        urlInput.addEventListener('input', function() {
            // اگر URL وارد شد، textarea را غیرفعال کن
            if (this.value.trim()) {
                if (textInput) {
                    textInput.disabled = true;
                    textInput.placeholder = 'برای استفاده از URL، ابتدا فیلد URL را پاک کنید';
                }
            } else {
                if (textInput) {
                    textInput.disabled = false;
                    textInput.placeholder = 'متن خود را اینجا وارد کنید یا paste کنید...\n\nمثال:\nTP53 participates in apoptosis and interacts with BRCA1. Trastuzumab treats breast cancer.';
                }
            }
        });
    }
    
    // Text input change handler - اگر متن وارد شد، URL را غیرفعال کن
    if (textInput) {
        textInput.addEventListener('input', function() {
            const urlInput = document.getElementById('url-input');
            if (this.value.trim()) {
                if (urlInput) {
                    urlInput.disabled = true;
                }
            } else {
                if (urlInput) {
                    urlInput.disabled = false;
                }
            }
        });
    }

    // Build graph button
    if (buildGraphBtn) {
        buildGraphBtn.addEventListener('click', function() {
            buildGraphFromText();
        });
    }

    function buildGraphFromText() {
        const urlInput = document.getElementById('url-input');
        const url = urlInput ? urlInput.value.trim() : '';
        const text = textInput ? textInput.value.trim() : '';
        
        // بررسی اینکه حداقل یکی از text یا URL وارد شده باشد
        if (!text && !url) {
            showTextError('لطفاً متن یا URL را وارد کنید');
            return;
        }
        
        // اگر URL وارد شده، بررسی معتبر بودن آن
        if (url && !isValidUrl(url)) {
            showTextError('لطفاً یک URL معتبر وارد کنید (مثال: https://example.com)');
            return;
        }

        const method = extractionMethod ? extractionMethod.value : 'simple';
        const maxEntitiesValue = maxEntities ? parseInt(maxEntities.value) : 100;
        const maxRelationshipsValue = maxRelationships ? parseInt(maxRelationships.value) : 200;
        
        // Prepare request data
        const requestData = {
            method: method,
            max_entities: maxEntitiesValue,
            max_relationships: maxRelationshipsValue
        };
        
        // اضافه کردن text یا url
        if (url) {
            requestData.url = url;
            
            // اضافه کردن تنظیم استخراج ویکی‌پدیا
            const useWikipediaExtraction = document.getElementById('use-wikipedia-extraction');
            if (useWikipediaExtraction) {
                requestData.use_wikipedia_extraction = useWikipediaExtraction.checked;
            }
        } else {
            requestData.text = text;
        }
        
        // New parameters for Persian and advanced features
        const languageSelect = document.getElementById('language-select');
        if (languageSelect) {
            requestData.language = languageSelect.value || 'auto';
        }
        
        const enableCoreference = document.getElementById('enable-coreference');
        if (enableCoreference) {
            requestData.enable_coreference = enableCoreference.checked;
        }
        
        const chunkingStrategy = document.getElementById('chunking-strategy');
        if (chunkingStrategy) {
            requestData.chunking_strategy = chunkingStrategy.value || 'smart';
        }
        
        const chunkOverlap = document.getElementById('chunk-overlap');
        if (chunkOverlap) {
            requestData.chunk_overlap = parseFloat(chunkOverlap.value) || 0.2;
        }
        
        const maxTokens = document.getElementById('max-tokens');
        if (maxTokens) {
            requestData.max_tokens = parseInt(maxTokens.value) || 512;
        }
        
        const spanModelType = document.getElementById('span-model-type');
        if (spanModelType) {
            requestData.span_model_type = spanModelType.value || 'biobert';
        }
        
        // Add new method-specific parameters
        const structureIterations = document.getElementById('structure-iterations');
        if (structureIterations && method === 'joint_er') {
            requestData.structure_iterations = parseInt(structureIterations.value) || 3;
        }
        
        const maxGenerationLength = document.getElementById('max-generation-length');
        if (maxGenerationLength && method === 'autoregressive') {
            requestData.max_generation_length = parseInt(maxGenerationLength.value) || 2048;
        }
        
        const useRag = document.getElementById('use-rag');
        if (useRag && method === 'edc') {
            requestData.use_rag = useRag.checked;
        }
        
        const chunkSize = document.getElementById('chunk-size');
        if (chunkSize && method === 'incremental') {
            requestData.chunk_size = parseInt(chunkSize.value) || 500;
        }
        
        const overlapSize = document.getElementById('overlap-size');
        if (overlapSize && method === 'incremental') {
            requestData.overlap = parseInt(overlapSize.value) || 100;
        }
        
        const incrementalBaseMethod = document.getElementById('incremental-base-method');
        if (incrementalBaseMethod && method === 'incremental') {
            requestData.base_method = incrementalBaseMethod.value || 'spacy';
        }

        // Add LLM-specific parameters
        if (method === 'llm' || method === 'llm_multipass' || method === 'autoregressive' || method === 'edc') {
            requestData.llm_model = llmModel ? llmModel.value : 'mistralai/Mistral-7B-Instruct-v0.2';
            requestData.confidence_threshold = confidenceThreshold ? parseFloat(confidenceThreshold.value) : 0.5;
            
            // Add HuggingFace token if provided
            const hfToken = document.getElementById('hf-token');
            if (hfToken && hfToken.value) {
                requestData.hf_token = hfToken.value;
            }
        }
        
        // Add multipass-specific parameters
        if (method === 'llm_multipass') {
            const maxGleanings = document.getElementById('max-gleanings');
            requestData.max_gleanings = maxGleanings ? parseInt(maxGleanings.value) : 2;
        }
        
        // Add hybrid-specific parameters
        if (method === 'hybrid') {
            const hybridCheckboxes = document.querySelectorAll('#hybrid-settings input[type="checkbox"]:checked');
            requestData.hybrid_methods = Array.from(hybridCheckboxes).map(cb => cb.value);
            if (requestData.hybrid_methods.length === 0) {
                requestData.hybrid_methods = ['spacy', 'llm']; // Default
            }
        }
        
        // Add preprocessing parameter
        const enablePreprocessing = document.getElementById('enable-preprocessing');
        if (enablePreprocessing) {
            requestData.enable_preprocessing = enablePreprocessing.checked;
        }
        
        // Add advanced settings
        const enableEntityResolution = document.getElementById('enable-entity-resolution');
        const enableRelationshipWeighting = document.getElementById('enable-relationship-weighting');
        const minRelationshipWeight = document.getElementById('min-relationship-weight');
        const removeIsolatedNodes = document.getElementById('remove-isolated-nodes');
        
        requestData.enable_entity_resolution = enableEntityResolution ? enableEntityResolution.checked : true;
        requestData.enable_relationship_weighting = enableRelationshipWeighting ? enableRelationshipWeighting.checked : true;
        requestData.min_relationship_weight = minRelationshipWeight ? parseFloat(minRelationshipWeight.value) : 0.0;
        requestData.remove_isolated_nodes = removeIsolatedNodes ? removeIsolatedNodes.checked : false;

        // Show progress
        textProgressSection.style.display = 'block';
        textResultSection.style.display = 'none';
        buildGraphBtn.disabled = true;

        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 85) progress = 85;
            textProgressFill.style.width = progress + '%';
            
            if (progress < 20) {
                textProgressText.textContent = url ? 'در حال دریافت محتوای URL...' : 'در حال پردازش متن...';
            } else if (progress < 40) {
                textProgressText.textContent = 'در حال استخراج موجودیت‌ها...';
            } else if (progress < 70) {
                textProgressText.textContent = 'در حال شناسایی روابط...';
            } else {
                textProgressText.textContent = 'در حال ساخت گراف...';
            }
        }, 300);

        // Send request to API
        fetch('/api/text_to_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            // Check if response is ok
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || `خطای HTTP ${response.status}: ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            clearInterval(progressInterval);
            textProgressFill.style.width = '100%';
            textProgressText.textContent = 'تکمیل شد!';

            setTimeout(() => {
                textProgressSection.style.display = 'none';
                
                if (data.success) {
                    showTextSuccess(data);
                } else {
                    // Show detailed error message
                    let errorMsg = data.error || 'خطا در ساخت گراف';
                    
                    // If error contains details, show them
                    if (data.details) {
                        errorMsg += '\n\nجزئیات فنی:\n' + data.details;
                    }
                    
                    // If error type is provided, add it
                    if (data.error_type) {
                        errorMsg = `[${data.error_type}] ${errorMsg}`;
                    }
                    
                    showTextError(errorMsg);
                }
            }, 500);
        })
        .catch(error => {
            clearInterval(progressInterval);
            textProgressSection.style.display = 'none';
            
            // Better error handling
            let errorMsg = 'خطا در ارتباط با سرور';
            
            // بررسی نوع خطا
            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                errorMsg = 'خطا در ارتباط با سرور. لطفاً بررسی کنید:\n';
                errorMsg += '1. سرور Flask در حال اجرا است (python web_app.py)\n';
                errorMsg += '2. سرور روی پورت 5000 در حال اجرا است\n';
                errorMsg += '3. آدرس سرور درست است (http://127.0.0.1:5000)\n';
                errorMsg += '4. فایروال یا آنتی‌ویروس مانع ارتباط نشده است';
            } else if (error.message) {
                errorMsg += ': ' + error.message;
            } else {
                errorMsg += ': ' + error.toString();
            }
            
            // نمایش جزئیات خطا در console برای debugging
            console.error('Error details:', error);
            console.error('Request data:', requestData);
            
            showTextError(errorMsg);
        });
    }

    let graphNetwork = null; // Store vis-network instance

    function showTextSuccess(data) {
        const stats = data.stats || {};
        const graphStats = document.getElementById('graph-stats');
        const successText = document.getElementById('text-success-text');

        // Display graph statistics
        if (graphStats) {
            let statsHTML = `
                <div class="stat-item">
                    <span class="stat-value">${stats.num_nodes || 0}</span>
                    <span class="stat-label">نود</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.num_edges || 0}</span>
                    <span class="stat-label">یال</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.num_entities || 0}</span>
                    <span class="stat-label">موجودیت</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.num_relationships || 0}</span>
                    <span class="stat-label">رابطه</span>
                </div>
            `;
            
            // Add additional statistics if available
            if (stats.avg_degree !== undefined) {
                statsHTML += `
                    <div class="stat-item">
                        <span class="stat-value">${stats.avg_degree.toFixed(2)}</span>
                        <span class="stat-label">میانگین درجه</span>
                    </div>
                `;
            }
            
            if (stats.density !== undefined) {
                statsHTML += `
                    <div class="stat-item">
                        <span class="stat-value">${(stats.density * 100).toFixed(2)}%</span>
                        <span class="stat-label">تراکم</span>
                    </div>
                `;
            }
            
            graphStats.innerHTML = statsHTML;
        }
        
        // Display resolution summary if available
        if (data.resolution_summary) {
            const resolutionInfo = data.resolution_summary;
            if (resolutionInfo.resolved_groups > 0) {
                const resolutionText = `Entity Resolution: ${resolutionInfo.resolved_groups} گروه ادغام شدند، ${resolutionInfo.total_resolved_entities} موجودیت حل شدند.`;
                if (successText) {
                    successText.textContent += '\n' + resolutionText;
                }
            }
        }

        if (successText) {
            successText.textContent = `گراف با موفقیت ساخته و ذخیره شد. فایل: ${data.filename || 'نامشخص'}`;
        }

        // Display graph preview
        if (data.graph_data) {
            displayGraphPreview(data.graph_data);
            displayGraphDetails(data.graph_data);
        }

        textSuccessMessage.style.display = 'block';
        textErrorMessage.style.display = 'none';
        textResultSection.style.display = 'block';
        buildGraphBtn.disabled = false;
    }

    function displayGraphPreview(graphData) {
        const container = document.getElementById('graph-visualization');
        if (!container || !graphData) return;

        // Destroy previous network if exists
        if (graphNetwork) {
            graphNetwork.destroy();
            graphNetwork = null;
        }

        // Prepare nodes for vis-network with kind information
        const nodes = new vis.DataSet(graphData.nodes.map(node => ({
            id: node.id,
            label: node.label.length > 20 ? node.label.substring(0, 20) + '...' : node.label,
            title: node.title || `${node.label}\nنوع: ${node.type}\nKind: ${node.kind || node.type}`,
            color: getNodeColor(node.kind || node.type),
            font: { size: 14, face: 'Tahoma' },
            shape: 'dot',
            size: 16,
            kind: node.kind || node.type,
            type: node.type
        })));

        // Prepare edges for vis-network with metaedge information
        const edges = new vis.DataSet(graphData.edges.map(edge => ({
            from: edge.from,
            to: edge.to,
            label: edge.label || edge.metaedge || '',
            title: edge.title || `رابطه: ${edge.metaedge || edge.label || 'related_to'}`,
            metaedge: edge.metaedge || edge.label || 'related_to',
            color: { color: '#667eea', highlight: '#764ba2' },
            arrows: { to: { enabled: true, scaleFactor: 0.8 } },
            font: { size: 12, align: 'middle' },
            width: 2
        })));

        // Network options
        const options = {
            nodes: {
                borderWidth: 2,
                shadow: true,
                font: {
                    size: 14,
                    face: 'Tahoma'
                }
            },
            edges: {
                width: 2,
                shadow: true,
                smooth: {
                    type: 'continuous',
                    roundness: 0.5
                },
                font: {
                    size: 11,
                    align: 'middle'
                }
            },
            physics: {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 100
                },
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.1,
                    springLength: 200,
                    springConstant: 0.04,
                    damping: 0.09
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true
            },
            layout: {
                improvedLayout: true
            }
        };

        // Create network
        const data = { nodes: nodes, edges: edges };
        graphNetwork = new vis.Network(container, data, options);

        // Add event listeners
        graphNetwork.on('click', function(params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Clicked node:', node);
            }
        });
    }

    function getNodeColor(nodeType) {
        const colorMap = {
            'Gene': '#667eea',
            'Disease': '#e74c3c',
            'Compound': '#f39c12',
            'Biological Process': '#27ae60',
            'Pathway': '#9b59b6',
            'Anatomy': '#3498db',
            'Symptom': '#e67e22',
            'Cellular Component': '#16a085',
            'Molecular Function': '#d35400',
            'Unknown': '#95a5a6'
        };
        return colorMap[nodeType] || '#95a5a6';
    }

    function displayGraphDetails(graphData) {
        const nodesList = document.getElementById('nodes-list');
        const edgesList = document.getElementById('edges-list');

        if (nodesList && graphData.nodes) {
            nodesList.innerHTML = graphData.nodes.map(node => {
                const kind = node.kind || node.type || 'Unknown';
                const type = node.type || 'Unknown';
                return `
                <div class="node-item">
                    <span class="node-name">${node.label}</span>
                    <span class="node-type">نوع: ${type} | Kind: ${kind}</span>
                </div>
            `;
            }).join('');
        }

        if (edgesList && graphData.edges) {
            // Get node labels for display
            const nodeLabels = {};
            graphData.nodes.forEach(node => {
                nodeLabels[node.id] = node.label;
            });

            edgesList.innerHTML = graphData.edges.map(edge => {
                const relation = edge.relation || edge.metaedge || edge.label || 'related_to';
                const relationMeaning = edge.relation_meaning || edge.description || relation;
                const metaedge = edge.metaedge || '';
                
                return `
                <div class="edge-item">
                    <div class="edge-header">
                        <span class="edge-relation"><strong>${relation}</strong></span>
                        ${metaedge && metaedge !== relation ? `<span class="edge-metaedge">(${metaedge})</span>` : ''}
                    </div>
                    <div class="edge-nodes">${nodeLabels[edge.from] || edge.from} → ${nodeLabels[edge.to] || edge.to}</div>
                    <div class="edge-meaning">
                        <i class="fas fa-info-circle"></i> <span>مفهوم: ${relationMeaning}</span>
                    </div>
                </div>
            `;
            }).join('');
        }
    }

    // Toggle graph details
    window.toggleGraphDetails = function() {
        const details = document.getElementById('graph-details');
        if (details) {
            if (details.style.display === 'none') {
                details.style.display = 'block';
            } else {
                details.style.display = 'none';
            }
        }
    };

    function showTextError(message) {
        const errorText = document.getElementById('text-error-text');
        if (errorText) {
            errorText.textContent = message;
        }
        textErrorMessage.style.display = 'block';
        textSuccessMessage.style.display = 'none';
        textResultSection.style.display = 'block';
        buildGraphBtn.disabled = false;
    }

    // Global function for reset text upload
    window.resetTextUpload = function() {
        if (textInput) {
            textInput.value = '';
            if (textLength) textLength.textContent = '0';
        }
        
        // Destroy graph network if exists
        if (graphNetwork) {
            graphNetwork.destroy();
            graphNetwork = null;
        }
        
        // Clear graph details
        const nodesList = document.getElementById('nodes-list');
        const edgesList = document.getElementById('edges-list');
        if (nodesList) nodesList.innerHTML = '';
        if (edgesList) edgesList.innerHTML = '';
        
        // Hide details
        const details = document.getElementById('graph-details');
        if (details) details.style.display = 'none';
        
        textProgressSection.style.display = 'none';
        textResultSection.style.display = 'none';
        buildGraphBtn.disabled = false;
    };
}); 