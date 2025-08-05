// GraphRAG Web Application JavaScript

class GraphRAGApp {
    constructor() {
        this.initializeEventListeners();
        this.loadGraphInfo();
        this.loadSampleQueries();
    }

    initializeEventListeners() {
        // Submit button
        document.getElementById('submit-btn').addEventListener('click', () => {
            this.processQuery();
        });

        // Sample query button
        document.getElementById('sample-btn').addEventListener('click', () => {
            this.loadRandomSampleQuery();
        });

        // Config button
        document.getElementById('config-btn').addEventListener('click', () => {
            this.openConfigModal();
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Enhanced tab switching
        document.querySelectorAll('.enhanced-tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchEnhancedTab(e.target.dataset.enhancedTab);
            });
        });

        // Enhanced service toggle
        document.getElementById('use-enhanced-service').addEventListener('change', (e) => {
            this.toggleEnhancedService(e.target.checked);
        });

        // Enter key in query textarea
        document.getElementById('query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.processQuery();
            }
        });

        // Modal event listeners
        this.initializeModalEvents();
    }

    initializeModalEvents() {
        const modal = document.getElementById('config-modal');
        const closeBtn = modal.querySelector('.close');
        const closeConfigBtn = document.getElementById('close-config-btn');
        const saveConfigBtn = document.getElementById('save-config-btn');
        const resetConfigBtn = document.getElementById('reset-config-btn');

        // Close modal
        closeBtn.addEventListener('click', () => this.closeConfigModal());
        closeConfigBtn.addEventListener('click', () => this.closeConfigModal());
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeConfigModal();
            }
        });

        // Save config
        saveConfigBtn.addEventListener('click', () => this.saveConfig());

        // Reset config
        resetConfigBtn.addEventListener('click', () => this.resetConfig());

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.applyPreset(e.target.dataset.preset);
            });
        });
    }

    async loadGraphInfo() {
        try {
            const response = await fetch('/api/graph_info');
            const data = await response.json();
            
            if (data.success) {
                this.displayGraphInfo(data);
            } else {
                console.error('Error loading graph info:', data.error);
            }
        } catch (error) {
            console.error('Error loading graph info:', error);
        }
    }

    displayGraphInfo(data) {
        document.getElementById('total-nodes').textContent = data.total_nodes;
        document.getElementById('total-edges').textContent = data.total_edges;
        document.getElementById('node-types').textContent = Object.keys(data.node_types).length;

        // Create chart
        this.createNodeTypesChart(data.node_types);
    }

    createNodeTypesChart(nodeTypes) {
        const ctx = document.createElement('canvas');
        ctx.id = 'nodeTypesChart';
        document.getElementById('node-types-chart').appendChild(ctx);

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(nodeTypes),
                datasets: [{
                    data: Object.values(nodeTypes),
                    backgroundColor: [
                        '#667eea',
                        '#764ba2',
                        '#f093fb',
                        '#f5576c',
                        '#4facfe',
                        '#00f2fe',
                        '#43e97b',
                        '#38f9d7'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }

    async loadSampleQueries() {
        try {
            const response = await fetch('/api/sample_queries');
            const data = await response.json();
            this.sampleQueries = data.queries;
        } catch (error) {
            console.error('Error loading sample queries:', error);
            this.sampleQueries = [];
        }
    }

    loadRandomSampleQuery() {
        if (this.sampleQueries && this.sampleQueries.length > 0) {
            const randomQuery = this.sampleQueries[Math.floor(Math.random() * this.sampleQueries.length)];
            document.getElementById('query').value = randomQuery;
        }
    }

    async processQuery() {
        const query = document.getElementById('query').value.trim();
        if (!query) {
            alert('لطفاً سوال خود را وارد کنید');
            return;
        }

        // بررسی نوع سرویس (معمولی یا پیشرفته)
        const useEnhancedService = document.getElementById('use-enhanced-service')?.checked || false;

        if (useEnhancedService) {
            await this.processEnhancedQuery(query);
        } else {
            await this.processStandardQuery(query);
        }
    }

    async processStandardQuery(query) {
        const tokenExtractionMethod = document.getElementById('token-extraction-method').value;
        const tokenExtractionModel = document.getElementById('token-extraction-model').value;
        const retrievalMethod = document.getElementById('retrieval-method').value;
        const generationModel = document.getElementById('generation-model').value;
        const textGenerationType = document.getElementById('text-generation-type').value;
        const maxDepth = parseInt(document.getElementById('max-depth').value);

        // تنظیمات جدید
        const similarityThreshold = parseFloat(document.getElementById('similarity-threshold')?.value || 0.3);
        const nHopDepth = parseInt(document.getElementById('n-hop-depth')?.value || 2);
        const pagerankDamping = parseFloat(document.getElementById('pagerank-damping')?.value || 0.85);
        const entityResolutionThreshold = parseFloat(document.getElementById('entity-resolution-threshold')?.value || 0.8);
        const communityDetectionResolution = parseFloat(document.getElementById('community-detection-resolution')?.value || 1.0);
        const enableLLMCaching = document.getElementById('enable-llm-caching')?.checked || true;
        const enableEntityResolution = document.getElementById('enable-entity-resolution')?.checked || true;
        const enableCommunityDetection = document.getElementById('enable-community-detection')?.checked || true;
        const enableSemanticSearch = document.getElementById('enable-semantic-search')?.checked || true;
        const enablePagerankScoring = document.getElementById('enable-pagerank-scoring')?.checked || true;

        this.showLoading(true);

        try {
            const response = await fetch('/api/process_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    token_extraction_method: tokenExtractionMethod,
                    token_extraction_model: tokenExtractionModel,
                    retrieval_method: retrievalMethod,
                    generation_model: generationModel,
                    text_generation_type: textGenerationType,
                    max_depth: maxDepth,
                    // تنظیمات جدید
                    similarity_threshold: similarityThreshold,
                    n_hop_depth: nHopDepth,
                    pagerank_damping: pagerankDamping,
                    entity_resolution_threshold: entityResolutionThreshold,
                    community_detection_resolution: communityDetectionResolution,
                    enable_llm_caching: enableLLMCaching,
                    enable_entity_resolution: enableEntityResolution,
                    enable_community_detection: enableCommunityDetection,
                    enable_semantic_search: enableSemanticSearch,
                    enable_pagerank_scoring: enablePagerankScoring
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayResults(data.result);
            } else {
                this.showError(data.error);
            }
        } catch (error) {
            this.showError('خطا در ارتباط با سرور: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async processEnhancedQuery(query) {
        const tokenExtractionMethod = document.getElementById('enhanced-token-extraction-method')?.value || 'llm_based';
        const retrievalAlgorithm = document.getElementById('enhanced-retrieval-algorithm')?.value || 'hybrid';
        const communityDetectionMethod = document.getElementById('enhanced-community-detection-method')?.value || 'louvain';
        const maxDepth = parseInt(document.getElementById('enhanced-max-depth')?.value || 3);
        const maxNodes = parseInt(document.getElementById('enhanced-max-nodes')?.value || 20);
        const maxEdges = parseInt(document.getElementById('enhanced-max-edges')?.value || 40);
        const similarityThreshold = parseFloat(document.getElementById('enhanced-similarity-threshold')?.value || 0.3);

        this.showLoading(true);

        try {
            const response = await fetch('/api/enhanced_process_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    token_extraction_method: tokenExtractionMethod,
                    retrieval_algorithm: retrievalAlgorithm,
                    community_detection_method: communityDetectionMethod,
                    max_depth: maxDepth,
                    max_nodes: maxNodes,
                    max_edges: maxEdges,
                    similarity_threshold: similarityThreshold
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayEnhancedResults(data.result);
            } else {
                this.showError(data.error);
            }
        } catch (error) {
            this.showError('خطا در ارتباط با سرور: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        // Show results section
        document.getElementById('results-section').style.display = 'block';

        // Display process steps
        this.displayProcessSteps(result.process_steps);

        // Display keywords
        this.displayKeywords(result.keywords);

        // Display matched nodes
        this.displayMatchedNodes(result.matched_nodes);

        // Display retrieved information
        this.displayRetrievedNodes(result.retrieved_nodes);
        this.displayRetrievedEdges(result.retrieved_edges);
        this.displayRetrievedPaths(result.paths);
        this.displayContextText(result.context_text);

        // Display answer
        this.displayAnswer(result.answer, result.confidence, result.generation_model);

        // Scroll to results
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    }

    displayProcessSteps(steps) {
        const container = document.getElementById('process-steps');
        container.innerHTML = steps.map(step => 
            `<div class="step-item">${step}</div>`
        ).join('');
    }

    displayKeywords(keywords) {
        const container = document.getElementById('keywords');
        container.innerHTML = keywords.map(keyword => 
            `<span class="keyword-tag">${keyword}</span>`
        ).join('');
    }

    displayMatchedNodes(matchedNodes) {
        const container = document.getElementById('matched-nodes');
        container.innerHTML = Object.entries(matchedNodes).map(([token, nodeName]) => 
            `<div class="node-card">
                <div class="node-header">
                    <span class="node-name">${nodeName}</span>
                    <span class="node-type">تطبیق یافته</span>
                </div>
                <div class="node-details">
                    <div class="node-detail">
                        <span>توکن:</span>
                        <span>${token}</span>
                    </div>
                </div>
            </div>`
        ).join('');
    }

    displayRetrievedNodes(nodes) {
        const container = document.getElementById('retrieved-nodes');
        container.innerHTML = nodes.map(node => 
            `<div class="node-card">
                <div class="node-header">
                    <span class="node-name">${node.name}</span>
                    <span class="node-type">${node.kind}</span>
                </div>
                <div class="node-details">
                    <div class="node-detail">
                        <span>شناسه:</span>
                        <span>${node.id}</span>
                    </div>
                    <div class="node-detail">
                        <span>عمق:</span>
                        <span>${node.depth}</span>
                    </div>
                    <div class="node-detail">
                        <span>امتیاز:</span>
                        <span>${node.score.toFixed(2)}</span>
                    </div>
                </div>
            </div>`
        ).join('');
    }

    displayRetrievedEdges(edges) {
        const container = document.getElementById('retrieved-edges');
        container.innerHTML = edges.map(edge => 
            `<div class="edge-item">
                <div><strong>از:</strong> ${this.getNodeNameById(edge.source)}</div>
                <div><strong>به:</strong> ${this.getNodeNameById(edge.target)}</div>
                <span class="edge-relation">${edge.relation}</span>
            </div>`
        ).join('');
    }

    displayRetrievedPaths(paths) {
        const container = document.getElementById('retrieved-paths');
        if (paths.length === 0) {
            container.innerHTML = '<div class="path-item">هیچ مسیری یافت نشد</div>';
            return;
        }

        container.innerHTML = paths.map((path, index) => 
            `<div class="path-item">
                <div><strong>مسیر ${index + 1}:</strong></div>
                <div class="path-nodes">
                    ${path.map((nodeId, i) => 
                        `${i > 0 ? '<span class="path-arrow">→</span>' : ''}<span class="path-node">${this.getNodeNameById(nodeId)}</span>`
                    ).join('')}
                </div>
            </div>`
        ).join('');
    }

    displayContextText(contextText) {
        document.getElementById('context-text').textContent = contextText;
    }

    displayAnswer(answer, confidence, model) {
        document.getElementById('answer-text').textContent = answer;
        document.getElementById('confidence-score').textContent = (confidence * 100).toFixed(1) + '%';
        document.getElementById('model-name').textContent = model;
    }

    displayEnhancedResults(result) {
        // Show results section
        document.getElementById('results-section').style.display = 'block';

        // Display query analysis
        this.displayQueryAnalysis(result.query_analysis);

        // Display nodes
        this.displayEnhancedNodes(result.nodes);

        // Display edges
        this.displayEnhancedEdges(result.edges);

        // Display communities if available
        if (result.communities && result.communities.length > 0) {
            this.displayCommunities(result.communities);
        }

        // Display similarities if available
        if (result.similarities && result.similarities.length > 0) {
            this.displaySimilarities(result.similarities);
        }

        // Display paths if available
        if (result.paths && result.paths.length > 0) {
            this.displayEnhancedPaths(result.paths);
        }

        // Scroll to results
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    }

    displayQueryAnalysis(analysis) {
        const container = document.getElementById('query-analysis');
        if (!container) return;

        container.innerHTML = `
            <div class="analysis-section">
                <h4>تحلیل سوال</h4>
                <div class="analysis-item">
                    <strong>سوال:</strong> ${analysis.query}
                </div>
                <div class="analysis-item">
                    <strong>نوع پاسخ:</strong> ${analysis.answer_types.join(', ')}
                </div>
                <div class="analysis-item">
                    <strong>موجودیت‌ها:</strong> ${analysis.entities.join(', ')}
                </div>
                <div class="analysis-item">
                    <strong>نودهای شروع:</strong> ${analysis.start_nodes.join(', ')}
                </div>
                <div class="analysis-item">
                    <strong>الگوریتم:</strong> ${analysis.algorithm}
                </div>
                <div class="analysis-item">
                    <strong>روش استخراج توکن:</strong> ${analysis.token_extraction_method}
                </div>
            </div>
        `;
    }

    displayEnhancedNodes(nodes) {
        const container = document.getElementById('enhanced-nodes');
        if (!container) return;

        if (!nodes || nodes.length === 0) {
            container.innerHTML = '<div class="no-results">هیچ نودی یافت نشد</div>';
            return;
        }

        container.innerHTML = nodes.map(node => {
            const nodeData = typeof node === 'object' ? node : { id: node };
            return `
                <div class="node-card enhanced">
                    <div class="node-header">
                        <span class="node-name">${nodeData.id}</span>
                        ${nodeData.source ? `<span class="node-source">${nodeData.source}</span>` : ''}
                    </div>
                    <div class="node-details">
                        ${nodeData.pagerank ? `<div class="node-detail"><span>PageRank:</span><span>${nodeData.pagerank.toFixed(4)}</span></div>` : ''}
                        ${nodeData.similarity ? `<div class="node-detail"><span>شباهت:</span><span>${nodeData.similarity.toFixed(4)}</span></div>` : ''}
                        ${nodeData.community !== undefined ? `<div class="node-detail"><span>جامعه:</span><span>${nodeData.community}</span></div>` : ''}
                        ${nodeData.attributes ? `<div class="node-detail"><span>نوع:</span><span>${nodeData.attributes.kind || 'نامشخص'}</span></div>` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }

    displayEnhancedEdges(edges) {
        const container = document.getElementById('enhanced-edges');
        if (!container) return;

        if (!edges || edges.length === 0) {
            container.innerHTML = '<div class="no-results">هیچ یالی یافت نشد</div>';
            return;
        }

        container.innerHTML = edges.map(edge => `
            <div class="edge-item enhanced">
                <div class="edge-header">
                    <span class="edge-source">${edge.source}</span>
                    <span class="edge-arrow">→</span>
                    <span class="edge-target">${edge.target}</span>
                </div>
                <div class="edge-details">
                    <div class="edge-detail">
                        <span>رابطه:</span>
                        <span>${edge.relation || 'نامشخص'}</span>
                    </div>
                    <div class="edge-detail">
                        <span>وزن:</span>
                        <span>${edge.weight || 1.0}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    displayCommunities(communities) {
        const container = document.getElementById('enhanced-communities');
        if (!container) return;

        container.innerHTML = communities.map(community => `
            <div class="community-card">
                <div class="community-header">
                    <span class="community-id">جامعه ${community.id}</span>
                    <span class="community-size">${community.size} نود</span>
                </div>
                <div class="community-nodes">
                    ${community.nodes.slice(0, 10).map(node => `<span class="community-node">${node}</span>`).join('')}
                    ${community.nodes.length > 10 ? `<span class="more-nodes">+${community.nodes.length - 10} بیشتر</span>` : ''}
                </div>
            </div>
        `).join('');
    }

    displaySimilarities(similarities) {
        const container = document.getElementById('enhanced-similarities');
        if (!container) return;

        container.innerHTML = similarities.map(sim => `
            <div class="similarity-item">
                <span class="similarity-node">${sim.node}</span>
                <span class="similarity-score">${sim.score.toFixed(4)}</span>
            </div>
        `).join('');
    }

    displayEnhancedPaths(paths) {
        const container = document.getElementById('enhanced-paths');
        if (!container) return;

        container.innerHTML = paths.map((path, index) => `
            <div class="path-item enhanced">
                <div class="path-header">
                    <span class="path-number">مسیر ${index + 1}</span>
                    <span class="path-length">${path.length} نود</span>
                </div>
                <div class="path-nodes">
                    ${path.map((node, i) => 
                        `${i > 0 ? '<span class="path-arrow">→</span>' : ''}<span class="path-node">${node}</span>`
                    ).join('')}
                </div>
            </div>
        `).join('');
    }

    getNodeNameById(nodeId) {
        // This would need to be implemented with actual node data
        // For now, we'll extract the name from the ID
        const parts = nodeId.split('::');
        return parts.length > 1 ? parts[1] : nodeId;
    }

    switchTab(tabName) {
        // Remove active class from all tabs and panes
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

        // Add active class to selected tab and pane
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`tab-${tabName}`).classList.add('active');
    }

    switchEnhancedTab(tabName) {
        // Remove active class from all enhanced tabs and panes
        document.querySelectorAll('.enhanced-tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.enhanced-tab-pane').forEach(pane => pane.classList.remove('active'));

        // Add active class to selected enhanced tab and pane
        document.querySelector(`[data-enhanced-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`tab-${tabName}`).classList.add('active');
    }

    toggleEnhancedService(enabled) {
        const enhancedSettings = document.getElementById('enhanced-settings');
        const enhancedResults = document.getElementById('enhanced-results');
        const standardResults = document.getElementById('results-section');

        if (enabled) {
            enhancedSettings.style.display = 'block';
            enhancedResults.style.display = 'block';
            standardResults.style.display = 'none';
        } else {
            enhancedSettings.style.display = 'none';
            enhancedResults.style.display = 'none';
            standardResults.style.display = 'block';
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    showError(message) {
        alert('خطا: ' + message);
    }

    // Configuration Methods
    async openConfigModal() {
        const modal = document.getElementById('config-modal');
        modal.style.display = 'block';
        
        // Load current config
        await this.loadCurrentConfig();
        
        // Load presets
        await this.loadPresets();
    }

    closeConfigModal() {
        const modal = document.getElementById('config-modal');
        modal.style.display = 'none';
    }

    async loadCurrentConfig() {
        try {
            const response = await fetch('/api/config');
            const data = await response.json();
            
            if (data.success) {
                this.displayCurrentConfig(data.config);
                this.populateConfigForm(data.config);
            }
        } catch (error) {
            console.error('Error loading config:', error);
        }
    }

    displayCurrentConfig(config) {
        const display = document.getElementById('current-config-display');
        display.innerHTML = '';
        
        Object.entries(config).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'config-item';
            item.innerHTML = `
                <span class="config-key">${this.formatConfigKey(key)}:</span>
                <span class="config-value">${value}</span>
            `;
            display.appendChild(item);
        });
    }

    formatConfigKey(key) {
        const keyMap = {
            'max_nodes': 'حداکثر نودها',
            'max_edges': 'حداکثر یال‌ها',
            'max_depth': 'حداکثر عمق',
            'max_paths': 'حداکثر مسیرها',
            'max_context_length': 'حداکثر طول متن زمینه',
            'max_answer_tokens': 'حداکثر توکن‌های پاسخ',
            'max_prompt_tokens': 'حداکثر توکن‌های ورودی',
            'enable_verbose_logging': 'نمایش جزئیات',
            'enable_biological_enrichment': 'غنی‌سازی زیستی',
            'enable_smart_filtering': 'فیلتر هوشمند'
        };
        return keyMap[key] || key;
    }

    populateConfigForm(config) {
        // Populate number inputs
        document.getElementById('max-nodes').value = config.max_nodes || 10;
        document.getElementById('max-edges').value = config.max_edges || 20;
        document.getElementById('max-depth-config').value = config.max_depth || 3;
        document.getElementById('max-paths').value = config.max_paths || 5;
        document.getElementById('max-context-length').value = config.max_context_length || 2000;
        document.getElementById('max-answer-tokens').value = config.max_answer_tokens || 1000;
        document.getElementById('max-prompt-tokens').value = config.max_prompt_tokens || 4000;
        
        // Populate checkboxes
        document.getElementById('enable-verbose-logging').checked = config.enable_verbose_logging || false;
        document.getElementById('enable-biological-enrichment').checked = config.enable_biological_enrichment || false;
        document.getElementById('enable-smart-filtering').checked = config.enable_smart_filtering || false;
    }

    async loadPresets() {
        try {
            const response = await fetch('/api/config/presets');
            const data = await response.json();
            this.presets = data.presets;
        } catch (error) {
            console.error('Error loading presets:', error);
        }
    }

    applyPreset(presetName) {
        if (!this.presets || !this.presets[presetName]) {
            console.error('Preset not found:', presetName);
            return;
        }

        const preset = this.presets[presetName];
        const config = preset.config;
        
        // Update form values
        this.populateConfigForm(config);
        
        // Update current config display
        this.displayCurrentConfig(config);
        
        // Update active preset button
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-preset="${presetName}"]`).classList.add('active');
        
        // Show preset description
        this.showPresetDescription(preset);
    }

    showPresetDescription(preset) {
        // You can add a toast or notification here
        console.log(`Applied preset: ${preset.name} - ${preset.description}`);
    }

    async saveConfig() {
        const config = this.getConfigFromForm();
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ config })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayCurrentConfig(data.config);
                this.showSuccessMessage('تنظیمات با موفقیت ذخیره شد');
            } else {
                this.showErrorMessage('خطا در ذخیره تنظیمات: ' + data.error);
            }
        } catch (error) {
            console.error('Error saving config:', error);
            this.showErrorMessage('خطا در ذخیره تنظیمات');
        }
    }

    getConfigFromForm() {
        return {
            max_nodes: parseInt(document.getElementById('max-nodes').value),
            max_edges: parseInt(document.getElementById('max-edges').value),
            max_depth: parseInt(document.getElementById('max-depth-config').value),
            max_paths: parseInt(document.getElementById('max-paths').value),
            max_context_length: parseInt(document.getElementById('max-context-length').value),
            max_answer_tokens: parseInt(document.getElementById('max-answer-tokens').value),
            max_prompt_tokens: parseInt(document.getElementById('max-prompt-tokens').value),
            enable_verbose_logging: document.getElementById('enable-verbose-logging').checked,
            enable_biological_enrichment: document.getElementById('enable-biological-enrichment').checked,
            enable_smart_filtering: document.getElementById('enable-smart-filtering').checked
        };
    }

    async resetConfig() {
        // Reset to default values
        const defaultConfig = {
            max_nodes: 10,
            max_edges: 20,
            max_depth: 3,
            max_paths: 5,
            max_context_length: 2000,
            max_answer_tokens: 1000,
            max_prompt_tokens: 4000,
            enable_verbose_logging: true,
            enable_biological_enrichment: true,
            enable_smart_filtering: true
        };
        
        this.populateConfigForm(defaultConfig);
        this.displayCurrentConfig(defaultConfig);
        
        // Remove active class from preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
        });
    }

    showSuccessMessage(message) {
        // Simple alert for now, you can replace with a better notification system
        alert(message);
    }

    showErrorMessage(message) {
        // Simple alert for now, you can replace with a better notification system
        alert('خطا: ' + message);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new GraphRAGApp();
});

// Utility functions
function formatNumber(num) {
    return new Intl.NumberFormat('fa-IR').format(num);
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleString('fa-IR');
}

// Add some sample data for demonstration
window.sampleData = {
    nodes: {
        'Gene::HMGB3': 'HMGB3',
        'Gene::PCNA': 'PCNA',
        'Disease::Diabetes': 'Type 2 Diabetes',
        'Drug::Metformin': 'Metformin'
    }
}; 