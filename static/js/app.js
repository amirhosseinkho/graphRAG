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

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Enter key in query textarea
        document.getElementById('query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.processQuery();
            }
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

        const retrievalMethod = document.getElementById('retrieval-method').value;
        const generationModel = document.getElementById('generation-model').value;
        const maxDepth = parseInt(document.getElementById('max-depth').value);

        this.showLoading(true);

        try {
            const response = await fetch('/api/process_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    retrieval_method: retrievalMethod,
                    generation_model: generationModel,
                    max_depth: maxDepth
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

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    showError(message) {
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