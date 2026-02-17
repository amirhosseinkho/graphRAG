// JavaScript for View Graph Page
document.addEventListener('DOMContentLoaded', function() {
    const graphStats = document.getElementById('graph-stats');
    const nodesList = document.getElementById('nodes-list');
    const edgesList = document.getElementById('edges-list');

    const config = window.VIEW_GRAPH_CONFIG || {};
    const graphPath = config.graphPath;
    const graphName = config.graphName || 'گراف';

    let graphNetwork = null; // vis-network instance

    if (!graphPath) {
        if (graphStats) {
            graphStats.innerHTML = `
                <div class="error-card">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>مسیر گراف مشخص نشده است</h3>
                    <p>لطفاً از طریق صفحه مدیریت گراف‌ها یک گراف را انتخاب کنید.</p>
                </div>
            `;
        }
        return;
    }

    // Load graph data from server
    loadGraphData(graphPath);

    function loadGraphData(path) {
        fetch('/api/graph_view_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ graph_path: path })
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                showError(data.error || 'خطا در بارگذاری داده‌های گراف');
                return;
            }

            if (data.stats) {
                displayStats(data.stats, data.node_types || {});
            }

            if (data.graph_data) {
                displayGraphPreview(data.graph_data);
                displayGraphDetails(data.graph_data);
            }
        })
        .catch(error => {
            showError('خطا در ارتباط با سرور: ' + error.message);
        });
    }

    function displayStats(stats, nodeTypes) {
        if (!graphStats) return;

        const nodeTypesList = Object.entries(nodeTypes)
            .map(([type, count]) => `<li><strong>${type}:</strong> ${count}</li>`)
            .join('');

        let statsHTML = `
            <div class="graph-stats">
                <div class="stat-item">
                    <span class="stat-value">${stats.num_nodes || 0}</span>
                    <span class="stat-label">نود</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.num_edges || 0}</span>
                    <span class="stat-label">یال</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${(stats.avg_degree || 0).toFixed ? (stats.avg_degree || 0).toFixed(2) : stats.avg_degree || 0}</span>
                    <span class="stat-label">میانگین درجه</span>
                </div>
        `;

        if (stats.density !== undefined) {
            statsHTML += `
                <div class="stat-item">
                    <span class="stat-value">${(stats.density * 100).toFixed(2)}%</span>
                    <span class="stat-label">تراکم</span>
                </div>
            `;
        }

        statsHTML += '</div>';

        if (nodeTypesList) {
            statsHTML += `
                <div class="node-types-info">
                    <h4>انواع نودها:</h4>
                    <ul>${nodeTypesList}</ul>
                </div>
            `;
        }

        graphStats.innerHTML = statsHTML;
    }

    function displayGraphPreview(graphData) {
        const container = document.getElementById('graph-visualization');
        if (!container || !graphData) return;

        // اگر کتابخانه vis-network بارگذاری نشده باشد، فقط خطا در کنسول لاگ می‌کنیم
        if (typeof vis === 'undefined') {
            console.error('vis-network is not loaded; skipping graph visualization.');
            showError('کتابخانه نمایش گراف (vis-network) بارگذاری نشد. لطفاً اتصال اینترنت را بررسی کنید یا فایل vis-network را به‌صورت محلی اضافه کنید.');
            return;
        }

        // Destroy previous network if exists
        if (graphNetwork) {
            graphNetwork.destroy();
            graphNetwork = null;
        }

        // Prepare nodes for vis-network
        const nodes = new vis.DataSet(graphData.nodes.map(node => ({
            id: node.id,
            label: node.label && node.label.length > 20 ? node.label.substring(0, 20) + '...' : (node.label || String(node.id)),
            title: node.title || `${node.label || node.id}\nنوع: ${node.type || 'Unknown'}\nKind: ${node.kind || node.type || 'Unknown'}`,
            color: getNodeColor(node.kind || node.type),
            font: { size: 14, face: 'Tahoma' },
            shape: 'dot',
            size: 16,
            kind: node.kind || node.type,
            type: node.type
        })));

        // Prepare edges for vis-network
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

        const data = { nodes: nodes, edges: edges };
        graphNetwork = new vis.Network(container, data, options);
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

    function showError(message) {
        if (graphStats) {
            graphStats.innerHTML = `
                <div class="error-card">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>خطا در بارگذاری گراف</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    }

    // Toggle graph details (used by button in template)
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
});


