// JavaScript for Manage Graphs Page
document.addEventListener('DOMContentLoaded', function() {
    const graphsContainer = document.getElementById('graphs-container');
    const currentGraphInfo = document.getElementById('current-graph-info');
    const actionsSection = document.getElementById('actions-section');
    const refreshBtn = document.getElementById('refresh-btn');
    const loadGraphBtn = document.getElementById('load-graph-btn');
    const deleteGraphBtn = document.getElementById('delete-graph-btn');
    const graphDetailsModal = document.getElementById('graph-details-modal');
    const confirmationModal = document.getElementById('confirmation-modal');

    let selectedGraph = null;
    let currentGraphPath = null;

    // Load initial data
    loadCurrentGraphInfo();
    loadGraphsList();

    // Event listeners
    refreshBtn.addEventListener('click', function() {
        loadGraphsList();
    });

    loadGraphBtn.addEventListener('click', function() {
        if (selectedGraph) {
            loadSelectedGraph();
        }
    });

    deleteGraphBtn.addEventListener('click', function() {
        if (selectedGraph) {
            showDeleteConfirmation();
        }
    });

    // Modal close buttons
    document.querySelectorAll('.close, .close-modal-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            graphDetailsModal.style.display = 'none';
            confirmationModal.style.display = 'none';
        });
    });

    // Close modals when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === graphDetailsModal) {
            graphDetailsModal.style.display = 'none';
        }
        if (event.target === confirmationModal) {
            confirmationModal.style.display = 'none';
        }
    });

    function loadCurrentGraphInfo() {
        fetch('/api/graph_info')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayCurrentGraphInfo(data);
                } else {
                    displayCurrentGraphError(data.error);
                }
            })
            .catch(error => {
                displayCurrentGraphError('خطا در بارگذاری اطلاعات گراف فعلی');
            });
    }

    function displayCurrentGraphInfo(data) {
        const nodeTypesList = Object.entries(data.node_types)
            .map(([type, count]) => `<li><strong>${type}:</strong> ${count}</li>`)
            .join('');

        currentGraphInfo.innerHTML = `
            <div class="current-graph-card">
                <div class="graph-stats">
                    <div class="stat-item">
                        <i class="fas fa-circle-nodes"></i>
                        <span class="stat-number">${data.total_nodes.toLocaleString()}</span>
                        <span class="stat-label">کل نودها</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-project-diagram"></i>
                        <span class="stat-number">${data.total_edges.toLocaleString()}</span>
                        <span class="stat-label">کل یال‌ها</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-tags"></i>
                        <span class="stat-number">${Object.keys(data.node_types).length}</span>
                        <span class="stat-label">انواع نود</span>
                    </div>
                </div>
                <div class="node-types-info">
                    <h4>انواع نودها:</h4>
                    <ul>${nodeTypesList}</ul>
                </div>
            </div>
        `;
    }

    function displayCurrentGraphError(error) {
        currentGraphInfo.innerHTML = `
            <div class="error-card">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>خطا در بارگذاری گراف</h3>
                <p>${error}</p>
            </div>
        `;
    }

    function loadGraphsList() {
        graphsContainer.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-spinner fa-spin"></i>
                <p>در حال بارگذاری لیست گراف‌ها...</p>
            </div>
        `;

        fetch('/api/list_graphs')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayGraphsList(data.graphs);
                } else {
                    displayGraphsError(data.error);
                }
            })
            .catch(error => {
                displayGraphsError('خطا در بارگذاری لیست گراف‌ها');
            });
    }

    function displayGraphsList(graphs) {
        if (graphs.length === 0) {
            graphsContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <h3>هیچ گرافی یافت نشد</h3>
                    <p>برای شروع، یک فایل گراف آپلود کنید</p>
                    <a href="/upload_graph" class="btn btn-primary">
                        <i class="fas fa-upload"></i> آپلود گراف
                    </a>
                </div>
            `;
            return;
        }

        const graphsHTML = graphs.map(graph => {
            const fileSize = formatFileSize(graph.size);
            const fileDate = new Date(graph.date).toLocaleDateString('fa-IR');
            const isSelected = selectedGraph && selectedGraph.path === graph.path;
            
            return `
                <div class="graph-card ${isSelected ? 'selected' : ''}" data-graph-path="${graph.path}">
                    <div class="graph-header">
                        <div class="graph-type-badge ${graph.type}">
                            <i class="fas fa-${graph.type === 'builtin' ? 'database' : 'upload'}"></i>
                            ${graph.type === 'builtin' ? 'سیستمی' : 'آپلود شده'}
                        </div>
                        <div class="graph-actions">
                            <button class="btn btn-sm btn-info view-details-btn" title="مشاهده جزئیات">
                                <i class="fas fa-eye"></i>
                            </button>
                        </div>
                    </div>
                    <div class="graph-info">
                        <h4 class="graph-name">${graph.name}</h4>
                        <div class="graph-meta">
                            <span class="meta-item">
                                <i class="fas fa-calendar"></i>
                                ${fileDate}
                            </span>
                            <span class="meta-item">
                                <i class="fas fa-weight-hanging"></i>
                                ${fileSize}
                            </span>
                        </div>
                    </div>
                    <div class="graph-selection">
                        <label class="radio-label">
                            <input type="radio" name="selected-graph" value="${graph.path}" ${isSelected ? 'checked' : ''}>
                            <span class="checkmark"></span>
                            انتخاب این گراف
                        </label>
                    </div>
                </div>
            `;
        }).join('');

        graphsContainer.innerHTML = `
            <div class="graphs-grid">
                ${graphsHTML}
            </div>
        `;

        // Add event listeners
        addGraphCardEventListeners();
    }

    function displayGraphsError(error) {
        graphsContainer.innerHTML = `
            <div class="error-card">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>خطا در بارگذاری لیست</h3>
                <p>${error}</p>
                <button class="btn btn-primary" onclick="location.reload()">
                    <i class="fas fa-redo"></i> تلاش مجدد
                </button>
            </div>
        `;
    }

    function addGraphCardEventListeners() {
        // Radio button selection
        document.querySelectorAll('input[name="selected-graph"]').forEach(radio => {
            radio.addEventListener('change', function() {
                const graphPath = this.value;
                const graphCard = this.closest('.graph-card');
                
                // Remove selection from all cards
                document.querySelectorAll('.graph-card').forEach(card => {
                    card.classList.remove('selected');
                });
                
                // Add selection to current card
                graphCard.classList.add('selected');
                
                // Find the selected graph data
                selectedGraph = {
                    path: graphPath,
                    name: graphCard.querySelector('.graph-name').textContent
                };
                
                // Show actions
                actionsSection.style.display = 'block';
            });
        });

        // View details buttons
        document.querySelectorAll('.view-details-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const graphCard = this.closest('.graph-card');
                const graphPath = graphCard.dataset.graphPath;
                const graphName = graphCard.querySelector('.graph-name').textContent;

                // انتقال به صفحه نمایش گراف با ویژوال و جدول آماری
                const url = `/view_graph?graph_path=${encodeURIComponent(graphPath)}&graph_name=${encodeURIComponent(graphName)}`;
                window.location.href = url;
            });
        });
    }

    function showGraphDetails(graphPath, graphName) {
        // For now, we'll show basic info. In a real implementation, 
        // you might want to load more detailed information about the graph
        const detailsHTML = `
            <div class="graph-details-content">
                <h4>${graphName}</h4>
                <p><strong>مسیر:</strong> ${graphPath}</p>
                <p><strong>نوع:</strong> ${graphPath.includes('uploaded_graphs') ? 'آپلود شده' : 'سیستمی'}</p>
                <p><strong>تاریخ ایجاد:</strong> ${new Date().toLocaleDateString('fa-IR')}</p>
            </div>
        `;
        
        document.getElementById('graph-details').innerHTML = detailsHTML;
        graphDetailsModal.style.display = 'block';
        
        // Store the graph info for modal actions
        selectedGraph = { path: graphPath, name: graphName };
    }

    function loadSelectedGraph() {
        if (!selectedGraph) return;

        loadGraphBtn.disabled = true;
        loadGraphBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال بارگذاری...';

        fetch('/api/load_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                graph_path: selectedGraph.path
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showSuccessMessage(data.message);
                // Reload current graph info
                loadCurrentGraphInfo();
            } else {
                showErrorMessage(data.error);
            }
        })
        .catch(error => {
            showErrorMessage('خطا در بارگذاری گراف: ' + error.message);
        })
        .finally(() => {
            loadGraphBtn.disabled = false;
            loadGraphBtn.innerHTML = '<i class="fas fa-play"></i> بارگذاری گراف انتخاب شده';
        });
    }

    function showDeleteConfirmation() {
        if (!selectedGraph) return;

        document.getElementById('confirmation-text').textContent = 
            `آیا مطمئن هستید که می‌خواهید گراف "${selectedGraph.name}" را حذف کنید؟ این عملیات قابل بازگشت نیست.`;
        
        confirmationModal.style.display = 'block';
        
        // Set up confirmation button
        const confirmBtn = document.getElementById('confirm-btn');
        confirmBtn.onclick = function() {
            deleteSelectedGraph();
            confirmationModal.style.display = 'none';
        };
    }

    function deleteSelectedGraph() {
        if (!selectedGraph) return;

        deleteGraphBtn.disabled = true;
        deleteGraphBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال حذف...';

        fetch('/api/delete_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                graph_path: selectedGraph.path
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showSuccessMessage(data.message);
                // Reload graphs list
                loadGraphsList();
                selectedGraph = null;
                actionsSection.style.display = 'none';
            } else {
                showErrorMessage(data.error);
            }
        })
        .catch(error => {
            showErrorMessage('خطا در حذف گراف: ' + error.message);
        })
        .finally(() => {
            deleteGraphBtn.disabled = false;
            deleteGraphBtn.innerHTML = '<i class="fas fa-trash"></i> حذف گراف انتخاب شده';
        });
    }

    function showSuccessMessage(message) {
        // Create a temporary success message
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message-temp';
        successDiv.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(successDiv);
        
        setTimeout(() => {
            successDiv.remove();
        }, 3000);
    }

    function showErrorMessage(message) {
        // Create a temporary error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message-temp';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}); 