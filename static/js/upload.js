// JavaScript for Upload Graph Page
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
}); 