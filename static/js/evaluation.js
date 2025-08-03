document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const queryInput = document.getElementById('queryInput');
    const retrievalMethod1Select = document.getElementById('retrieval_method1');
    const generationModel1Select = document.getElementById('generation_model1');
    const maxDepth1Input = document.getElementById('max_depth1');
    const inputType1Select = document.getElementById('input_type1');
    const answer1Textarea = document.getElementById('answer1');
    
    const retrievalMethod2Select = document.getElementById('retrieval_method2');
    const generationModel2Select = document.getElementById('generation_model2');
    const maxDepth2Input = document.getElementById('max_depth2');
    const inputType2Select = document.getElementById('input_type2');
    const answer2Textarea = document.getElementById('answer2');
    
    const generateBtn = document.getElementById('generateBtn');
    const compareBtn = document.getElementById('compareBtn');
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');
    const methodOptions = document.querySelectorAll('.method-option');

    let selectedMethod = 'cosine_tfidf';

    // Initialize
    initializePage();

    function initializePage() {
        // Set default selected method
        selectMethod('cosine_tfidf');
        
        // Add event listeners
        methodOptions.forEach(option => {
            option.addEventListener('click', function() {
                const method = this.getAttribute('data-method');
                selectMethod(method);
            });
        });

        generateBtn.addEventListener('click', generateBothAnswers);
        compareBtn.addEventListener('click', compareAnswers);
        
        // Add method change listeners for both answer boxes
        inputType1Select.addEventListener('change', () => handleInputTypeChange(1));
        inputType2Select.addEventListener('change', () => handleInputTypeChange(2));
        
        // Add auto-generation listeners
        retrievalMethod1Select.addEventListener('change', () => handleAutoGeneration(1));
        generationModel1Select.addEventListener('change', () => handleAutoGeneration(1));
        maxDepth1Input.addEventListener('change', () => handleAutoGeneration(1));
        
        retrievalMethod2Select.addEventListener('change', () => handleAutoGeneration(2));
        generationModel2Select.addEventListener('change', () => handleAutoGeneration(2));
        maxDepth2Input.addEventListener('change', () => handleAutoGeneration(2));
    }

    function selectMethod(method) {
        // Remove previous selection
        methodOptions.forEach(option => {
            option.classList.remove('selected');
        });
        
        // Add selection to clicked option
        const selectedOption = document.querySelector(`[data-method="${method}"]`);
        if (selectedOption) {
            selectedOption.classList.add('selected');
        }
        
        selectedMethod = method;
    }

    function handleInputTypeChange(answerBoxNumber) {
        const inputTypeSelect = answerBoxNumber === 1 ? inputType1Select : inputType2Select;
        const textarea = answerBoxNumber === 1 ? answer1Textarea : answer2Textarea;
        
        if (inputTypeSelect.value === 'manual') {
            textarea.disabled = false;
            textarea.placeholder = 'پاسخ خود را اینجا تایپ کنید...';
        } else {
            textarea.disabled = true;
            textarea.placeholder = 'پاسخ با روش انتخاب شده تولید خواهد شد...';
            
            // Generate answer based on current settings
            generateAnswer(answerBoxNumber);
        }
    }

    function handleAutoGeneration(answerBoxNumber) {
        const inputTypeSelect = answerBoxNumber === 1 ? inputType1Select : inputType2Select;
        
        if (inputTypeSelect.value === 'auto') {
            generateAnswer(answerBoxNumber);
        }
    }

    function generateAnswer(answerBoxNumber) {
        const query = queryInput.value;
        if (!query.trim()) {
            showError('لطفاً ابتدا سوال را وارد کنید');
            return;
        }

        const textarea = answerBoxNumber === 1 ? answer1Textarea : answer2Textarea;
        const retrievalMethod = answerBoxNumber === 1 ? retrievalMethod1Select.value : retrievalMethod2Select.value;
        const generationModel = answerBoxNumber === 1 ? generationModel1Select.value : generationModel2Select.value;
        const maxDepth = answerBoxNumber === 1 ? maxDepth1Input.value : maxDepth2Input.value;

        // Show loading state
        textarea.value = 'در حال تولید پاسخ...';
        textarea.style.color = '#7f8c8d';

        // Call API to generate answer
        fetch('/api/process_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                retrieval_method: retrievalMethod,
                generation_model: generationModel,
                max_depth: parseInt(maxDepth)
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                textarea.value = data.result.answer || 'پاسخ تولید شد اما متن خالی است';
            } else {
                textarea.value = `خطا در تولید پاسخ: ${data.error}`;
            }
            textarea.style.color = '#2c3e50';
        })
        .catch(error => {
            textarea.value = `خطا در ارتباط با سرور: ${error.message}`;
            textarea.style.color = '#e74c3c';
        });
    }



    function generateBothAnswers() {
        const query = queryInput.value;
        if (!query.trim()) {
            showError('لطفاً ابتدا سوال را وارد کنید');
            return;
        }

        // Set both input types to auto
        inputType1Select.value = 'auto';
        inputType2Select.value = 'auto';

        // Generate both answers
        generateAnswer(1);
        generateAnswer(2);

        // Show success message
        showSuccess('در حال تولید پاسخ‌ها...');
    }

    function compareAnswers() {
        const text1 = answer1Textarea.value.trim();
        const text2 = answer2Textarea.value.trim();
        
        if (!text1 || !text2) {
            showError('لطفاً هر دو پاسخ را وارد کنید');
            return;
        }

        // Show loading state
        compareBtn.disabled = true;
        compareBtn.textContent = 'در حال مقایسه...';
        resultsSection.style.display = 'none';

        // Call API to compare texts
        fetch('/api/compare_texts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text1: text1,
                text2: text2,
                method: selectedMethod
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                displayResults(data);
            }
        })
        .catch(error => {
            showError('خطا در ارتباط با سرور: ' + error.message);
        })
        .finally(() => {
            compareBtn.disabled = false;
            compareBtn.textContent = '🔍 مقایسه پاسخ‌ها';
        });
    }

    function displayResults(data) {
        const qualityClass = getQualityClass(data.quality_level);
        
        resultsContent.innerHTML = `
            <div class="result-item">
                <h4>📊 نتایج مقایسه با ${data.method_name}</h4>
                <div class="similarity-score">${(data.similarity_score * 100).toFixed(1)}%</div>
                <div class="quality-level ${qualityClass}">سطح کیفیت: ${data.quality_level}</div>
                
                <h5>📝 متن‌های پردازش شده:</h5>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>متن اول:</strong><br>
                    <small>${data.text1_processed}</small>
                </div>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>متن دوم:</strong><br>
                    <small>${data.text2_processed}</small>
                </div>
                
                <div style="margin-top: 15px; padding: 10px; background: #e8f5e8; border-radius: 5px;">
                    <strong>💡 تفسیر نتیجه:</strong><br>
                    ${getInterpretation(data.similarity_score, data.method_name)}
                </div>
            </div>
        `;
        
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function getQualityClass(qualityLevel) {
        const qualityMap = {
            'عالی': 'quality-excellent',
            'خیلی خوب': 'quality-very-good',
            'خوب': 'quality-good',
            'متوسط': 'quality-medium',
            'ضعیف': 'quality-poor',
            'خیلی ضعیف': 'quality-very-poor'
        };
        return qualityMap[qualityLevel] || 'quality-medium';
    }

    function getInterpretation(score, methodName) {
        const percentage = (score * 100).toFixed(1);
        
        if (score >= 0.9) {
            return `پاسخ‌ها بسیار مشابه هستند (${percentage}%). این نشان‌دهنده کیفیت بالای هر دو روش است.`;
        } else if (score >= 0.8) {
            return `پاسخ‌ها بسیار شبیه هستند (${percentage}%). هر دو روش عملکرد خوبی دارند.`;
        } else if (score >= 0.7) {
            return `پاسخ‌ها شبیه هستند (${percentage}%). تفاوت‌های جزئی وجود دارد.`;
        } else if (score >= 0.6) {
            return `پاسخ‌ها تا حدی مشابه هستند (${percentage}%). تفاوت‌های قابل توجهی وجود دارد.`;
        } else if (score >= 0.5) {
            return `پاسخ‌ها کم‌شباهت هستند (${percentage}%). تفاوت‌های زیادی وجود دارد.`;
        } else {
            return `پاسخ‌ها بسیار متفاوت هستند (${percentage}%). نیاز به بررسی بیشتر وجود دارد.`;
        }
    }

    function showSuccess(message) {
        resultsContent.innerHTML = `
            <div style="background: #d5f4e6; color: #27ae60; padding: 15px; border-radius: 8px; margin: 10px 0;">
                ✅ ${message}
            </div>
        `;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        resultsContent.innerHTML = `
            <div class="error-message">
                ❌ ${message}
            </div>
        `;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}); 