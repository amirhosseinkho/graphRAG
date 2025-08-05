document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const queryInput = document.getElementById('queryInput');
    const textGenerationTypeSelect = document.getElementById('text-generation-type');
    
    // Answer 1 elements
    const tokenExtractionMethod1Select = document.getElementById('token_extraction_method1');
    const tokenExtractionModel1Select = document.getElementById('token_extraction_model1');
    const retrievalMethod1Select = document.getElementById('retrieval_method1');
    const generationModel1Select = document.getElementById('generation_model1');
    const maxDepth1Input = document.getElementById('max_depth1');
    const inputType1Select = document.getElementById('input_type1');
    const answer1Textarea = document.getElementById('answer1');
    
    // Answer 2 elements
    const tokenExtractionMethod2Select = document.getElementById('token_extraction_method2');
    const tokenExtractionModel2Select = document.getElementById('token_extraction_model2');
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
    
    // GPT comparison elements
    const gptCompareBtn = document.getElementById('gptCompareBtn');
    const gptStatus = document.getElementById('gptStatus');
    const gptComparisonModelSelect = document.getElementById('gptComparisonModel');
    const method1Label = document.getElementById('method1Label');
    const method2Label = document.getElementById('method2Label');
    const comparisonType = document.getElementById('comparisonType');
    const autoLabel1Btn = document.getElementById('autoLabel1Btn');
    const autoLabel2Btn = document.getElementById('autoLabel2Btn');

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
        gptCompareBtn.addEventListener('click', compareWithGPT);
        
        // Add auto-label buttons
        autoLabel1Btn.addEventListener('click', () => generateAutoLabel(1));
        autoLabel2Btn.addEventListener('click', () => generateAutoLabel(2));
        
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
        
        // Add background text change listener
        textGenerationTypeSelect.addEventListener('change', () => {
            // Update auto labels when background text changes
            generateAutoLabel(1);
            generateAutoLabel(2);
        });
        
        // Generate initial auto labels
        generateAutoLabel(1);
        generateAutoLabel(2);
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

    function generateAutoLabel(answerBoxNumber) {
        const retrievalMethod = answerBoxNumber === 1 ? retrievalMethod1Select.value : retrievalMethod2Select.value;
        const generationModel = answerBoxNumber === 1 ? generationModel1Select.value : generationModel2Select.value;
        const textGenerationType = textGenerationTypeSelect.value;
        
        let label = '';
        
        // Generate label based on retrieval method
        const retrievalLabels = {
            'NO_RETRIEVAL': 'بدون بازیابی',
            'BFS': 'BFS (جستجوی سطح اول)',
            'DFS': 'DFS (جستجوی عمیق اول)',
            'SHORTEST_PATH': 'کوتاه‌ترین مسیر',
            'NEIGHBORS': 'همسایه‌ها',
            'HYBRID': 'ترکیبی',
            'MULTI_METHOD': 'چند روشی',
            'ENSEMBLE': 'گروهی',
            'ADAPTIVE': 'تطبیقی',
            'INTELLIGENT': 'جستجوی هوشمند'
        };
        
        // Generate label based on generation model
        const modelLabels = {
            'OPENAI_GPT_4O': 'GPT-4o',
            'OPENAI_GPT_4O_MINI': 'GPT-4o Mini',
            'OPENAI_GPT_4_TURBO': 'GPT-4 Turbo',
            'OPENAI_GPT_4': 'GPT-4',
            'OPENAI_GPT_3_5_TURBO': 'GPT-3.5 Turbo',
            'OPENAI_GPT_3_5_TURBO_16K': 'GPT-3.5 Turbo 16K',
            'HUGGINGFACE': 'HuggingFace',
            'GPT_SIMULATION': 'GPT Simulation',
            'CUSTOM': 'Custom Model',
            'SIMPLE': 'Simple Template'
        };
        
        // Generate label based on text generation type
        const textTypeLabels = {
            'SIMPLE': 'متن ساده',
            'INTELLIGENT': 'متن هوشمند',
            'SCIENTIFIC_ANALYTICAL': 'سبک علمی',
            'NARRATIVE_DESCRIPTIVE': 'سبک روایی',
            'DATA_DRIVEN': 'سبک داده‌محور',
            'STEP_BY_STEP': 'سبک گام به گام',
            'CONCISE_DIRECT': 'سبک فشرده'
        };
        
        const retrievalLabel = retrievalLabels[retrievalMethod] || 'روش نامشخص';
        const modelLabel = modelLabels[generationModel] || 'مدل نامشخص';
        const textTypeLabel = textTypeLabels[textGenerationType] || 'متن نامشخص';
        
        label = `${retrievalLabel} + ${modelLabel} + ${textTypeLabel}`;
        
        // Update the appropriate label input
        if (answerBoxNumber === 1) {
            method1Label.value = label;
        } else {
            method2Label.value = label;
        }
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
        
        // Update auto label when method changes
        generateAutoLabel(answerBoxNumber);
    }

    function generateAnswer(answerBoxNumber) {
        const query = queryInput.value;
        if (!query.trim()) {
            showError('لطفاً ابتدا سوال را وارد کنید');
            return;
        }

        const textarea = answerBoxNumber === 1 ? answer1Textarea : answer2Textarea;
        const tokenExtractionMethod = answerBoxNumber === 1 ? tokenExtractionMethod1Select.value : tokenExtractionMethod2Select.value;
        const tokenExtractionModel = answerBoxNumber === 1 ? tokenExtractionModel1Select.value : tokenExtractionModel2Select.value;
        const retrievalMethod = answerBoxNumber === 1 ? retrievalMethod1Select.value : retrievalMethod2Select.value;
        const generationModel = answerBoxNumber === 1 ? generationModel1Select.value : generationModel2Select.value;
        const maxDepth = answerBoxNumber === 1 ? maxDepth1Input.value : maxDepth2Input.value;
        const textGenerationType = textGenerationTypeSelect.value;

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
                token_extraction_method: tokenExtractionMethod,
                token_extraction_model: tokenExtractionModel,
                retrieval_method: retrievalMethod,
                generation_model: generationModel,
                max_depth: parseInt(maxDepth),
                text_generation_type: textGenerationType
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

    function compareWithGPT() {
        const text1 = answer1Textarea.value.trim();
        const text2 = answer2Textarea.value.trim();
        const label1 = method1Label.value.trim() || 'روش اول';
        const label2 = method2Label.value.trim() || 'روش دوم';
        const comparisonTypeValue = comparisonType.value;
        const gptModel = gptComparisonModelSelect.value;
        
        if (!text1 || !text2) {
            showError('لطفاً هر دو پاسخ را وارد کنید');
            return;
        }

        // Show loading state
        gptCompareBtn.disabled = true;
        gptCompareBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال مقایسه...';
        gptStatus.className = 'gpt-status processing';
        gptStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال ارسال به GPT...';
        resultsSection.style.display = 'none';

        // Call API to compare with GPT
        fetch('/api/compare_with_gpt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text1: text1,
                text2: text2,
                label1: label1,
                label2: label2,
                comparison_type: comparisonTypeValue,
                gpt_model: gptModel
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                gptStatus.className = 'gpt-status error';
                gptStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> خطا در مقایسه';
            } else {
                displayGPTResults(data);
                gptStatus.className = 'gpt-status success';
                gptStatus.innerHTML = '<i class="fas fa-check-circle"></i> مقایسه تکمیل شد';
            }
        })
        .catch(error => {
            showError('خطا در ارتباط با سرور: ' + error.message);
            gptStatus.className = 'gpt-status error';
            gptStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> خطا در ارتباط';
        })
        .finally(() => {
            gptCompareBtn.disabled = false;
            gptCompareBtn.innerHTML = '<i class="fas fa-robot"></i> مقایسه با GPT';
        });
    }

    function displayGPTResults(data) {
        const comparisonTypeText = getComparisonTypeText(data.comparison_type);
        const modelName = data.gpt_model || 'GPT';
        
        resultsContent.innerHTML = `
            <div class="gpt-result">
                <h4><i class="fas fa-robot"></i> تحلیل ${modelName} - ${comparisonTypeText}</h4>
                
                <div class="gpt-analysis">
                    <h5><i class="fas fa-chart-bar"></i> خلاصه مقایسه</h5>
                    <p>${data.summary}</p>
                </div>
                
                <div class="gpt-analysis">
                    <h5><i class="fas fa-star"></i> امتیازدهی</h5>
                    <div style="margin: 10px 0;">
                        <span class="gpt-score">${data.label1}: ${data.score1}/100</span>
                        <span class="gpt-score">${data.label2}: ${data.score2}/100</span>
                    </div>
                    <p>${data.scoring_explanation}</p>
                </div>
                
                <div class="gpt-analysis">
                    <h5><i class="fas fa-search"></i> تحلیل جزئیات</h5>
                    <p><strong>نقاط قوت ${data.label1}:</strong></p>
                    <p>${data.strengths1}</p>
                    <p><strong>نقاط قوت ${data.label2}:</strong></p>
                    <p>${data.strengths2}</p>
                    <p><strong>نقاط ضعف ${data.label1}:</strong></p>
                    <p>${data.weaknesses1}</p>
                    <p><strong>نقاط ضعف ${data.label2}:</strong></p>
                    <p>${data.weaknesses2}</p>
                </div>
                
                <div class="gpt-recommendation">
                    <h5><i class="fas fa-lightbulb"></i> توصیه نهایی</h5>
                    <p>${data.recommendation}</p>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; font-size: 0.9rem; color: #6c757d;">
                    <strong>نکته:</strong> این تحلیل توسط ${modelName} انجام شده و صرفاً جنبه راهنمایی دارد. برای تصمیم‌گیری نهایی، عوامل دیگری مانند سرعت، هزینه، و نیازهای خاص پروژه نیز باید در نظر گرفته شوند.
                </div>
            </div>
        `;
        
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function getComparisonTypeText(type) {
        const types = {
            'comprehensive': 'مقایسه جامع',
            'accuracy': 'تمرکز بر دقت',
            'completeness': 'تمرکز بر جامعیت',
            'clarity': 'تمرکز بر وضوح',
            'relevance': 'تمرکز بر مرتبط بودن'
        };
        return types[type] || 'مقایسه';
    }
}); 