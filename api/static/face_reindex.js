/**
 * FaceReindex - reusable module for reindexing faces on a single photo
 * 
 * Features:
 * - Threshold slider popup for adjusting detection sensitivity
 * - Progress toast notifications with spinner
 * - Success/error feedback
 * - Consistent behavior across all pages
 * 
 * Usage:
 *   const reindexer = new FaceReindex({
 *     onClearOverlays: () => { faceOverlays.innerHTML = ''; },
 *     onComplete: (data) => { 
 *       currentFaces = data.faces;
 *       renderFaceOverlays();
 *     }
 *   });
 *   reindexer.reindex(imageId);
 */
class FaceReindex {
    constructor(options = {}) {
        this.onClearOverlays = options.onClearOverlays || (() => {});
        this.onComplete = options.onComplete || (() => {});
        this.onError = options.onError || (() => {});
        this.toastElement = null;
        this.popupElement = null;
        this.defaultDetThresh = 0.45;  // Detection threshold (lower = more faces)
        this.defaultThreshold = 0.6;   // Assignment threshold (higher = stricter matching)
        this.defaultHd = false;        // HD 1280px detection (better for small/distant faces)
        this.injectStyles();
    }

    /**
     * Inject CSS styles if not already present
     */
    injectStyles() {
        if (document.getElementById('face-reindex-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'face-reindex-styles';
        style.textContent = `
            .face-reindex-toast {
                position: fixed;
                bottom: 80px;
                left: 50%;
                transform: translateX(-50%);
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                z-index: 10000;
                display: flex;
                align-items: center;
                gap: 10px;
                animation: faceReindexFadeIn 0.3s ease;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }
            
            .face-reindex-toast.progress {
                background: #1e3a5f;
                border: 1px solid #3b82f6;
                color: #93c5fd;
            }
            
            .face-reindex-toast.success {
                background: #14532d;
                border: 1px solid #22c55e;
                color: #86efac;
            }
            
            .face-reindex-toast.error {
                background: #7f1d1d;
                border: 1px solid #ef4444;
                color: #fca5a5;
            }
            
            .face-reindex-toast .spinner {
                width: 16px;
                height: 16px;
                border: 2px solid transparent;
                border-top-color: currentColor;
                border-radius: 50%;
                animation: faceReindexSpin 1s linear infinite;
            }
            
            @keyframes faceReindexSpin {
                to { transform: rotate(360deg); }
            }
            
            @keyframes faceReindexFadeIn {
                from { 
                    opacity: 0; 
                    transform: translateX(-50%) translateY(10px); 
                }
                to { 
                    opacity: 1; 
                    transform: translateX(-50%) translateY(0); 
                }
            }
            
            /* Popup overlay */
            .face-reindex-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.6);
                z-index: 10001;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: faceReindexOverlayFadeIn 0.2s ease;
            }
            
            @keyframes faceReindexOverlayFadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            /* Popup dialog */
            .face-reindex-popup {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 12px;
                padding: 24px;
                min-width: 320px;
                max-width: 400px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                animation: faceReindexPopupSlideIn 0.2s ease;
            }
            
            @keyframes faceReindexPopupSlideIn {
                from { 
                    opacity: 0; 
                    transform: scale(0.95) translateY(-10px); 
                }
                to { 
                    opacity: 1; 
                    transform: scale(1) translateY(0); 
                }
            }
            
            .face-reindex-popup h3 {
                margin: 0 0 16px 0;
                color: #f3f4f6;
                font-size: 16px;
                font-weight: 600;
            }
            
            .face-reindex-popup .threshold-info {
                color: #9ca3af;
                font-size: 13px;
                margin-bottom: 16px;
                line-height: 1.5;
            }
            
            .face-reindex-popup .slider-container {
                margin-bottom: 20px;
            }
            
            .face-reindex-popup .slider-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .face-reindex-popup .slider-label {
                color: #d1d5db;
                font-size: 13px;
            }
            
            .face-reindex-popup .slider-value {
                background: #374151;
                color: #60a5fa;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 600;
                font-family: monospace;
            }
            
            .face-reindex-popup input[type="range"] {
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: #374151;
                outline: none;
                -webkit-appearance: none;
                cursor: pointer;
            }
            
            .face-reindex-popup input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #3b82f6;
                cursor: pointer;
                border: 2px solid #60a5fa;
                transition: background 0.15s;
            }
            
            .face-reindex-popup input[type="range"]::-webkit-slider-thumb:hover {
                background: #60a5fa;
            }
            
            .face-reindex-popup input[type="range"]::-moz-range-thumb {
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #3b82f6;
                cursor: pointer;
                border: 2px solid #60a5fa;
            }
            
            .face-reindex-popup .slider-hints {
                display: flex;
                justify-content: space-between;
                margin-top: 6px;
                font-size: 11px;
                color: #6b7280;
            }
            
            .face-reindex-popup .slider-desc {
                margin-top: 8px;
                font-size: 12px;
                color: #9ca3af;
                font-style: italic;
            }

            .face-reindex-popup .hd-row {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 20px;
                padding: 10px 12px;
                background: #111827;
                border-radius: 6px;
                border: 1px solid #374151;
                cursor: pointer;
            }

            .face-reindex-popup .hd-row:hover {
                border-color: #4b5563;
            }

            .face-reindex-popup .hd-row input[type="checkbox"] {
                width: 16px;
                height: 16px;
                accent-color: #3b82f6;
                cursor: pointer;
            }

            .face-reindex-popup .hd-label {
                flex: 1;
            }

            .face-reindex-popup .hd-label strong {
                display: block;
                color: #d1d5db;
                font-size: 13px;
                font-weight: 600;
            }

            .face-reindex-popup .hd-label span {
                color: #6b7280;
                font-size: 12px;
            }

            .face-reindex-popup .buttons {
                display: flex;
                gap: 12px;
                justify-content: flex-end;
            }
            
            .face-reindex-popup .btn {
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                cursor: pointer;
                border: none;
                transition: all 0.15s;
            }
            
            .face-reindex-popup .btn-cancel {
                background: #374151;
                color: #d1d5db;
                border: 1px solid #4b5563;
            }
            
            .face-reindex-popup .btn-cancel:hover {
                background: #4b5563;
            }
            
            .face-reindex-popup .btn-reindex {
                background: #3b82f6;
                color: white;
            }
            
            .face-reindex-popup .btn-reindex:hover {
                background: #2563eb;
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Show threshold selection popup with two sliders
     * @returns {Promise<{det_thresh: number, threshold: number}|null>} - Selected thresholds or null if cancelled
     */
    showThresholdPopup() {
        return new Promise((resolve) => {
            // Create overlay
            const overlay = document.createElement('div');
            overlay.className = 'face-reindex-overlay';
            
            // Create popup
            const popup = document.createElement('div');
            popup.className = 'face-reindex-popup';
            popup.innerHTML = `
                <h3>🔄 Переиндексация лиц</h3>
                <div class="threshold-info">
                    Настройте пороги для детекции и назначения лиц.
                </div>
                
                <!-- HD detection checkbox -->
                <label class="hd-row">
                    <input type="checkbox" id="hdCheckbox" ${this.defaultHd ? 'checked' : ''}>
                    <div class="hd-label">
                        <strong>🔭 HD детекция (1280px)</strong>
                        <span>Лучше находит маленькие и дальние лица. Медленнее ~2×.</span>
                    </div>
                </label>

                <!-- Detection threshold slider -->
                <div class="slider-container">
                    <div class="slider-header">
                        <span class="slider-label">🔍 Порог детекции:</span>
                        <span class="slider-value" id="detThreshValue">${this.defaultDetThresh.toFixed(2)}</span>
                    </div>
                    <input type="range" id="detThreshSlider"
                           min="0.10" max="0.8" step="0.05"
                           value="${this.defaultDetThresh}">
                    <div class="slider-hints">
                        <span>0.10 (больше лиц)</span>
                        <span>0.80 (только явные)</span>
                    </div>
                    <div class="slider-desc">Ниже порог = больше лиц найдено (включая размытые/под углом)</div>
                </div>
                
                <!-- Assignment threshold slider -->
                <div class="slider-container">
                    <div class="slider-header">
                        <span class="slider-label">🔗 Порог назначения:</span>
                        <span class="slider-value" id="thresholdValue">${this.defaultThreshold.toFixed(2)}</span>
                    </div>
                    <input type="range" id="thresholdSlider" 
                           min="0.3" max="0.95" step="0.05" 
                           value="${this.defaultThreshold}">
                    <div class="slider-hints">
                        <span>0.30 (больше назначений)</span>
                        <span>0.95 (точнее)</span>
                    </div>
                    <div class="slider-desc">Выше порог = строже соответствие для авто-назначения персоны</div>
                </div>
                
                <div class="buttons">
                    <button class="btn btn-cancel" id="btnCancel">Отмена</button>
                    <button class="btn btn-reindex" id="btnReindex">Переиндексировать</button>
                </div>
            `;
            
            overlay.appendChild(popup);
            document.body.appendChild(overlay);
            this.popupElement = overlay;
            
            // Get elements
            const hdCheckbox = popup.querySelector('#hdCheckbox');
            const detThreshSlider = popup.querySelector('#detThreshSlider');
            const detThreshValue = popup.querySelector('#detThreshValue');
            const thresholdSlider = popup.querySelector('#thresholdSlider');
            const thresholdValue = popup.querySelector('#thresholdValue');
            const btnCancel = popup.querySelector('#btnCancel');
            const btnReindex = popup.querySelector('#btnReindex');
            
            // Update value displays on slider change
            detThreshSlider.addEventListener('input', () => {
                detThreshValue.textContent = parseFloat(detThreshSlider.value).toFixed(2);
            });
            
            thresholdSlider.addEventListener('input', () => {
                thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
            });
            
            // Close popup helper
            const closePopup = () => {
                overlay.remove();
                this.popupElement = null;
            };
            
            // Cancel button
            btnCancel.addEventListener('click', () => {
                closePopup();
                resolve(null);
            });
            
            // Reindex button
            btnReindex.addEventListener('click', () => {
                const result = {
                    det_thresh: parseFloat(detThreshSlider.value),
                    threshold: parseFloat(thresholdSlider.value),
                    hd: hdCheckbox.checked
                };
                closePopup();
                resolve(result);
            });
            
            // Click outside to cancel
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    closePopup();
                    resolve(null);
                }
            });
            
            // Escape key to cancel
            const handleKeydown = (e) => {
                if (e.key === 'Escape') {
                    closePopup();
                    document.removeEventListener('keydown', handleKeydown);
                    resolve(null);
                }
            };
            document.addEventListener('keydown', handleKeydown);
            
            // Focus first slider
            detThreshSlider.focus();
        });
    }

    /**
     * Main reindex method - shows popup first, then executes
     * @param {number} imageId - The image ID to reindex
     * @returns {Promise<Object>} - The API response data
     */
    async reindex(imageId) {
        if (!imageId) {
            this.showError('Нет ID изображения');
            return null;
        }

        // Show threshold selection popup
        const thresholds = await this.showThresholdPopup();
        if (thresholds === null) {
            // User cancelled
            return null;
        }

        return await this.executeReindex(imageId, thresholds.det_thresh, thresholds.threshold, thresholds.hd);
    }

    /**
     * Execute reindex with specific thresholds
     * @param {number} imageId - The image ID to reindex
     * @param {number} detThresh - Detection threshold (lower = more faces detected)
     * @param {number} threshold - Assignment threshold for auto-matching
     * @returns {Promise<Object>} - The API response data
     */
    async executeReindex(imageId, detThresh, threshold, hd = false) {
        try {
            // 1. Clear overlays immediately
            this.onClearOverlays();
            
            // 2. Show initial progress
            this.showProgress('Удаление старых лиц...');
            
            // 3. Call API with both threshold parameters
            const response = await fetch(`/photo/${imageId}/faces/reindex?det_thresh=${detThresh}&threshold=${threshold}&hd=${hd}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            
            // 4. Update progress during detection
            this.updateProgress('Поиск новых лиц...');
            
            // 5. Parse response
            const data = await response.json();
            
            // 6. Show success message
            const message = this.formatSuccessMessage(data);
            this.showSuccess(message);
            
            // 7. Call completion callback
            this.onComplete(data);
            
            return data;
            
        } catch (error) {
            this.showError('Ошибка: ' + error.message);
            this.onError(error);
            return null;
        }
    }

    /**
     * Format success message based on result
     */
    formatSuccessMessage(data) {
        let msg = `Найдено лиц: ${data.detected_faces || 0}`;
        if (data.assigned > 0) {
            msg += `, назначено: ${data.assigned}`;
        }
        return msg;
    }

    /**
     * Show progress toast
     */
    showProgress(message) {
        this.removeToast();
        const toast = document.createElement('div');
        toast.className = 'face-reindex-toast progress';
        toast.innerHTML = `<span class="spinner"></span> ${this.escapeHtml(message)}`;
        this.toastElement = toast;
        document.body.appendChild(toast);
    }

    /**
     * Update progress message
     */
    updateProgress(message) {
        if (this.toastElement) {
            this.toastElement.innerHTML = `<span class="spinner"></span> ${this.escapeHtml(message)}`;
        }
    }

    /**
     * Show success toast
     */
    showSuccess(message) {
        if (this.toastElement) {
            this.toastElement.className = 'face-reindex-toast success';
            this.toastElement.innerHTML = `✓ ${this.escapeHtml(message)}`;
            setTimeout(() => this.removeToast(), 3000);
        }
    }

    /**
     * Show error toast
     */
    showError(message) {
        this.removeToast();
        const toast = document.createElement('div');
        toast.className = 'face-reindex-toast error';
        toast.innerHTML = `✗ ${this.escapeHtml(message)}`;
        this.toastElement = toast;
        document.body.appendChild(toast);
        setTimeout(() => this.removeToast(), 4000);
    }

    /**
     * Remove toast element
     */
    removeToast() {
        if (this.toastElement) {
            this.toastElement.remove();
            this.toastElement = null;
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FaceReindex };
}
