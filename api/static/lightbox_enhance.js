/**
 * LightboxEnhance — zoom, pan, full-size loading, fullscreen for any lightbox.
 *
 * Usage:
 *   <script src="/lightbox_enhance.js"></script>
 *   LightboxEnhance.init({
 *       lightbox:        '#lightbox',          // lightbox container selector
 *       img:             '#lightboxImg',        // image element selector
 *       content:         '#lightboxContent',    // lightbox-content wrapper selector
 *       controls:        '.lightbox-controls',  // controls bar selector (buttons appended here)
 *       getImageId:      () => currentImageId,  // function returning current image_id
 *       onBeforeZoom:    null,                  // optional callback before zoom starts
 *   });
 *
 * Features:
 *   - 🔍 Scroll wheel zoom (Ctrl optional) + pinch-to-zoom on touch
 *   - 🖱️ Pan (drag) when zoomed in
 *   - 📐 Double-click to toggle zoom 1:1 / fit
 *   - 🖼️ "Full size" button to load original image (no 2000px limit)
 *   - ⛶  Fullscreen button + F11 key
 *   - Reset on image change (watches img.src via MutationObserver)
 */
(function () {
    'use strict';

    // ── State ──────────────────────────────────────────────
    let _cfg = null;
    let _lightbox = null;
    let _img = null;
    let _content = null;
    let _scale = 1;
    let _panX = 0;
    let _panY = 0;
    let _isDragging = false;
    let _dragStartX = 0;
    let _dragStartY = 0;
    let _panStartX = 0;
    let _panStartY = 0;
    let _isFullSize = false;
    let _fullSizeBtn = null;
    let _fullscreenBtn = null;
    let _zoomIndicator = null;
    let _pinchStartDist = 0;
    let _pinchStartScale = 1;
    let _lastImageId = null;

    const MIN_SCALE = 0.5;
    const MAX_SCALE = 15;

    // ── CSS injection ──────────────────────────────────────
    function _injectCSS() {
        if (document.getElementById('lightbox-enhance-styles')) return;
        const style = document.createElement('style');
        style.id = 'lightbox-enhance-styles';
        style.textContent = `
            /* Zoom indicator */
            .lbe-zoom-indicator {
                position: absolute;
                top: 60px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0,0,0,0.75);
                color: #fff;
                padding: 4px 14px;
                border-radius: 16px;
                font-size: 13px;
                font-family: monospace;
                z-index: 40;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.25s;
                white-space: nowrap;
            }
            .lbe-zoom-indicator.visible {
                opacity: 1;
            }

            /* Zoomed-in cursor */
            .lbe-zoomed .lightbox-content {
                cursor: grab !important;
            }
            .lbe-zoomed .lightbox-content.lbe-dragging {
                cursor: grabbing !important;
            }
            .lbe-zoomed .lightbox-content img {
                pointer-events: none;
            }

            /* Override max-width/max-height when full-size loaded or zoomed */
            .lbe-zoomed .lightbox-content img {
                max-width: none !important;
                max-height: none !important;
            }

            /* Full-size button active state */
            .lightbox-controls .btn-icon.lbe-fullsize-active {
                background: rgba(66, 165, 245, 0.85) !important;
                color: #fff;
            }
            .lightbox-controls .btn-icon.lbe-fullsize-loading {
                animation: lbe-pulse 1s infinite;
            }
            @keyframes lbe-pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.4; }
            }

            /* Fullscreen button active */
            .lightbox-controls .btn-icon.lbe-fs-active {
                background: rgba(76, 175, 80, 0.85) !important;
                color: #fff;
            }

            /* Fullscreen mode: truly full */
            .lightbox:-webkit-full-screen,
            .lightbox:fullscreen {
                background: #000 !important;
            }

            /* Face overlays follow zoom transform */
            .lbe-zoomed #faceOverlays {
                transform-origin: 0 0;
            }
        `;
        document.head.appendChild(style);
    }

    // ── Helpers ─────────────────────────────────────────────
    function _clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

    function _applyTransform() {
        if (!_img) return;
        _img.style.transformOrigin = '0 0';
        _img.style.transform = `translate(${_panX}px, ${_panY}px) scale(${_scale})`;

        // Sync face overlays if they exist
        const overlays = _content?.querySelector('#faceOverlays');
        if (overlays) {
            overlays.style.transformOrigin = '0 0';
            overlays.style.transform = `translate(${_panX}px, ${_panY}px) scale(${_scale})`;
        }

        // Zoom indicator
        _showZoomIndicator();

        // Toggle class for cursor changes
        if (_scale > 1.01) {
            _lightbox.classList.add('lbe-zoomed');
        } else {
            _lightbox.classList.remove('lbe-zoomed');
        }
    }

    function _showZoomIndicator() {
        if (!_zoomIndicator) return;
        const pct = Math.round(_scale * 100);
        let label = pct + '%';
        if (_isFullSize) label += ' · FULL';
        _zoomIndicator.textContent = label;
        _zoomIndicator.classList.add('visible');
        clearTimeout(_zoomIndicator._hideTimer);
        _zoomIndicator._hideTimer = setTimeout(() => {
            _zoomIndicator.classList.remove('visible');
        }, 1200);
    }

    function _resetZoom() {
        _scale = 1;
        _panX = 0;
        _panY = 0;
        if (_img) {
            _img.style.transform = '';
            _img.style.transformOrigin = '';
            // Restore the page CSS constraints
            _img.style.maxWidth = '';
            _img.style.maxHeight = '';
        }
        const overlays = _content?.querySelector('#faceOverlays');
        if (overlays) {
            overlays.style.transform = '';
            overlays.style.transformOrigin = '';
        }
        _lightbox?.classList.remove('lbe-zoomed');
    }

    function _constrainPan() {
        if (_scale <= 1) {
            _panX = 0;
            _panY = 0;
            return;
        }
        // Allow panning so the image doesn't go completely offscreen
        const rect = _content.getBoundingClientRect();
        const imgW = (_img.naturalWidth || _img.width) * _scale;
        const imgH = (_img.naturalHeight || _img.height) * _scale;
        const maxPanX = Math.max(0, (imgW - rect.width) / 2 + rect.width * 0.4);
        const maxPanY = Math.max(0, (imgH - rect.height) / 2 + rect.height * 0.4);
        _panX = _clamp(_panX, -maxPanX, maxPanX);
        _panY = _clamp(_panY, -maxPanY, maxPanY);
    }

    // ── Zoom around point ──────────────────────────────────
    function _zoomAtPoint(cx, cy, factor) {
        const oldScale = _scale;
        const newScale = _clamp(_scale * factor, MIN_SCALE, MAX_SCALE);
        if (newScale === oldScale) return;

        // Get image position relative to content container
        const rect = _content.getBoundingClientRect();
        // Position of cursor relative to the image's transformed origin
        const ix = cx - rect.left - _panX;
        const iy = cy - rect.top - _panY;

        // Adjust pan so the point under cursor stays fixed
        const ratio = newScale / oldScale;
        _panX = cx - rect.left - ix * ratio;
        _panY = cy - rect.top - iy * ratio;
        _scale = newScale;

        _constrainPan();
        _applyTransform();
    }

    // ── Full-size loading ──────────────────────────────────
    function _loadFullSize() {
        const imageId = _cfg.getImageId();
        if (!imageId) return;

        if (_isFullSize) {
            // Toggle back to normal
            _isFullSize = false;
            _fullSizeBtn.classList.remove('lbe-fullsize-active');
            _resetZoom();
            _img.src = `/image/${imageId}/full`;
            return;
        }

        _fullSizeBtn.classList.add('lbe-fullsize-loading');

        // Pre-load the original image
        const loader = new Image();
        loader.onload = function () {
            // Swap the src
            _img.src = loader.src;
            _isFullSize = true;
            _fullSizeBtn.classList.remove('lbe-fullsize-loading');
            _fullSizeBtn.classList.add('lbe-fullsize-active');
            // Remove CSS max constraints so we can zoom to real pixels
            _img.style.maxWidth = 'none';
            _img.style.maxHeight = 'none';
            // Fit to viewport
            _fitToViewport();
            _showZoomIndicator();
        };
        loader.onerror = function () {
            _fullSizeBtn.classList.remove('lbe-fullsize-loading');
        };
        loader.src = `/image/${imageId}/original`;
    }

    function _fitToViewport() {
        if (!_img.naturalWidth) return;
        const vw = _lightbox.clientWidth * 0.95;
        const vh = _lightbox.clientHeight * 0.95;
        const fitScale = Math.min(vw / _img.naturalWidth, vh / _img.naturalHeight, 1);
        _scale = fitScale;
        _panX = 0;
        _panY = 0;
        _applyTransform();
    }

    // ── Fullscreen ──────────────────────────────────────────
    function _toggleFullscreen() {
        if (document.fullscreenElement || document.webkitFullscreenElement) {
            (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        } else {
            const el = _lightbox;
            (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
        }
    }

    function _onFullscreenChange() {
        const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
        if (_fullscreenBtn) {
            _fullscreenBtn.classList.toggle('lbe-fs-active', isFs);
            _fullscreenBtn.title = isFs ? 'Выйти из полноэкранного режима' : 'Полный экран';
            _fullscreenBtn.textContent = isFs ? '⛶' : '⛶';
        }
    }

    // ── Create buttons ─────────────────────────────────────
    function _createButtons() {
        const controls = _lightbox.querySelector(_cfg.controls);
        if (!controls) return;

        // Full-size button
        _fullSizeBtn = document.createElement('button');
        _fullSizeBtn.className = 'btn-icon';
        _fullSizeBtn.title = 'Загрузить оригинал (полный размер)';
        _fullSizeBtn.textContent = '🔍';
        _fullSizeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            _loadFullSize();
        });
        controls.appendChild(_fullSizeBtn);

        // Fullscreen button
        _fullscreenBtn = document.createElement('button');
        _fullscreenBtn.className = 'btn-icon';
        _fullscreenBtn.title = 'Полный экран';
        _fullscreenBtn.textContent = '⛶';
        _fullscreenBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            _toggleFullscreen();
        });
        controls.appendChild(_fullscreenBtn);

        // Zoom indicator
        _zoomIndicator = document.createElement('div');
        _zoomIndicator.className = 'lbe-zoom-indicator';
        _lightbox.appendChild(_zoomIndicator);
    }

    // ── Mouse wheel zoom ───────────────────────────────────
    function _onWheel(e) {
        if (!_lightbox.classList.contains('active')) return;
        e.preventDefault();
        e.stopPropagation();

        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        _zoomAtPoint(e.clientX, e.clientY, factor);
    }

    // ── Mouse drag (pan) ───────────────────────────────────
    function _onMouseDown(e) {
        if (!_lightbox.classList.contains('active')) return;
        if (_scale <= 1.01) return;
        if (e.button !== 0) return;
        // Don't intercept button clicks
        if (e.target.closest('.btn-icon, .lightbox-controls, .lightbox-close, .lightbox-nav, .face-overlay, .face-popup, .lightbox-info, .lightbox-tags-row')) return;

        e.preventDefault();
        _isDragging = true;
        _dragStartX = e.clientX;
        _dragStartY = e.clientY;
        _panStartX = _panX;
        _panStartY = _panY;
        _content.classList.add('lbe-dragging');
    }

    function _onMouseMove(e) {
        if (!_isDragging) return;
        e.preventDefault();
        _panX = _panStartX + (e.clientX - _dragStartX);
        _panY = _panStartY + (e.clientY - _dragStartY);
        _constrainPan();
        _applyTransform();
    }

    function _onMouseUp() {
        if (!_isDragging) return;
        _isDragging = false;
        _content.classList.remove('lbe-dragging');
    }

    // ── Double click: toggle 1:1 / fit ─────────────────────
    function _onDblClick(e) {
        if (!_lightbox.classList.contains('active')) return;
        if (e.target.closest('.btn-icon, .lightbox-controls, .lightbox-close, .lightbox-nav, .face-overlay, .face-popup, .lightbox-info, .lightbox-tags-row')) return;
        e.preventDefault();
        e.stopPropagation();

        if (_scale > 1.05) {
            // Reset to fit
            if (_isFullSize) {
                _fitToViewport();
            } else {
                _resetZoom();
                _applyTransform();
            }
        } else {
            // Zoom to ~100% (1:1 pixels) at cursor point
            const target = _isFullSize ? 1.0 : 2.0;
            const factor = target / _scale;
            _zoomAtPoint(e.clientX, e.clientY, factor);
        }
    }

    // ── Touch: pinch-to-zoom + pan ─────────────────────────
    function _getTouchDist(touches) {
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function _getTouchCenter(touches) {
        return {
            x: (touches[0].clientX + touches[1].clientX) / 2,
            y: (touches[0].clientY + touches[1].clientY) / 2
        };
    }

    function _onTouchStart(e) {
        if (!_lightbox.classList.contains('active')) return;
        if (e.touches.length === 2) {
            e.preventDefault();
            _pinchStartDist = _getTouchDist(e.touches);
            _pinchStartScale = _scale;
        } else if (e.touches.length === 1 && _scale > 1.01) {
            if (e.target.closest('.btn-icon, .lightbox-controls, .lightbox-close, .lightbox-nav, .face-overlay, .face-popup')) return;
            _isDragging = true;
            _dragStartX = e.touches[0].clientX;
            _dragStartY = e.touches[0].clientY;
            _panStartX = _panX;
            _panStartY = _panY;
        }
    }

    function _onTouchMove(e) {
        if (!_lightbox.classList.contains('active')) return;
        if (e.touches.length === 2 && _pinchStartDist > 0) {
            e.preventDefault();
            const dist = _getTouchDist(e.touches);
            const center = _getTouchCenter(e.touches);
            const newScale = _clamp(_pinchStartScale * (dist / _pinchStartDist), MIN_SCALE, MAX_SCALE);
            const ratio = newScale / _scale;
            const rect = _content.getBoundingClientRect();
            const ix = center.x - rect.left - _panX;
            const iy = center.y - rect.top - _panY;
            _panX = center.x - rect.left - ix * ratio;
            _panY = center.y - rect.top - iy * ratio;
            _scale = newScale;
            _constrainPan();
            _applyTransform();
        } else if (_isDragging && e.touches.length === 1) {
            e.preventDefault();
            _panX = _panStartX + (e.touches[0].clientX - _dragStartX);
            _panY = _panStartY + (e.touches[0].clientY - _dragStartY);
            _constrainPan();
            _applyTransform();
        }
    }

    function _onTouchEnd(e) {
        if (e.touches.length < 2) {
            _pinchStartDist = 0;
        }
        if (e.touches.length === 0) {
            _isDragging = false;
        }
    }

    // ── Keyboard handler ───────────────────────────────────
    function _onKeyDown(e) {
        if (!_lightbox.classList.contains('active')) return;
        var t = e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA') return;

        if (e.key === 'F11') {
            e.preventDefault();
            _toggleFullscreen();
            return;
        }

        // + / = zoom in, - zoom out, 0 reset
        if (e.key === '+' || e.key === '=') {
            e.preventDefault();
            const cx = _lightbox.clientWidth / 2;
            const cy = _lightbox.clientHeight / 2;
            _zoomAtPoint(cx, cy, 1.3);
        }
        if (e.key === '-') {
            e.preventDefault();
            const cx = _lightbox.clientWidth / 2;
            const cy = _lightbox.clientHeight / 2;
            _zoomAtPoint(cx, cy, 1 / 1.3);
        }
        if (e.key === '0') {
            e.preventDefault();
            if (_isFullSize) {
                _fitToViewport();
            } else {
                _resetZoom();
                _applyTransform();
            }
        }
    }

    // ── Observe image src changes to reset zoom ────────────
    function _watchImgSrc() {
        if (!_img) return;
        const observer = new MutationObserver((mutations) => {
            for (const m of mutations) {
                if (m.type === 'attributes' && m.attributeName === 'src') {
                    _onImageChanged();
                }
            }
        });
        observer.observe(_img, { attributes: true, attributeFilter: ['src'] });
    }

    function _onImageChanged() {
        _resetZoom();
        _isFullSize = false;
        if (_fullSizeBtn) {
            _fullSizeBtn.classList.remove('lbe-fullsize-active', 'lbe-fullsize-loading');
        }
    }

    // ── Prevent lightbox close when zoomed & panning ───────
    function _patchClickToClose() {
        // We intercept clicks on the lightbox/content to prevent close when zoomed
        _lightbox.addEventListener('click', function (e) {
            if (_scale > 1.01) {
                // When zoomed, don't close on click
                if (e.target === _lightbox || e.target === _content || e.target === _img) {
                    e.stopImmediatePropagation();
                }
            }
        }, true); // capture phase — runs before the page's click handler
    }

    // ── Public API ─────────────────────────────────────────
    function init(cfg) {
        _cfg = Object.assign({
            lightbox: '#lightbox',
            img: '#lightboxImg',
            content: '#lightboxContent',
            controls: '.lightbox-controls',
            getImageId: function () { return null; },
        }, cfg);

        _lightbox = document.querySelector(_cfg.lightbox);
        _img = document.querySelector(_cfg.img);
        _content = document.querySelector(_cfg.content);

        if (!_lightbox || !_img || !_content) {
            console.warn('LightboxEnhance: elements not found', _cfg);
            return;
        }

        _injectCSS();
        _createButtons();
        _watchImgSrc();
        _patchClickToClose();

        // Event listeners
        _content.addEventListener('wheel', _onWheel, { passive: false });
        _content.addEventListener('mousedown', _onMouseDown);
        document.addEventListener('mousemove', _onMouseMove);
        document.addEventListener('mouseup', _onMouseUp);
        _content.addEventListener('dblclick', _onDblClick);

        // Touch
        _content.addEventListener('touchstart', _onTouchStart, { passive: false });
        _content.addEventListener('touchmove', _onTouchMove, { passive: false });
        _content.addEventListener('touchend', _onTouchEnd);

        // Keyboard
        document.addEventListener('keydown', _onKeyDown);

        // Fullscreen change
        document.addEventListener('fullscreenchange', _onFullscreenChange);
        document.addEventListener('webkitfullscreenchange', _onFullscreenChange);
    }

    function resetZoom() {
        _resetZoom();
    }

    window.LightboxEnhance = {
        init: init,
        resetZoom: resetZoom
    };
})();
