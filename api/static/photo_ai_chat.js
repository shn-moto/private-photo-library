/**
 * photo_ai_chat.js — AI Q&A chat about the current photo in lightbox.
 *
 * Uses Gemini Vision API (server-side) to answer questions about the photo:
 *   - "Что на фото?" — describe the image
 *   - "Где это?" — identify the location
 *   - "Что тут написано?" — OCR text recognition
 *   - "Переведи текст на русский" — translate visible text
 *   - Free-form follow-up questions
 *
 * window.PhotoAIChat API:
 *   init({ getImageId })    — initialize (pass function returning current lightbox image_id)
 *   open()                  — open chat panel
 *   close()                 — close chat panel
 *   onImageChanged()        — reset conversation when user navigates to another photo
 */
window.PhotoAIChat = (function () {
    'use strict';

    var _getImageId = null;
    var _panel = null;
    var _chatEl = null;
    var _inputEl = null;
    var _sendBtn = null;
    var _history = [];   // conversation_history for server
    var _imageId = null;  // current photo id
    var _loading = false;

    // ── CSS injection ────────────────────────────────────────────────────
    if (!document.getElementById('photo-ai-chat-styles')) {
        var s = document.createElement('style');
        s.id = 'photo-ai-chat-styles';
        s.textContent = '\
/* AI chat button in lightbox controls */\
.btn-icon.photo-ai-btn {\
    font-style: normal;\
    font-size: 16px;\
}\
\
/* Chat panel — slides in from right */\
.photo-ai-panel {\
    position: fixed;\
    top: 0; right: 0;\
    width: 380px;\
    max-width: 92vw;\
    height: 100vh;\
    background: #13152a;\
    border-left: 1px solid #333;\
    z-index: 1200;\
    display: flex;\
    flex-direction: column;\
    transform: translateX(100%);\
    transition: transform 0.25s ease;\
    box-shadow: -4px 0 24px rgba(0,0,0,0.5);\
}\
.photo-ai-panel.open {\
    transform: translateX(0);\
}\
\
/* Header */\
.photo-ai-header {\
    display: flex;\
    align-items: center;\
    justify-content: space-between;\
    padding: 12px 16px;\
    border-bottom: 1px solid #333;\
    background: #16213e;\
    flex-shrink: 0;\
}\
.photo-ai-header h3 {\
    margin: 0;\
    font-size: 15px;\
    color: #4ade80;\
    font-weight: 600;\
}\
.photo-ai-close {\
    width: 30px; height: 30px;\
    border-radius: 50%;\
    background: rgba(255,255,255,0.08);\
    border: none;\
    color: #aaa;\
    font-size: 18px;\
    cursor: pointer;\
    display: flex;\
    align-items: center;\
    justify-content: center;\
    transition: background 0.15s;\
}\
.photo-ai-close:hover { background: #e94560; color: #fff; }\
\
/* Chat messages area */\
.photo-ai-messages {\
    flex: 1;\
    overflow-y: auto;\
    padding: 12px 14px;\
    display: flex;\
    flex-direction: column;\
    gap: 10px;\
}\
\
/* Message bubble */\
.pai-msg {\
    display: flex;\
    gap: 8px;\
    max-width: 95%;\
}\
.pai-msg.user {\
    align-self: flex-end;\
    flex-direction: row-reverse;\
}\
.pai-msg.assistant {\
    align-self: flex-start;\
}\
.pai-avatar {\
    width: 28px; height: 28px;\
    border-radius: 50%;\
    display: flex;\
    align-items: center;\
    justify-content: center;\
    font-size: 14px;\
    flex-shrink: 0;\
}\
.pai-msg.user .pai-avatar {\
    background: rgba(59,130,246,0.2);\
}\
.pai-msg.assistant .pai-avatar {\
    background: rgba(74,222,128,0.2);\
}\
.pai-bubble {\
    padding: 8px 12px;\
    border-radius: 12px;\
    font-size: 13px;\
    line-height: 1.5;\
    word-break: break-word;\
    white-space: pre-wrap;\
}\
.pai-msg.user .pai-bubble {\
    background: rgba(59,130,246,0.15);\
    color: #93c5fd;\
    border-bottom-right-radius: 4px;\
}\
.pai-msg.assistant .pai-bubble {\
    background: rgba(74,222,128,0.1);\
    color: #d1d5db;\
    border-bottom-left-radius: 4px;\
}\
\
/* Loading dots */\
.pai-dots {\
    display: flex;\
    gap: 4px;\
    padding: 8px 12px;\
}\
.pai-dots span {\
    width: 8px; height: 8px;\
    border-radius: 50%;\
    background: #4ade80;\
    animation: paiPulse 1.2s ease-in-out infinite;\
}\
.pai-dots span:nth-child(2) { animation-delay: 0.2s; }\
.pai-dots span:nth-child(3) { animation-delay: 0.4s; }\
@keyframes paiPulse {\
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }\
    40% { opacity: 1; transform: scale(1.1); }\
}\
\
/* Quick chips */\
.pai-chips {\
    display: flex;\
    flex-wrap: wrap;\
    gap: 6px;\
    padding: 8px 14px;\
    border-bottom: 1px solid #222;\
    flex-shrink: 0;\
}\
.pai-chip {\
    padding: 5px 10px;\
    border-radius: 14px;\
    background: rgba(74,222,128,0.1);\
    border: 1px solid rgba(74,222,128,0.2);\
    color: #86efac;\
    font-size: 11px;\
    cursor: pointer;\
    transition: background 0.15s;\
    white-space: nowrap;\
}\
.pai-chip:hover {\
    background: rgba(74,222,128,0.25);\
}\
\
/* Input area */\
.pai-input-area {\
    display: flex;\
    gap: 8px;\
    padding: 10px 14px;\
    border-top: 1px solid #333;\
    background: #16213e;\
    flex-shrink: 0;\
}\
.pai-input {\
    flex: 1;\
    background: #1a1a3e;\
    border: 1px solid #333;\
    border-radius: 8px;\
    padding: 8px 12px;\
    color: #e0e0e0;\
    font-size: 13px;\
    resize: none;\
    font-family: inherit;\
    outline: none;\
    min-height: 20px;\
    max-height: 80px;\
}\
.pai-input:focus { border-color: #4ade80; }\
.pai-input::placeholder { color: #555; }\
.pai-send {\
    padding: 0 14px;\
    border-radius: 8px;\
    background: #22c55e;\
    border: none;\
    color: #000;\
    font-weight: 600;\
    font-size: 13px;\
    cursor: pointer;\
    transition: background 0.15s;\
    white-space: nowrap;\
}\
.pai-send:hover { background: #16a34a; }\
.pai-send:disabled { background: #333; color: #666; cursor: default; }\
\
/* Mobile */\
@media (max-width: 500px) {\
    .photo-ai-panel { width: 100vw; max-width: 100vw; }\
}\
';
        document.head.appendChild(s);
    }

    // ── Build panel DOM ──────────────────────────────────────────────────

    function _createPanel() {
        if (_panel) return;

        var div = document.createElement('div');
        div.className = 'photo-ai-panel';
        div.id = 'photoAIPanel';
        div.innerHTML =
            '<div class="photo-ai-header">' +
                '<h3>✨ AI Ассистент</h3>' +
                '<button class="photo-ai-close" title="Закрыть">&times;</button>' +
            '</div>' +
            '<div class="pai-chips">' +
                '<span class="pai-chip" data-q="Что на фото?">🔍 Что на фото?</span>' +
                '<span class="pai-chip" data-q="Где это?">📍 Где это?</span>' +
                '<span class="pai-chip" data-q="Что тут написано?">📝 Что тут написано?</span>' +
                '<span class="pai-chip" data-q="Переведи текст">🌐 Переведи текст</span>' +
            '</div>' +
            '<div class="photo-ai-messages" id="paiMessages"></div>' +
            '<div class="pai-input-area">' +
                '<textarea class="pai-input" id="paiInput" placeholder="Спросите о фото..." rows="1"></textarea>' +
                '<button class="pai-send" id="paiSend">➤</button>' +
            '</div>';

        document.body.appendChild(div);
        _panel = div;

        _chatEl = div.querySelector('#paiMessages');
        _inputEl = div.querySelector('#paiInput');
        _sendBtn = div.querySelector('#paiSend');

        // Close button
        div.querySelector('.photo-ai-close').onclick = close;

        // Send button
        _sendBtn.onclick = function () { _send(); };

        // Enter to send (Shift+Enter for newline)
        _inputEl.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                _send();
            }
        });

        // Auto-resize textarea
        _inputEl.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 80) + 'px';
        });

        // Chips
        div.querySelectorAll('.pai-chip').forEach(function (chip) {
            chip.onclick = function () {
                var q = chip.getAttribute('data-q');
                _inputEl.value = q;
                _send();
            };
        });

        // Escape when focused in input — close panel
        _inputEl.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                close();
                return;
            }
        });
    }

    // ── Message rendering ────────────────────────────────────────────────

    function _addMsg(role, text) {
        var msg = document.createElement('div');
        msg.className = 'pai-msg ' + role;

        var avatar = document.createElement('div');
        avatar.className = 'pai-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🤖';

        var bubble = document.createElement('div');
        bubble.className = 'pai-bubble';
        bubble.textContent = text;

        msg.appendChild(avatar);
        msg.appendChild(bubble);
        _chatEl.appendChild(msg);
        _chatEl.scrollTop = _chatEl.scrollHeight;
    }

    function _showDots() {
        var dots = document.createElement('div');
        dots.className = 'pai-msg assistant';
        dots.id = 'paiDotsMsg';

        var avatar = document.createElement('div');
        avatar.className = 'pai-avatar';
        avatar.textContent = '🤖';

        var dotsWrap = document.createElement('div');
        dotsWrap.className = 'pai-dots';
        dotsWrap.innerHTML = '<span></span><span></span><span></span>';

        dots.appendChild(avatar);
        dots.appendChild(dotsWrap);
        _chatEl.appendChild(dots);
        _chatEl.scrollTop = _chatEl.scrollHeight;
    }

    function _hideDots() {
        var d = document.getElementById('paiDotsMsg');
        if (d) d.remove();
    }

    // ── Send message ─────────────────────────────────────────────────────

    async function _send() {
        if (_loading) return;
        var text = (_inputEl.value || '').trim();
        if (!text) return;

        var imageId = _getImageId ? _getImageId() : null;
        if (!imageId) return;

        // Reset history if image changed
        if (imageId !== _imageId) {
            _imageId = imageId;
            _history = [];
            _chatEl.innerHTML = '';
        }

        _inputEl.value = '';
        _inputEl.style.height = 'auto';
        _addMsg('user', text);

        _loading = true;
        _sendBtn.disabled = true;
        _showDots();

        try {
            var resp = await fetch('/ai/photo-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: imageId,
                    message: text,
                    conversation_history: _history
                })
            });

            _hideDots();

            if (!resp.ok) {
                var errData = {};
                try { errData = await resp.json(); } catch (_) {}
                var errMsg = errData.detail || ('Ошибка: ' + resp.status);
                _addMsg('assistant', '\u274C ' + errMsg);
                return;
            }

            var data = await resp.json();
            _addMsg('assistant', data.message || '(пустой ответ)');
            _history = data.conversation_history || _history;

        } catch (err) {
            _hideDots();
            _addMsg('assistant', '\u274C Ошибка связи: ' + err.message);
        } finally {
            _loading = false;
            _sendBtn.disabled = false;
            _inputEl.focus();
        }
    }

    // ── Public API ───────────────────────────────────────────────────────

    function init(opts) {
        _getImageId = opts.getImageId;
        _createPanel();
    }

    function open() {
        if (!_panel) _createPanel();

        var imageId = _getImageId ? _getImageId() : null;
        // If image changed, reset chat
        if (imageId && imageId !== _imageId) {
            _imageId = imageId;
            _history = [];
            _chatEl.innerHTML = '';
        }

        _panel.classList.add('open');
        setTimeout(function () { _inputEl.focus(); }, 300);
    }

    function close() {
        if (_panel) _panel.classList.remove('open');
    }

    function onImageChanged() {
        // Will be detected on next send — just close panel if open
        // This avoids clearing chat when user accidentally swipes
    }

    return {
        init: init,
        open: open,
        close: close,
        onImageChanged: onImageChanged
    };
})();
