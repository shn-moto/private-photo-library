/**
 * ai_helper.js — AI assistant via server-side Gemini API.
 *
 * Thin wrapper over server endpoints:
 *   POST /ai/assistant       — map page assistant
 *   POST /ai/search-assistant — search page assistant
 *   POST /ai/clip-prompt     — CLIP prompt optimization
 *
 * Usage:
 *   <script src="/static/ai_helper.js"></script>
 *
 *   const result = await AIHelper.mapAssistant(message, conversationHistory, currentState);
 *   const result = await AIHelper.searchAssistant(message, conversationHistory, currentState);
 *   const result = await AIHelper.optimizeClipPrompt(query, model);
 */
var AIHelper = (function () {
    'use strict';

    // ==================== Helper ====================

    async function _postJSON(url, body) {
        var resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!resp.ok) {
            var detail = '';
            try {
                var err = await resp.json();
                detail = err.detail || JSON.stringify(err);
            } catch (e) {
                detail = resp.statusText;
            }
            throw new Error(detail || ('HTTP ' + resp.status));
        }
        return resp.json();
    }

    // ==================== Public API ====================

    async function mapAssistant(message, conversationHistory, currentState) {
        return _postJSON('/ai/assistant', {
            message: message,
            conversation_history: conversationHistory || [],
            current_state: currentState || {}
        });
    }

    async function searchAssistant(message, conversationHistory, currentState) {
        return _postJSON('/ai/search-assistant', {
            message: message,
            conversation_history: conversationHistory || [],
            current_state: currentState || {}
        });
    }

    async function optimizeClipPrompt(query, modelName) {
        try {
            var result = await _postJSON('/ai/clip-prompt', {
                query: query,
                model: modelName || null
            });
            if (result && result.clip_prompt) {
                console.log('CLIP prompt optimized:', query, '→', result.clip_prompt);
                return result;
            }
        } catch (e) {
            console.warn('CLIP prompt optimization failed, using original:', e);
        }
        return { clip_prompt: query, original_query: query };
    }

    // Stubs for backward compatibility (context is managed server-side now)
    function loadContext() { return Promise.resolve({ persons: [], tags: [] }); }
    function invalidateContext() {}

    return {
        loadContext: loadContext,
        invalidateContext: invalidateContext,
        optimizeClipPrompt: optimizeClipPrompt,
        mapAssistant: mapAssistant,
        searchAssistant: searchAssistant
    };
})();
