/**
 * tag_manager.js — Reusable tag management component
 * Follows the same pattern as album_picker.js, person_selector.js
 *
 * window.TagManager API:
 *   renderTagDots(itemEl, tags)                                    — thumbnail text pills
 *   openBulkModal(imageIds, { isAdmin, onClose, onApplied })       — bulk add/remove modal
 *   renderLightboxTags(el, tags, imageId, isAdmin, onChanged)      — lightbox pills with × and +
 *   loadPhotoTags(imageId) → Promise<tags[]>                       — fetch from API
 *   openTagPicker(imageId, anchorEl, currentTagIds, onApplied)     — inline dropdown picker
 *   closeTagPicker()
 *   invalidateCache()                                              — clear tag list cache
 */
window.TagManager = (() => {
    'use strict';

    // ── CSS injection (all tag-related styles in one place) ──────────────
    if (!document.getElementById('tag-manager-styles')) {
        const s = document.createElement('style');
        s.id = 'tag-manager-styles';
        s.textContent = `
/* Tag text pills on thumbnails */
.photo-tag-dots {
    position: absolute; bottom: 3px; right: 3px;
    display: flex; gap: 2px; flex-direction: column; align-items: flex-end;
    max-height: calc(100% - 28px); overflow: hidden; pointer-events: none;
}
.photo-tag-thumb {
    display: inline-block; padding: 1px 5px; border-radius: 3px;
    font-size: 9px; font-weight: 700; color: #fff;
    white-space: nowrap; max-width: 76px;
    overflow: hidden; text-overflow: ellipsis;
    text-shadow: 0 1px 2px rgba(0,0,0,0.6); letter-spacing: 0.2px;
}
/* Lightbox tag row */
.lightbox-tags-row {
    position: absolute; bottom: 50px; left: 50%;
    transform: translateX(-50%);
    display: flex; gap: 5px; flex-wrap: wrap;
    justify-content: center; max-width: 90vw;
    pointer-events: auto; z-index: 15;
}
.photo-tag-pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 8px 3px 8px; border-radius: 12px;
    font-size: 12px; color: #fff; cursor: default; white-space: nowrap;
}
.photo-tag-pill .remove-tag-btn {
    cursor: pointer; opacity: 0.6; margin-left: 2px;
    font-size: 15px; line-height: 1; transition: opacity 0.15s;
    flex-shrink: 0;
}
.photo-tag-pill .remove-tag-btn:hover { opacity: 1; color: #ff6b6b; }
/* Tag picker dropdown */
.tag-picker-popup {
    position: fixed; background: #16213e;
    border: 1px solid #2a3a6e; border-radius: 8px; padding: 6px;
    z-index: 9999; box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    min-width: 180px; max-height: 260px; overflow-y: auto;
}
.tag-picker-item {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 10px; cursor: pointer;
    border-radius: 4px; font-size: 13px; color: #eee;
    transition: background 0.15s;
}
.tag-picker-item:hover { background: rgba(255,255,255,0.08); }
`;
        document.head.appendChild(s);
    }

    // ── Helpers ──────────────────────────────────────────────────────────
    function _esc(s) {
        return String(s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // ── Tag list cache ────────────────────────────────────────────────────
    let _cache = null;
    function invalidateCache() { _cache = null; }

    async function _fetchTags() {
        if (_cache) return _cache;
        const r = await fetch('/tags');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        _cache = await r.json();
        return _cache;
    }

    // ── Thumbnail tag pills ───────────────────────────────────────────────
    function renderTagDots(itemEl, tags) {
        const old = itemEl.querySelector('.photo-tag-dots');
        if (old) old.remove();
        if (!tags || tags.length === 0) return;
        const wrap = document.createElement('div');
        wrap.className = 'photo-tag-dots';
        tags.slice(0, 5).forEach(t => {
            const pill = document.createElement('span');
            pill.className = 'photo-tag-thumb';
            pill.style.background = t.color + 'cc';
            pill.textContent = t.name;
            wrap.appendChild(pill);
        });
        // Insert before select-checkbox so it stays on top
        const chk = itemEl.querySelector('.select-checkbox');
        if (chk) itemEl.insertBefore(wrap, chk);
        else itemEl.appendChild(wrap);
    }

    // ── Bulk tag modal (Add / Remove buttons) ─────────────────────────────
    function openBulkModal(imageIds, { isAdmin = false, onClose, onApplied } = {}) {
        _fetchTags().then(allTags => {
            const old = document.getElementById('tagMgrModal');
            if (old) old.remove();

            const userTags = allTags.filter(t => !t.is_system);
            const sysTags  = isAdmin ? allTags.filter(t => t.is_system) : [];

            let listHtml = userTags.map(_tagCheckRow).join('');
            if (sysTags.length) {
                listHtml += '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:6px 0;">';
                listHtml += sysTags.map(_tagCheckRow).join('');
            }

            const modal = document.createElement('div');
            modal.id = 'tagMgrModal';
            modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:5000;display:flex;align-items:center;justify-content:center;';
            modal.innerHTML = `
                <div style="background:#16213e;border-radius:12px;padding:24px;max-width:420px;width:90%;max-height:70vh;overflow-y:auto;">
                    <div style="font-size:15px;font-weight:600;color:#eee;margin-bottom:14px;">🏷 Теги для ${imageIds.length} фото</div>
                    <div id="tagMgrList" style="border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:8px;max-height:300px;overflow-y:auto;margin-bottom:16px;">
                        ${listHtml || '<div style="color:#666;padding:6px 2px;">Нет доступных тегов</div>'}
                    </div>
                    <div style="display:flex;gap:8px;justify-content:flex-end;">
                        <button id="tagMgrCancel" style="background:#0f3460;color:#eee;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">Отмена</button>
                        <button id="tagMgrAdd"    style="background:#e94560;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">Добавить</button>
                        <button id="tagMgrRemove" style="background:#555;color:#eee;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">Снять</button>
                    </div>
                </div>`;
            document.body.appendChild(modal);

            // Add "create new tag" row inside tag list
            const tagList = modal.querySelector('#tagMgrList');
            tagList.appendChild(_createNewTagRow(newTag => {
                // Add new tag as checked checkbox row and keep modal open
                const row = document.createElement('div');
                row.innerHTML = _tagCheckRow(newTag);
                const label = row.firstElementChild;
                if (label) {
                    const cb = label.querySelector('input[type=checkbox]');
                    if (cb) cb.checked = true;
                    // Insert before the create-tag row
                    const createRow = tagList.lastElementChild;
                    tagList.insertBefore(label, createRow);
                }
            }));

            const close = () => { modal.remove(); if (onClose) onClose(); };
            modal.querySelector('#tagMgrCancel').addEventListener('click', close);
            modal.addEventListener('click', e => { if (e.target === modal) close(); });

            const apply = async (mode) => {
                const checked = [...modal.querySelectorAll('input[type=checkbox]:checked')];
                const tagIds = checked.map(cb => parseInt(cb.value));
                const tagObjects = checked.map(cb => ({
                    tag_id: parseInt(cb.value),
                    name: cb.dataset.name || '',
                    color: cb.dataset.color || '#6b7280',
                    is_system: cb.dataset.system === '1'
                }));
                modal.remove();
                if (onClose) onClose();
                if (!tagIds.length) return;
                try {
                    const r = await fetch('/photos/tags/bulk', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_ids: imageIds, tag_ids: tagIds, mode })
                    });
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    if (onApplied) onApplied({ imageIds, tagIds, tagObjects, mode });
                } catch (e) { alert('Ошибка: ' + e.message); }
            };

            modal.querySelector('#tagMgrAdd').addEventListener('click',    () => apply('add'));
            modal.querySelector('#tagMgrRemove').addEventListener('click', () => apply('remove'));
        }).catch(e => alert('Ошибка загрузки тегов: ' + e.message));
    }

    function _tagCheckRow(t) {
        return `<label style="display:flex;align-items:center;gap:8px;padding:4px 2px;cursor:pointer;font-size:13px;color:#eee;">
            <input type="checkbox" value="${t.tag_id}" data-name="${_esc(t.name)}" data-color="${_esc(t.color)}" data-system="${t.is_system ? '1' : ''}">
            <span style="width:10px;height:10px;border-radius:50%;background:${_esc(t.color)};flex-shrink:0;"></span>
            ${_esc(t.name)}
            ${t.is_system ? '<span style="color:#666;font-size:11px;">🔒</span>' : ''}
        </label>`;
    }

    // ── Lightbox tag pills with × (remove) and + (add) ───────────────────
    function renderLightboxTags(containerEl, tags, imageId, isAdmin, onChanged) {
        containerEl.innerHTML = '';

        (tags || []).forEach(tag => {
            const pill = document.createElement('span');
            pill.className = 'photo-tag-pill';
            pill.style.background = tag.color + '22';
            pill.style.border = `1px solid ${tag.color}88`;

            const dot = document.createElement('span');
            dot.style.cssText = `width:8px;height:8px;border-radius:50%;background:${tag.color};display:inline-block;flex-shrink:0;`;
            pill.appendChild(dot);

            const nm = document.createElement('span');
            nm.textContent = tag.name;
            pill.appendChild(nm);

            // × remove button — admin can remove any tag; non-admin can remove user tags only
            const canRemove = isAdmin || !tag.is_system;
            if (canRemove) {
                const rm = document.createElement('span');
                rm.className = 'remove-tag-btn';
                rm.textContent = '×';
                rm.title = 'Снять тег';
                rm.addEventListener('click', async e => {
                    e.stopPropagation();
                    try {
                        const r = await fetch(`/photo/${imageId}/tags`, {
                            method: 'DELETE',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ tag_ids: [tag.tag_id] })
                        });
                        if (r.ok) {
                            const updated = await loadPhotoTags(imageId);
                            renderLightboxTags(containerEl, updated, imageId, isAdmin, onChanged);
                            if (onChanged) onChanged(imageId, updated);
                        }
                    } catch (err) { console.error('remove tag:', err); }
                });
                pill.appendChild(rm);
            }
            containerEl.appendChild(pill);
        });

        // + add button — available for all users
        const addBtn = document.createElement('button');
        addBtn.style.cssText = 'background:rgba(255,255,255,0.1);border:1px dashed rgba(255,255,255,0.3);color:#aaa;border-radius:10px;padding:2px 10px;font-size:13px;cursor:pointer;line-height:1.6;';
        addBtn.textContent = '+';
        addBtn.title = 'Добавить тег';
        const currentTagIds = (tags || []).map(t => t.tag_id);
        addBtn.addEventListener('click', e => {
            e.stopPropagation();
            openTagPicker(imageId, addBtn, currentTagIds, updatedTags => {
                renderLightboxTags(containerEl, updatedTags, imageId, isAdmin, onChanged);
                if (onChanged) onChanged(imageId, updatedTags);
            }, isAdmin);
        });
        containerEl.appendChild(addBtn);
    }

    // ── Load tags for a single photo ──────────────────────────────────────
    async function loadPhotoTags(imageId) {
        const r = await fetch(`/photo/${imageId}/tags`);
        if (!r.ok) return [];
        const data = await r.json();
        return Array.isArray(data) ? data : (data.tags || []);
    }

    // ── Inline tag picker dropdown ────────────────────────────────────────
    let _pickerEl = null;
    let _pickerCleanup = null;

    function closeTagPicker() {
        if (_pickerEl) { _pickerEl.remove(); _pickerEl = null; }
        if (_pickerCleanup) {
            document.removeEventListener('click', _pickerCleanup);
            _pickerCleanup = null;
        }
    }

    async function openTagPicker(imageId, anchorEl, currentTagIds, onApplied, isAdmin) {
        closeTagPicker();
        _pickerEl = document.createElement('div');
        _pickerEl.className = 'tag-picker-popup';

        try {
            const allTags = await _fetchTags();
            // Non-admin users only see user tags; admin sees all
            const visibleTags = isAdmin ? allTags : allTags.filter(t => !t.is_system);
            const available = visibleTags.filter(t => !currentTagIds.includes(t.tag_id));
            if (!available.length) {
                _pickerEl.innerHTML = '<div style="color:#888;padding:8px 10px;font-size:13px;">Все теги уже добавлены</div>';
            } else {
                available.forEach(tag => {
                    const item = document.createElement('div');
                    item.className = 'tag-picker-item';

                    const dot = document.createElement('span');
                    dot.style.cssText = `width:10px;height:10px;border-radius:50%;background:${tag.color};flex-shrink:0;`;
                    const nm = document.createElement('span');
                    nm.textContent = tag.name;
                    item.appendChild(dot);
                    item.appendChild(nm);
                    if (tag.is_system) {
                        const lk = document.createElement('span');
                        lk.style.cssText = 'opacity:0.5;font-size:10px;margin-left:auto;';
                        lk.textContent = '🔒';
                        item.appendChild(lk);
                    }

                    item.addEventListener('click', async e => {
                        e.stopPropagation();
                        closeTagPicker();
                        try {
                            const r = await fetch(`/photo/${imageId}/tags`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ tag_ids: [tag.tag_id] })
                            });
                            if (r.ok) {
                                const updated = await loadPhotoTags(imageId);
                                if (onApplied) onApplied(updated);
                            }
                        } catch (err) { console.error('add tag:', err); }
                    });
                    _pickerEl.appendChild(item);
                });
            }

            // "Create new tag" input row
            _pickerEl.appendChild(_createNewTagRow(async (newTag) => {
                closeTagPicker();
                // Assign the newly created tag to the photo
                try {
                    const r = await fetch(`/photo/${imageId}/tags`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ tag_ids: [newTag.tag_id] })
                    });
                    if (r.ok) {
                        const updated = await loadPhotoTags(imageId);
                        if (onApplied) onApplied(updated);
                    }
                } catch (err) { console.error('add new tag:', err); }
            }));
        } catch {
            _pickerEl.innerHTML = '<div style="color:#e94560;padding:8px 10px;font-size:12px;">Ошибка загрузки</div>';
        }

        document.body.appendChild(_pickerEl);

        // Position near anchor (avoid going off-screen)
        const rect = anchorEl.getBoundingClientRect();
        const spaceBelow = window.innerHeight - rect.bottom;
        if (spaceBelow < 200 && rect.top > 200) {
            _pickerEl.style.top  = 'auto';
            _pickerEl.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
        } else {
            _pickerEl.style.top = (rect.bottom + 4) + 'px';
        }
        _pickerEl.style.left = Math.min(rect.left, window.innerWidth - 200) + 'px';

        setTimeout(() => {
            _pickerCleanup = closeTagPicker;
            document.addEventListener('click', _pickerCleanup, { once: true });
        }, 0);
    }

    // ── Create new tag row (shared by picker and bulk modal) ─────────────
    const _TAG_COLORS = ['#ef4444','#f97316','#eab308','#22c55e','#06b6d4','#3b82f6','#8b5cf6','#ec4899','#6b7280'];

    function _createNewTagRow(onCreated) {
        const row = document.createElement('div');
        row.style.cssText = 'border-top:1px solid rgba(255,255,255,0.1);margin-top:4px;padding:6px 8px;';
        row.innerHTML = `
            <div style="display:flex;gap:6px;align-items:center;">
                <input type="text" placeholder="Новый тег..." maxlength="50"
                    style="flex:1;background:#0f3460;border:1px solid #2a3a6e;border-radius:6px;
                        padding:5px 8px;color:#eee;font-size:12px;outline:none;min-width:0;">
                <button style="background:#22c55e;color:#fff;border:none;border-radius:6px;
                    padding:5px 10px;font-size:12px;cursor:pointer;white-space:nowrap;flex-shrink:0;"
                    disabled>+</button>
            </div>`;
        const input = row.querySelector('input');
        const btn = row.querySelector('button');
        input.addEventListener('input', () => { btn.disabled = !input.value.trim(); });
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter' && input.value.trim()) { e.preventDefault(); e.stopPropagation(); btn.click(); }
        });
        // Stop click from closing picker
        row.addEventListener('click', e => e.stopPropagation());
        btn.addEventListener('click', async e => {
            e.stopPropagation();
            const name = input.value.trim();
            if (!name) return;
            btn.disabled = true;
            input.disabled = true;
            try {
                const color = _TAG_COLORS[Math.floor(Math.random() * _TAG_COLORS.length)];
                const r = await fetch('/tags', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, color })
                });
                if (!r.ok) {
                    const err = await r.json().catch(() => ({}));
                    alert(err.detail || 'Ошибка создания тега');
                    btn.disabled = false; input.disabled = false;
                    return;
                }
                const newTag = await r.json();
                invalidateCache();
                if (onCreated) onCreated(newTag);
            } catch (err) {
                alert('Ошибка: ' + err.message);
                btn.disabled = false; input.disabled = false;
            }
        });
        return row;
    }

    // ── Public API ────────────────────────────────────────────────────────
    return {
        renderTagDots,
        openBulkModal,
        renderLightboxTags,
        loadPhotoTags,
        openTagPicker,
        closeTagPicker,
        invalidateCache,
    };
})();
