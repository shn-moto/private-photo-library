/**
 * TagFilter — reusable tag filter component
 *
 * Usage:
 *   const tagFilter = new TagFilter(containerEl, {
 *     isAdmin: true,      // show system tags (private/trash/document)
 *     onChanged: () => {} // callback when selection changes
 *   });
 *
 *   tagFilter.getSelectedIds()  // → [1, 3]
 *   tagFilter.clearSelection()
 *
 * Pattern follows person_selector.js
 */
class TagFilter {
    constructor(container, options = {}) {
        this.container = container;
        this.isAdmin = options.isAdmin || false;
        this.onChanged = options.onChanged || null;
        this.tags = [];           // all loaded tags
        this.selectedIds = new Set();
        this._dropdownOpen = false;
        this._rendered = false;
        this._init();
    }

    async _init() {
        this._render();
        await this._loadTags();
    }

    _render() {
        this.container.innerHTML = `
            <div class="tag-filter-wrap" style="position:relative;display:inline-block;">
                <button class="btn btn-secondary tag-filter-btn" id="tagFilterBtn"
                        style="display:inline-flex;align-items:center;gap:5px;font-size:13px;padding:6px 10px;">
                    <span>🏷</span>
                    <span class="tag-filter-label">Теги</span>
                    <span class="tag-filter-count" style="display:none;background:#e94560;color:#fff;
                        border-radius:10px;padding:1px 6px;font-size:11px;font-weight:bold;"></span>
                    <span style="opacity:0.6;font-size:10px;">▾</span>
                </button>
                <div class="tag-filter-dropdown" style="display:none;position:absolute;top:calc(100% + 4px);
                    left:0;z-index:300;background:#16213e;border:1px solid #2a3a6e;border-radius:8px;
                    min-width:220px;max-height:320px;overflow-y:auto;box-shadow:0 8px 24px rgba(0,0,0,0.4);">
                    <div class="tag-filter-list"></div>
                    <div class="tag-filter-empty" style="padding:12px;color:#666;font-size:13px;text-align:center;">
                        Загрузка...
                    </div>
                </div>
            </div>`;

        this.btn = this.container.querySelector('.tag-filter-btn');
        this.dropdown = this.container.querySelector('.tag-filter-dropdown');
        this.list = this.container.querySelector('.tag-filter-list');
        this.emptyMsg = this.container.querySelector('.tag-filter-empty');
        this.countBadge = this.container.querySelector('.tag-filter-count');

        this.btn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._toggleDropdown();
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!this.container.contains(e.target)) {
                this._closeDropdown();
            }
        });
    }

    async _loadTags() {
        try {
            const resp = await fetch('/tags');
            if (!resp.ok) return;
            this.tags = await resp.json();
            this._renderList();
        } catch (e) {
            console.error('TagFilter: failed to load tags', e);
        }
    }

    _renderList() {
        if (!this.tags.length) {
            this.emptyMsg.textContent = 'Нет доступных тегов';
            return;
        }
        this.emptyMsg.style.display = 'none';

        // Separate system and user tags
        const systemTags = this.tags.filter(t => t.is_system);
        const userTags = this.tags.filter(t => !t.is_system);

        let html = '';

        // User tags first
        userTags.forEach(tag => {
            html += this._tagItemHtml(tag);
        });

        // System tags (admin only, with separator)
        if (this.isAdmin && systemTags.length > 0) {
            if (userTags.length > 0) {
                html += `<div style="border-top:1px solid #2a3a6e;margin:4px 0;"></div>`;
            }
            systemTags.forEach(tag => {
                html += this._tagItemHtml(tag, true);
            });
        }

        this.list.innerHTML = html;

        // Bind click handlers
        this.list.querySelectorAll('.tag-filter-item').forEach(el => {
            el.addEventListener('click', () => {
                const tagId = parseInt(el.dataset.tagId);
                this._toggleTag(tagId, el);
            });
        });
    }

    _tagItemHtml(tag, isSystem = false) {
        const selected = this.selectedIds.has(tag.tag_id);
        const sysIcon = isSystem ? '<span style="opacity:0.5;font-size:10px;">🔒</span>' : '';
        return `
            <div class="tag-filter-item" data-tag-id="${tag.tag_id}"
                style="display:flex;align-items:center;gap:8px;padding:7px 12px;cursor:pointer;
                    transition:background 0.15s;border-radius:4px;margin:2px 4px;
                    ${selected ? 'background:rgba(233,69,96,0.15);' : ''}">
                <span style="width:10px;height:10px;border-radius:50%;flex-shrink:0;
                    background:${tag.color};display:inline-block;"></span>
                <span style="flex:1;font-size:13px;color:#eee;">${this._esc(tag.name)}</span>
                ${sysIcon}
                ${selected ? '<span style="color:#4ade80;font-size:14px;">✓</span>' : ''}
            </div>`;
    }

    _toggleTag(tagId, el) {
        if (this.selectedIds.has(tagId)) {
            this.selectedIds.delete(tagId);
        } else {
            this.selectedIds.add(tagId);
        }
        // Re-render item
        const tag = this.tags.find(t => t.tag_id === tagId);
        if (tag) {
            const isSystem = tag.is_system;
            el.outerHTML = this._tagItemHtml(tag, isSystem);
            // Re-bind the new element
            const newEl = this.list.querySelector(`[data-tag-id="${tagId}"]`);
            if (newEl) {
                newEl.addEventListener('click', () => this._toggleTag(tagId, newEl));
            }
        }
        this._updateCount();
        if (this.onChanged) this.onChanged();
    }

    _updateCount() {
        const n = this.selectedIds.size;
        if (n > 0) {
            this.countBadge.textContent = n;
            this.countBadge.style.display = 'inline-block';
        } else {
            this.countBadge.style.display = 'none';
        }
    }

    _toggleDropdown() {
        if (this._dropdownOpen) {
            this._closeDropdown();
        } else {
            this._openDropdown();
        }
    }

    _openDropdown() {
        this.dropdown.style.display = 'block';
        this._dropdownOpen = true;
    }

    _closeDropdown() {
        this.dropdown.style.display = 'none';
        this._dropdownOpen = false;
    }

    getSelectedIds() {
        return Array.from(this.selectedIds);
    }

    getSelectedTags() {
        return this.tags.filter(t => this.selectedIds.has(t.tag_id));
    }

    clearSelection() {
        this.selectedIds.clear();
        this._renderList();
        this._updateCount();
    }

    /** Build chips HTML for selected tags (for external display) */
    renderChips(chipsContainer, onRemove) {
        if (!chipsContainer) return;
        const selected = this.getSelectedTags();
        if (!selected.length) {
            chipsContainer.innerHTML = '';
            return;
        }
        chipsContainer.innerHTML = selected.map(t => `
            <span class="tag-chip"
                style="display:inline-flex;align-items:center;gap:4px;padding:3px 8px 3px 6px;
                    border-radius:12px;font-size:12px;color:#fff;cursor:default;
                    background:${t.color}22;border:1px solid ${t.color}88;">
                <span style="width:8px;height:8px;border-radius:50%;background:${t.color};
                    flex-shrink:0;display:inline-block;"></span>
                ${this._esc(t.name)}
                <span class="tag-chip-x" data-id="${t.tag_id}"
                    style="cursor:pointer;opacity:0.7;margin-left:2px;font-size:13px;line-height:1;">&times;</span>
            </span>`).join('');

        chipsContainer.querySelectorAll('.tag-chip-x').forEach(x => {
            x.addEventListener('click', () => {
                const tagId = parseInt(x.dataset.id);
                this.selectedIds.delete(tagId);
                const el = this.list.querySelector(`[data-tag-id="${tagId}"]`);
                if (el) {
                    const tag = this.tags.find(t => t.tag_id === tagId);
                    if (tag) el.outerHTML = this._tagItemHtml(tag, tag.is_system);
                    const newEl = this.list.querySelector(`[data-tag-id="${tagId}"]`);
                    if (newEl) newEl.addEventListener('click', () => this._toggleTag(tagId, newEl));
                }
                this._updateCount();
                if (onRemove) onRemove(tagId);
                if (this.onChanged) this.onChanged();
                this.renderChips(chipsContainer, onRemove);
            });
        });
    }

    _esc(s) {
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    destroy() {
        this.container.innerHTML = '';
    }
}
