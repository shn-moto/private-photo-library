/**
 * PersonSelector - reusable component for selecting persons (face filter)
 *
 * Usage:
 *   const selector = new PersonSelector({
 *     container: document.getElementById('myContainer'),
 *     onSelectionChange: (selectedPersons) => { ... },
 *   });
 *
 * selectedPersons: Array of { person_id, name, cover_face_id }
 */
class PersonSelector {
    constructor(options) {
        this.container = options.container;
        this.onSelectionChange = options.onSelectionChange || (() => {});
        this.selectedPersons = [];
        this.allPersons = [];
        this.isOpen = false;
        this.searchText = '';

        this.render();
        this.loadPersons();

        // Close dropdown on outside click
        this._onDocClick = (e) => {
            if (this.isOpen && !this.container.contains(e.target)) {
                this.closeDropdown();
            }
        };
        document.addEventListener('click', this._onDocClick);
    }

    async loadPersons() {
        try {
            const response = await fetch('/persons?limit=500');
            if (response.ok) {
                const data = await response.json();
                this.allPersons = (data.persons || []).filter(p => p.face_count > 0);
                this.preloadThumbnails();
            }
        } catch (err) {
            console.error('Failed to load persons:', err);
        }
    }

    preloadThumbnails() {
        for (const person of this.allPersons) {
            if (person.cover_face_id) {
                const img = new Image();
                img.src = `/faces/${person.cover_face_id}/thumb`;
            }
        }
    }

    render() {
        this.container.innerHTML = '';
        this.container.style.position = 'relative';
        this.container.style.display = 'inline-block';

        // Trigger button
        const btn = document.createElement('button');
        btn.className = 'btn btn-secondary person-selector-btn';
        btn.innerHTML = '<span class="ps-icon">&#128100;</span> <span class="ps-label">&#9660;</span>';
        btn.title = 'Фильтр по людям';
        btn.style.cssText = 'display:flex;align-items:center;gap:4px;padding:8px 12px;font-size:13px;cursor:pointer;';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleDropdown();
        });
        this.triggerBtn = btn;
        this.container.appendChild(btn);

        // Dropdown panel
        const dropdown = document.createElement('div');
        dropdown.className = 'person-selector-dropdown';
        dropdown.style.cssText = `
            display:none; position:absolute; top:100%; left:0; z-index:2000;
            background:#16213e; border:1px solid #333; border-radius:8px;
            box-shadow:0 8px 24px rgba(0,0,0,0.5); width:280px; max-height:400px;
            margin-top:4px; overflow:hidden;
        `;
        dropdown.addEventListener('click', (e) => e.stopPropagation());
        this.dropdown = dropdown;
        this.container.appendChild(dropdown);
    }

    toggleDropdown() {
        if (this.isOpen) {
            this.closeDropdown();
        } else {
            this.openDropdown();
        }
    }

    openDropdown() {
        this.isOpen = true;
        this.searchText = '';
        this.renderDropdownContent();
        this.dropdown.style.display = 'block';

        // Focus search input
        const input = this.dropdown.querySelector('.ps-search-input');
        if (input) setTimeout(() => input.focus(), 50);
    }

    closeDropdown() {
        this.isOpen = false;
        this.dropdown.style.display = 'none';
    }

    renderDropdownContent() {
        const filtered = this.searchText
            ? this.allPersons.filter(p =>
                p.name.toLowerCase().includes(this.searchText.toLowerCase()))
            : this.allPersons;

        const selectedIds = new Set(this.selectedPersons.map(p => p.person_id));

        let html = `
            <div style="padding:8px;">
                <input type="text" class="ps-search-input" placeholder="Поиск..."
                    value="${this.escapeHtml(this.searchText)}"
                    style="width:100%;padding:8px 10px;border:none;border-radius:6px;
                    background:#0f3460;color:#fff;font-size:13px;outline:none;">
            </div>
            <div class="ps-list" style="max-height:300px;overflow-y:auto;padding:0 4px 8px;">
        `;

        if (filtered.length === 0) {
            html += '<div style="padding:12px;color:#666;text-align:center;font-size:13px;">Нет персон</div>';
        } else {
            for (const person of filtered) {
                const isSelected = selectedIds.has(person.person_id);
                const thumbUrl = person.cover_face_id
                    ? `/faces/${person.cover_face_id}/thumb`
                    : '';
                const thumbStyle = thumbUrl
                    ? `background-image:url(${thumbUrl});background-size:cover;background-position:center;`
                    : 'background:#0f3460;';

                html += `
                    <div class="ps-item" data-person-id="${person.person_id}"
                        style="display:flex;align-items:center;gap:8px;padding:6px 8px;
                        border-radius:6px;cursor:pointer;transition:background 0.15s;
                        ${isSelected ? 'background:#0f3460;' : ''}">
                        <div style="width:36px;height:36px;border-radius:50%;flex-shrink:0;
                            border:2px solid ${isSelected ? '#4ade80' : '#333'};
                            ${thumbStyle}"></div>
                        <div style="flex:1;min-width:0;">
                            <div style="font-size:13px;color:#eee;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                                ${this.escapeHtml(person.name)}
                            </div>
                            <div style="font-size:11px;color:#666;">
                                ${person.face_count} лиц, ${person.photo_count} фото
                            </div>
                        </div>
                        ${isSelected ? '<span style="color:#4ade80;font-size:16px;">&#10003;</span>' : ''}
                    </div>
                `;
            }
        }

        html += '</div>';
        this.dropdown.innerHTML = html;

        // Search input handler
        const input = this.dropdown.querySelector('.ps-search-input');
        if (input) {
            input.addEventListener('input', (e) => {
                this.searchText = e.target.value;
                this.renderDropdownContent();
                // Re-focus the input
                const newInput = this.dropdown.querySelector('.ps-search-input');
                if (newInput) {
                    newInput.focus();
                    newInput.selectionStart = newInput.selectionEnd = newInput.value.length;
                }
            });
        }

        // Item click handlers
        this.dropdown.querySelectorAll('.ps-item').forEach(item => {
            item.addEventListener('mouseenter', () => {
                item.style.background = '#1a4a7a';
            });
            item.addEventListener('mouseleave', () => {
                const pid = parseInt(item.dataset.personId);
                const isSelected = selectedIds.has(pid);
                item.style.background = isSelected ? '#0f3460' : '';
            });
            item.addEventListener('click', () => {
                const pid = parseInt(item.dataset.personId);
                this.togglePerson(pid);
            });
        });
    }

    togglePerson(personId) {
        const idx = this.selectedPersons.findIndex(p => p.person_id === personId);
        if (idx >= 0) {
            this.selectedPersons.splice(idx, 1);
        } else {
            const person = this.allPersons.find(p => p.person_id === personId);
            if (person) {
                this.selectedPersons.push({
                    person_id: person.person_id,
                    name: person.name,
                    cover_face_id: person.cover_face_id
                });
            }
        }
        this.updateTriggerButton();
        this.renderDropdownContent();
        // Re-focus search
        const input = this.dropdown.querySelector('.ps-search-input');
        if (input) {
            input.focus();
            input.selectionStart = input.selectionEnd = input.value.length;
        }
        this.onSelectionChange(this.selectedPersons);
    }

    removePerson(personId) {
        this.selectedPersons = this.selectedPersons.filter(p => p.person_id !== personId);
        this.updateTriggerButton();
        if (this.isOpen) this.renderDropdownContent();
        this.onSelectionChange(this.selectedPersons);
    }

    clearSelection() {
        this.selectedPersons = [];
        this.updateTriggerButton();
        if (this.isOpen) this.renderDropdownContent();
        this.onSelectionChange(this.selectedPersons);
    }

    updateTriggerButton() {
        const count = this.selectedPersons.length;
        if (count > 0) {
            this.triggerBtn.innerHTML = `<span class="ps-icon">&#128100;</span> <span style="color:#4ade80;font-weight:bold;">${count}</span> <span class="ps-label">&#9660;</span>`;
            this.triggerBtn.style.background = '#0f3460';
            this.triggerBtn.style.borderColor = '#4ade80';
        } else {
            this.triggerBtn.innerHTML = '<span class="ps-icon">&#128100;</span> <span class="ps-label">&#9660;</span>';
            this.triggerBtn.style.background = '';
            this.triggerBtn.style.borderColor = '';
        }
    }

    getSelectedIds() {
        return this.selectedPersons.map(p => p.person_id);
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    destroy() {
        document.removeEventListener('click', this._onDocClick);
        this.container.innerHTML = '';
    }
}
