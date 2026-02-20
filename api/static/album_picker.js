/**
 * AlbumPicker - reusable modal for adding photos to albums
 *
 * Usage:
 *   const picker = new AlbumPicker({
 *     onAdd: (albumId, albumTitle, result) => { ... },
 *     onCancel: () => { ... }
 *   });
 *   picker.open([imageId1, imageId2, ...]);
 */
class AlbumPicker {
    constructor(options = {}) {
        this.onAdd = options.onAdd || (() => {});
        this.onCancel = options.onCancel || (() => {});
        this.albums = [];
        this.imageIds = [];
        this.searchText = '';
        this._buildModal();
        this._bindEvents();
    }

    _buildModal() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'album-picker-overlay';
        this.overlay.innerHTML = `
            <div class="album-picker-modal">
                <div class="album-picker-header">
                    <span class="album-picker-title">Добавить в альбом</span>
                    <button class="album-picker-close">&times;</button>
                </div>
                <div class="album-picker-search">
                    <input type="text" class="album-picker-search-input" placeholder="Поиск альбомов...">
                </div>
                <div class="album-picker-create">
                    <button class="album-picker-create-btn">+ Создать новый альбом</button>
                    <div class="album-picker-create-form" style="display:none;">
                        <input type="text" class="album-picker-new-title" placeholder="Название альбома..." maxlength="200">
                        <button class="album-picker-new-save">OK</button>
                    </div>
                </div>
                <div class="album-picker-list"></div>
                <div class="album-picker-loading">Загрузка...</div>
            </div>
        `;
        document.body.appendChild(this.overlay);

        this.modal = this.overlay.querySelector('.album-picker-modal');
        this.listEl = this.overlay.querySelector('.album-picker-list');
        this.loadingEl = this.overlay.querySelector('.album-picker-loading');
        this.searchInput = this.overlay.querySelector('.album-picker-search-input');
        this.createBtn = this.overlay.querySelector('.album-picker-create-btn');
        this.createForm = this.overlay.querySelector('.album-picker-create-form');
        this.newTitleInput = this.overlay.querySelector('.album-picker-new-title');
        this.newSaveBtn = this.overlay.querySelector('.album-picker-new-save');
    }

    _bindEvents() {
        // Close
        this.overlay.querySelector('.album-picker-close').addEventListener('click', () => this.close());
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.close();
        });

        // Search
        this.searchInput.addEventListener('input', () => {
            this.searchText = this.searchInput.value.trim().toLowerCase();
            this._renderList();
        });

        // Create new album
        this.createBtn.addEventListener('click', () => {
            this.createBtn.style.display = 'none';
            this.createForm.style.display = 'flex';
            this.newTitleInput.value = '';
            this.newTitleInput.focus();
        });

        this.newSaveBtn.addEventListener('click', () => this._createAndAdd());
        this.newTitleInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this._createAndAdd();
            if (e.key === 'Escape') this._hideCreateForm();
        });

        // Keyboard
        document.addEventListener('keydown', this._onKeyDown = (e) => {
            if (e.key === 'Escape' && this.overlay.classList.contains('active')) {
                this.close();
            }
        });
    }

    async open(imageIds) {
        this.imageIds = imageIds;
        this.searchText = '';
        this.searchInput.value = '';
        this._hideCreateForm();
        this.overlay.classList.add('active');
        this.listEl.innerHTML = '';
        this.loadingEl.style.display = 'block';

        try {
            await this._loadAlbums();
            this._renderList();
        } catch (err) {
            this.listEl.innerHTML = '<div class="album-picker-error">Ошибка загрузки альбомов</div>';
        } finally {
            this.loadingEl.style.display = 'none';
        }

        this.searchInput.focus();
    }

    close() {
        this.overlay.classList.remove('active');
        this.onCancel();
    }

    async _loadAlbums() {
        const resp = await fetch('/albums?limit=500');
        const data = await resp.json();
        this.albums = data.albums || [];
    }

    _renderList() {
        const filtered = this.albums.filter(a =>
            !this.searchText || a.title.toLowerCase().includes(this.searchText)
        );

        if (filtered.length === 0) {
            this.listEl.innerHTML = '<div class="album-picker-empty">Нет альбомов</div>';
            return;
        }

        this.listEl.innerHTML = filtered.map(album => `
            <div class="album-picker-item" data-album-id="${album.album_id}">
                <div class="album-picker-item-cover">
                    ${album.cover_image_id
                        ? `<img src="/image/${album.cover_image_id}/thumb?size=100" alt="">`
                        : '<div class="album-picker-item-nocover">&#128247;</div>'}
                </div>
                <div class="album-picker-item-info">
                    <div class="album-picker-item-title">${this._escapeHtml(album.title)}</div>
                    <div class="album-picker-item-count">${album.photo_count} фото</div>
                </div>
            </div>
        `).join('');

        // Click handlers
        this.listEl.querySelectorAll('.album-picker-item').forEach(el => {
            el.addEventListener('click', () => {
                const albumId = parseInt(el.dataset.albumId);
                const album = this.albums.find(a => a.album_id === albumId);
                this._addToAlbum(albumId, album ? album.title : '');
            });
        });
    }

    async _addToAlbum(albumId, albumTitle) {
        // Disable clicks
        this.listEl.style.pointerEvents = 'none';
        this.listEl.style.opacity = '0.5';

        try {
            const resp = await fetch(`/albums/${albumId}/photos`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image_ids: this.imageIds})
            });
            const result = await resp.json();

            this.overlay.classList.remove('active');
            this.onAdd(albumId, albumTitle, result);

            if (result.added > 0) {
                showToast(`Добавлено ${result.added} фото в "${albumTitle}"`);
            } else {
                showToast(`Все фото уже в альбоме "${albumTitle}"`);
            }
        } catch (err) {
            showToast('Ошибка добавления в альбом');
        } finally {
            this.listEl.style.pointerEvents = '';
            this.listEl.style.opacity = '';
        }
    }

    async _createAndAdd() {
        const title = this.newTitleInput.value.trim();
        if (!title) return;

        this.newSaveBtn.disabled = true;
        try {
            // Create album
            const resp = await fetch('/albums', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({title: title})
            });
            const data = await resp.json();
            const albumId = data.album_id;

            // Add photos
            await this._addToAlbum(albumId, title);
        } catch (err) {
            showToast('Ошибка создания альбома');
        } finally {
            this.newSaveBtn.disabled = false;
            this._hideCreateForm();
        }
    }

    _hideCreateForm() {
        this.createForm.style.display = 'none';
        this.createBtn.style.display = '';
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    destroy() {
        document.removeEventListener('keydown', this._onKeyDown);
        this.overlay.remove();
    }
}


// ==================== Toast Notification ====================

function showToast(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.textContent = message;
    document.body.appendChild(toast);
    // Trigger animation
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}


// ==================== Styles ====================

(function() {
    if (document.getElementById('album-picker-styles')) return;
    const style = document.createElement('style');
    style.id = 'album-picker-styles';
    style.textContent = `
        .album-picker-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            z-index: 5000;
            align-items: center;
            justify-content: center;
        }
        .album-picker-overlay.active {
            display: flex;
        }
        .album-picker-modal {
            background: #16213e;
            border-radius: 12px;
            width: 90%;
            max-width: 420px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }
        .album-picker-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .album-picker-title {
            font-size: 18px;
            font-weight: 600;
            color: #eee;
        }
        .album-picker-close {
            background: none;
            border: none;
            color: #888;
            font-size: 24px;
            cursor: pointer;
            padding: 0 4px;
            line-height: 1;
        }
        .album-picker-close:hover { color: #fff; }
        .album-picker-search {
            padding: 12px 16px 8px;
        }
        .album-picker-search-input {
            width: 100%;
            padding: 10px 14px;
            background: #0f3460;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #eee;
            font-size: 14px;
            outline: none;
        }
        .album-picker-search-input:focus {
            border-color: #e94560;
        }
        .album-picker-create {
            padding: 4px 16px 8px;
        }
        .album-picker-create-btn {
            width: 100%;
            padding: 10px;
            background: rgba(233,69,96,0.15);
            border: 1px dashed #e94560;
            border-radius: 8px;
            color: #e94560;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .album-picker-create-btn:hover {
            background: rgba(233,69,96,0.25);
        }
        .album-picker-create-form {
            display: flex;
            gap: 8px;
        }
        .album-picker-new-title {
            flex: 1;
            padding: 10px 14px;
            background: #0f3460;
            border: 1px solid #e94560;
            border-radius: 8px;
            color: #eee;
            font-size: 14px;
            outline: none;
        }
        .album-picker-new-save {
            padding: 10px 18px;
            background: #e94560;
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        }
        .album-picker-new-save:hover { background: #d63851; }
        .album-picker-new-save:disabled { opacity: 0.5; }
        .album-picker-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px 16px 16px;
        }
        .album-picker-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.15s;
        }
        .album-picker-item:hover {
            background: rgba(255,255,255,0.08);
        }
        .album-picker-item-cover {
            width: 48px;
            height: 48px;
            border-radius: 6px;
            overflow: hidden;
            background: #0f3460;
            flex-shrink: 0;
        }
        .album-picker-item-cover img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .album-picker-item-nocover {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: #555;
        }
        .album-picker-item-info {
            flex: 1;
            min-width: 0;
        }
        .album-picker-item-title {
            color: #eee;
            font-size: 14px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .album-picker-item-count {
            color: #888;
            font-size: 12px;
            margin-top: 2px;
        }
        .album-picker-loading, .album-picker-empty, .album-picker-error {
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 14px;
        }
        .album-picker-error { color: #e94560; }

        /* Toast */
        .toast-notification {
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%) translateY(20px);
            background: #4ade80;
            color: #000;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            opacity: 0;
            transition: opacity 0.3s, transform 0.3s;
            pointer-events: none;
            white-space: nowrap;
        }
        .toast-notification.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
    `;
    document.head.appendChild(style);
})();
