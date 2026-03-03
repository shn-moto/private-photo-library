/**
 * exif_info.js — Reusable EXIF/photo info component
 * Shows detailed photo metadata in a popup panel.
 *
 * window.ExifInfo API:
 *   show(imageId)           — fetch & display EXIF info popup for a photo
 *   hide()                  — close the popup
 *   renderBadge(cardEl, imageId) — add ℹ badge on a thumbnail card
 */
window.ExifInfo = (() => {
    'use strict';

    let _popup = null;
    let _currentId = null;
    let _cache = {};

    // ── CSS injection ────────────────────────────────────────────────────
    if (!document.getElementById('exif-info-styles')) {
        const s = document.createElement('style');
        s.id = 'exif-info-styles';
        s.textContent = `
/* ℹ badge on thumbnails */
.info-badge {
    position: absolute;
    bottom: 6px;
    right: 6px;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: rgba(0,0,0,0.55);
    color: #aec6ff;
    font-size: 13px;
    font-weight: 700;
    font-style: italic;
    font-family: Georgia, 'Times New Roman', serif;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 4;
    opacity: 0;
    transition: opacity 0.15s, background 0.15s;
    pointer-events: auto;
    line-height: 1;
    user-select: none;
}
.image-card:hover .info-badge,
.photo-card:hover .info-badge,
.timeline-photo:hover .info-badge { opacity: 1; }
.info-badge:hover { background: rgba(30,100,220,0.85); color: #fff; }

/* EXIF popup overlay */
.exif-popup-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.5);
    z-index: 2000;
    display: flex; align-items: center; justify-content: center;
    animation: exifFadeIn 0.15s;
}
@keyframes exifFadeIn { from { opacity: 0 } to { opacity: 1 } }

/* EXIF popup panel */
.exif-popup {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 0;
    max-width: 520px;
    width: 92vw;
    max-height: 80vh;
    overflow: hidden;
    display: flex; flex-direction: column;
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
    color: #e0e0e0;
    font-size: 13px;
}
.exif-popup-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid #333;
    background: #16213e;
    border-radius: 12px 12px 0 0;
}
.exif-popup-header h3 {
    margin: 0; font-size: 15px; color: #5dade2; font-weight: 600;
}
.exif-popup-close {
    width: 30px; height: 30px; border-radius: 50%;
    background: rgba(255,255,255,0.08); border: none;
    color: #aaa; font-size: 18px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s;
}
.exif-popup-close:hover { background: #e94560; color: #fff; }

.exif-popup-body {
    padding: 12px 18px 18px;
    overflow-y: auto;
    flex: 1;
}

/* Key-value table */
.exif-table {
    width: 100%;
    border-collapse: collapse;
}
.exif-table tr { border-bottom: 1px solid rgba(255,255,255,0.06); }
.exif-table tr:last-child { border-bottom: none; }
.exif-table td {
    padding: 6px 0;
    vertical-align: top;
}
.exif-table .exif-key {
    color: #888;
    white-space: nowrap;
    padding-right: 14px;
    font-size: 12px;
    width: 1%;
}
.exif-table .exif-val {
    color: #ddd;
    word-break: break-all;
    font-family: 'Consolas', 'SF Mono', monospace;
    font-size: 12px;
}
.exif-table .exif-val.highlight {
    color: #5dade2;
    font-weight: 600;
}

/* Section divider */
.exif-section td {
    color: #5dade2;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 0 4px;
    border-bottom: 1px solid rgba(93,173,226,0.2);
}
.exif-table tr:first-child.exif-section td { padding-top: 0; }

/* Loading state */
.exif-loading {
    text-align: center;
    padding: 30px;
    color: #888;
}
.exif-loading::after {
    content: '';
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid #444;
    border-top-color: #5dade2;
    border-radius: 50%;
    animation: exifSpin 0.6s linear infinite;
    vertical-align: middle;
    margin-left: 8px;
}
@keyframes exifSpin { to { transform: rotate(360deg); } }

/* Copy button */
.exif-copy-btn {
    background: rgba(93,173,226,0.15);
    border: 1px solid rgba(93,173,226,0.3);
    color: #5dade2;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    margin-left: 6px;
    transition: background 0.15s;
}
.exif-copy-btn:hover { background: rgba(93,173,226,0.3); }
`;
        document.head.appendChild(s);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    function _formatFileSize(bytes) {
        if (!bytes && bytes !== 0) return '—';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    }

    function _formatDate(val) {
        if (!val) return null;
        try {
            const d = new Date(val);
            if (isNaN(d.getTime())) return String(val);
            return d.toLocaleString('ru-RU', {
                day: '2-digit', month: '2-digit', year: 'numeric',
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            });
        } catch (_) { return String(val); }
    }

    function _copyText(text) {
        navigator.clipboard.writeText(text).then(() => {
            _showToast('Скопировано');
        }).catch(() => {});
    }

    function _showToast(msg) {
        const t = document.createElement('div');
        t.textContent = msg;
        Object.assign(t.style, {
            position: 'fixed', bottom: '80px', left: '50%', transform: 'translateX(-50%)',
            background: '#22c55e', color: '#000', padding: '6px 18px', borderRadius: '6px',
            fontWeight: '600', fontSize: '13px', zIndex: '3000',
            animation: 'exifFadeIn 0.15s'
        });
        document.body.appendChild(t);
        setTimeout(() => t.remove(), 1500);
    }

    // ── Row builders ─────────────────────────────────────────────────────

    function _row(key, val, cls) {
        if (val === null || val === undefined || val === '') return '';
        return `<tr><td class="exif-key">${key}</td><td class="exif-val${cls ? ' ' + cls : ''}">${val}</td></tr>`;
    }

    function _rowCopy(key, val, copyVal) {
        if (val === null || val === undefined || val === '') return '';
        return `<tr><td class="exif-key">${key}</td><td class="exif-val highlight">${val}<button class="exif-copy-btn" onclick="event.stopPropagation();ExifInfo._copy('${(copyVal || val).replace(/'/g, "\\'")}')">📋</button></td></tr>`;
    }

    function _section(title) {
        return `<tr class="exif-section"><td colspan="2">${title}</td></tr>`;
    }

    // ── Build HTML from photo data ───────────────────────────────────────

    function _buildContent(d) {
        const exif = d.exif_data || {};
        let html = '<table class="exif-table">';

        // ── File info ─────────────────────────
        html += _section('📄 Файл');
        html += _row('Имя', d.file_name);
        html += _row('Путь', d.file_path);
        html += _row('Формат', (d.file_format || '').toUpperCase());
        html += _row('Размер', _formatFileSize(d.file_size));
        html += _row('Разрешение', d.width && d.height ? `${d.width} × ${d.height}` : null);

        // ── Dates ─────────────────────────────
        html += _section('📅 Даты');
        html += _row('Дата снимка', _formatDate(d.photo_date));
        // EXIF dates
        const exifDateOrig = exif['DateTimeOriginal'] || exif['EXIF DateTimeOriginal'];
        const exifDateDig = exif['DateTimeDigitized'] || exif['EXIF DateTimeDigitized'];
        const exifDateTime = exif['DateTime'] || exif['Image DateTime'];
        if (exifDateOrig) html += _row('EXIF Original', exifDateOrig);
        if (exifDateDig && exifDateDig !== exifDateOrig) html += _row('EXIF Digitized', exifDateDig);
        if (exifDateTime && exifDateTime !== exifDateOrig) html += _row('EXIF DateTime', exifDateTime);

        // ── GPS ───────────────────────────────
        if (d.latitude && d.longitude) {
            html += _section('📍 GPS');
            const coords = `${d.latitude.toFixed(6)}, ${d.longitude.toFixed(6)}`;
            html += _rowCopy('Координаты', coords, coords);

            // GPS altitude from EXIF
            const alt = exif['GPSAltitude'] || exif['GPS GPSAltitude'];
            if (alt) html += _row('Высота', alt);
        }

        // ── Camera ────────────────────────────
        const make = exif['Make'] || exif['Image Make'];
        const model = exif['Model'] || exif['Image Model'];
        const lens = exif['LensModel'] || exif['EXIF LensModel'] || exif['LensInfo'];
        const focalLength = exif['FocalLength'] || exif['EXIF FocalLength'];
        const focalLength35 = exif['FocalLengthIn35mmFilm'] || exif['EXIF FocalLengthIn35mmFilm'];
        const fNumber = exif['FNumber'] || exif['EXIF FNumber'];
        const exposure = exif['ExposureTime'] || exif['EXIF ExposureTime'];
        const iso = exif['ISOSpeedRatings'] || exif['EXIF ISOSpeedRatings'] || exif['ISO'];
        const wb = exif['WhiteBalance'] || exif['EXIF WhiteBalance'];
        const flash = exif['Flash'] || exif['EXIF Flash'];
        const expProg = exif['ExposureProgram'] || exif['EXIF ExposureProgram'];
        const meter = exif['MeteringMode'] || exif['EXIF MeteringMode'];
        const expBias = exif['ExposureBiasValue'] || exif['EXIF ExposureBiasValue'];
        const software = exif['Software'] || exif['Image Software'];

        const hasCamera = make || model || lens || focalLength || fNumber || exposure || iso;
        if (hasCamera) {
            html += _section('📷 Камера');
            if (make || model) {
                const cam = [make, model].filter(Boolean).join(' ');
                html += _row('Камера', cam);
            }
            if (lens) html += _row('Объектив', lens);
            if (focalLength) {
                let fl = String(focalLength);
                if (focalLength35 && String(focalLength35) !== fl) fl += ` (${focalLength35} mm 35eq)`;
                html += _row('Фокус. расст.', fl);
            }
            if (fNumber) html += _row('Диафрагма', 'f/' + fNumber);
            if (exposure) html += _row('Выдержка', exposure + ' сек');
            if (iso) html += _row('ISO', iso);
            if (expBias) html += _row('Экспокоррекция', expBias);
            if (expProg) html += _row('Режим', expProg);
            if (meter) html += _row('Замер', meter);
            if (wb) html += _row('Баланс белого', wb);
            if (flash) html += _row('Вспышка', flash);
            if (software) html += _row('Софт', software);
        }

        // ── Rotation ──────────────────────────
        const userRot = (d.exif_data || {}).UserRotation;
        const exifOr = exif['Orientation'] || exif['Image Orientation'];
        if (userRot || exifOr) {
            html += _section('🔄 Ориентация');
            if (exifOr) html += _row('EXIF Orient.', exifOr);
            if (userRot) html += _row('Поворот', userRot + '°');
        }

        // ── All other EXIF fields ─────────────
        // Collect keys already shown
        const shownKeys = new Set([
            'DateTimeOriginal', 'EXIF DateTimeOriginal', 'DateTimeDigitized', 'EXIF DateTimeDigitized',
            'DateTime', 'Image DateTime',
            'GPSAltitude', 'GPS GPSAltitude', 'GPSLatitude', 'GPSLatitudeRef',
            'GPSLongitude', 'GPSLongitudeRef', 'GPS GPSLatitude', 'GPS GPSLatitudeRef',
            'GPS GPSLongitude', 'GPS GPSLongitudeRef',
            'Make', 'Image Make', 'Model', 'Image Model',
            'LensModel', 'EXIF LensModel', 'LensInfo',
            'FocalLength', 'EXIF FocalLength', 'FocalLengthIn35mmFilm', 'EXIF FocalLengthIn35mmFilm',
            'FNumber', 'EXIF FNumber', 'ExposureTime', 'EXIF ExposureTime',
            'ISOSpeedRatings', 'EXIF ISOSpeedRatings', 'ISO',
            'WhiteBalance', 'EXIF WhiteBalance', 'Flash', 'EXIF Flash',
            'ExposureProgram', 'EXIF ExposureProgram', 'MeteringMode', 'EXIF MeteringMode',
            'ExposureBiasValue', 'EXIF ExposureBiasValue',
            'Software', 'Image Software',
            'Orientation', 'Image Orientation',
            'UserRotation'
        ]);

        const otherKeys = Object.keys(exif)
            .filter(k => !shownKeys.has(k) && !k.startsWith('GPS') && !k.startsWith('Thumbnail'));
        if (otherKeys.length > 0) {
            html += _section('🔧 Прочее');
            otherKeys.sort().forEach(k => {
                const v = String(exif[k]);
                if (v && v.length <= 200) html += _row(k, v);
            });
        }

        html += '</table>';
        return html;
    }

    // ── Show popup ───────────────────────────────────────────────────────

    async function show(imageId) {
        _currentId = imageId;

        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'exif-popup-overlay';
        overlay.onclick = (e) => { if (e.target === overlay) hide(); };

        overlay.innerHTML = `
            <div class="exif-popup">
                <div class="exif-popup-header">
                    <h3>ℹ Информация — #${imageId}</h3>
                    <button class="exif-popup-close" title="Закрыть">&times;</button>
                </div>
                <div class="exif-popup-body">
                    <div class="exif-loading">Загрузка</div>
                </div>
            </div>
        `;

        overlay.querySelector('.exif-popup-close').onclick = hide;
        document.body.appendChild(overlay);
        _popup = overlay;

        // Listen for Escape
        document.addEventListener('keydown', _onKey);

        // Fetch data
        try {
            let data = _cache[imageId];
            if (!data) {
                const resp = await fetch(`/photo/${imageId}`);
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                data = await resp.json();
                _cache[imageId] = data;
            }

            // Check popup still open for this ID
            if (_currentId !== imageId || !_popup) return;

            const body = _popup.querySelector('.exif-popup-body');
            body.innerHTML = _buildContent(data);
        } catch (err) {
            if (_popup) {
                const body = _popup.querySelector('.exif-popup-body');
                body.innerHTML = `<div style="color:#e94560;text-align:center;padding:20px">Ошибка: ${err.message}</div>`;
            }
        }
    }

    function hide() {
        document.removeEventListener('keydown', _onKey);
        if (_popup) {
            _popup.remove();
            _popup = null;
        }
        _currentId = null;
    }

    function _onKey(e) {
        if (e.key === 'Escape') { hide(); e.stopPropagation(); }
    }

    // ── Render badge on thumbnail card ───────────────────────────────────

    function renderBadge(cardEl, imageId) {
        if (!cardEl || cardEl.querySelector('.info-badge')) return;
        const badge = document.createElement('span');
        badge.className = 'info-badge';
        badge.textContent = 'i';
        badge.title = 'Информация о фото';
        badge.onclick = (e) => {
            e.stopPropagation();
            e.preventDefault();
            show(imageId);
        };
        cardEl.appendChild(badge);
    }

    // ── Public API ───────────────────────────────────────────────────────

    return {
        show,
        hide,
        renderBadge,
        _copy: _copyText  // exposed for inline onclick
    };
})();
