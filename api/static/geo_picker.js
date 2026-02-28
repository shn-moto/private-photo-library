/**
 * GeoPicker — reusable modal for assigning GPS coordinates to selected photos.
 *
 * Accepts coordinates in multiple formats:
 *   - Decimal: 49.813887, 19.092332
 *   - DMS:     49°48'36.0"N 19°03'39.9"E
 *   - Google Maps URL with @lat,lon
 *   - Text address → geocoded via Gemini AI (/geo/geocode)
 *
 * Usage:
 *   const geoPicker = new GeoPicker({
 *       onAssigned: (imageIds, lat, lon) => { ... }  // optional callback
 *   });
 *   geoPicker.open([123, 456, 789]);
 *
 * Requires: /geo/geocode and /geo/assign API endpoints.
 */
(function () {
    'use strict';

    // Inject CSS once
    if (!document.getElementById('geo-picker-styles')) {
        const style = document.createElement('style');
        style.id = 'geo-picker-styles';
        style.textContent = `
.geo-picker-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.8);
    z-index: 2000;
    justify-content: center;
    align-items: center;
}
.geo-picker-overlay.active { display: flex; }
.geo-picker-box {
    background: #16213e;
    padding: 24px 28px;
    border-radius: 12px;
    max-width: 480px;
    width: 92%;
    text-align: left;
}
.geo-picker-box h3 {
    margin: 0 0 12px;
    color: #4ade80;
    font-size: 17px;
}
.geo-picker-hint {
    color: #aaa;
    font-size: 13px;
    margin-bottom: 12px;
    line-height: 1.5;
}
.geo-picker-hint span { color: #8899aa; }
.geo-picker-input {
    width: 100%;
    padding: 10px 12px;
    background: #0a1628;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    color: #e0e0e0;
    font-size: 15px;
    margin-bottom: 8px;
    box-sizing: border-box;
    outline: none;
}
.geo-picker-input:focus { border-color: #4ade80; }
.geo-picker-status {
    min-height: 20px;
    font-size: 13px;
    color: #8899aa;
    margin-bottom: 12px;
}
.geo-picker-buttons {
    display: flex;
    gap: 10px;
    justify-content: flex-end;
}
.geo-picker-btn {
    padding: 8px 18px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    transition: opacity 0.2s;
}
.geo-picker-btn:disabled { opacity: 0.5; cursor: default; }
.geo-picker-btn-cancel {
    background: rgba(255,255,255,0.1);
    color: #ccc;
}
.geo-picker-btn-cancel:hover { background: rgba(255,255,255,0.15); }
.geo-picker-btn-submit {
    background: #4ade80;
    color: #000;
    font-weight: 600;
}
.geo-picker-btn-submit:hover:not(:disabled) { background: #22c55e; }
`;
        document.head.appendChild(style);
    }

    class GeoPicker {
        constructor(options) {
            options = options || {};
            this.onAssigned = options.onAssigned || function () {};
            this.imageIds = [];
            this._buildDOM();
            this._bindEvents();
        }

        _buildDOM() {
            this.overlay = document.createElement('div');
            this.overlay.className = 'geo-picker-overlay';
            this.overlay.innerHTML =
                '<div class="geo-picker-box">' +
                    '<h3>\ud83d\udccd \u041d\u0430\u0437\u043d\u0430\u0447\u0438\u0442\u044c \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b</h3>' +
                    '<div class="geo-picker-hint">' +
                        '\u0412\u0432\u0435\u0434\u0438\u0442\u0435 \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b \u0438\u043b\u0438 \u0430\u0434\u0440\u0435\u0441. \u0424\u043e\u0440\u043c\u0430\u0442\u044b:<br>' +
                        '<span>49.813887, 19.092332</span><br>' +
                        '<span>49\u00b048\'36.0"N 19\u00b003\'39.9"E</span><br>' +
                        '<span>\u0438\u043b\u0438 \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0439 \u0430\u0434\u0440\u0435\u0441 (\u0433\u0435\u043e\u043a\u043e\u0434\u0438\u043d\u0433 \u0447\u0435\u0440\u0435\u0437 \u0418\u0418)</span>' +
                    '</div>' +
                    '<input type="text" class="geo-picker-input" placeholder="\u041a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b \u0438\u043b\u0438 \u0430\u0434\u0440\u0435\u0441..." autocomplete="off">' +
                    '<div class="geo-picker-status"></div>' +
                    '<div class="geo-picker-buttons">' +
                        '<button class="geo-picker-btn geo-picker-btn-cancel">\u041e\u0442\u043c\u0435\u043d\u0430</button>' +
                        '<button class="geo-picker-btn geo-picker-btn-submit">\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0438\u0442\u044c \u0438 \u043d\u0430\u0437\u043d\u0430\u0447\u0438\u0442\u044c</button>' +
                    '</div>' +
                '</div>';
            document.body.appendChild(this.overlay);

            this.input = this.overlay.querySelector('.geo-picker-input');
            this.status = this.overlay.querySelector('.geo-picker-status');
            this.submitBtn = this.overlay.querySelector('.geo-picker-btn-submit');
            this.cancelBtn = this.overlay.querySelector('.geo-picker-btn-cancel');
        }

        _bindEvents() {
            var self = this;

            this.cancelBtn.addEventListener('click', function () { self.close(); });

            this.overlay.addEventListener('click', function (e) {
                if (e.target === self.overlay) self.close();
            });

            this.input.addEventListener('keydown', function (e) {
                if (e.key === 'Enter') { e.preventDefault(); self.submitBtn.click(); }
                if (e.key === 'Escape') { e.preventDefault(); self.close(); }
            });

            // Reset confirmation state when user modifies input
            this.input.addEventListener('input', function () {
                if (self._pendingGeo) {
                    self._pendingGeo = null;
                    self.submitBtn.textContent = '\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0438\u0442\u044c \u0438 \u043d\u0430\u0437\u043d\u0430\u0447\u0438\u0442\u044c';
                }
            });

            this.submitBtn.addEventListener('click', function () { self._submit(); });

            // Global Escape (only when open)
            document.addEventListener('keydown', function (e) {
                if (e.key === 'Escape' && self.overlay.classList.contains('active')) {
                    e.stopPropagation();
                    self.close();
                }
            });
        }

        open(imageIds) {
            if (!imageIds || imageIds.length === 0) return;
            this.imageIds = imageIds.slice();
            this.input.value = '';
            this._pendingGeo = null;
            this.submitBtn.disabled = false;
            this.submitBtn.textContent = '\u041e\u043f\u0440\u0435\u0434\u0435\u043b\u0438\u0442\u044c \u0438 \u043d\u0430\u0437\u043d\u0430\u0447\u0438\u0442\u044c';
            this.status.textContent = '\u0411\u0443\u0434\u0435\u0442 \u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d\u043e \u0434\u043b\u044f ' + imageIds.length + ' \u0444\u043e\u0442\u043e';
            this.status.style.color = '#8899aa';
            this.overlay.classList.add('active');
            var inp = this.input;
            setTimeout(function () { inp.focus(); }, 100);
        }

        close() {
            this.overlay.classList.remove('active');
        }

        async _submit() {
            var query = this.input.value.trim();
            if (!query) {
                this.status.textContent = '\u0412\u0432\u0435\u0434\u0438\u0442\u0435 \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b \u0438\u043b\u0438 \u0430\u0434\u0440\u0435\u0441';
                this.status.style.color = '#e94560';
                return;
            }

            // If we have pending geocode result waiting for confirmation → assign
            if (this._pendingGeo) {
                await this._doAssign(this._pendingGeo);
                return;
            }

            this.submitBtn.disabled = true;
            this.status.innerHTML = '<span style="color:#4ade80;">\u23f3 \u041e\u043f\u0440\u0435\u0434\u0435\u043b\u044f\u044e \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b...</span>';

            try {
                // Step 1: Geocode
                var geoResp = await fetch('/geo/geocode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                if (!geoResp.ok) {
                    var err = await geoResp.json().catch(function () { return { detail: '\u041e\u0448\u0438\u0431\u043a\u0430' }; });
                    throw new Error(err.detail || '\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0438\u0442\u044c \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b');
                }
                var geo = await geoResp.json();

                // If AI/Nominatim source → show result and ask for confirmation
                if (geo.source === 'gemini' || geo.source === 'nominatim') {
                    this._pendingGeo = geo;
                    var srcLabel = geo.source === 'gemini' ? 'AI' : 'OSM';
                    this.status.innerHTML =
                        '<span style="color:#4ade80;">\ud83d\udccd ' + (geo.display || (geo.latitude + ', ' + geo.longitude)) +
                        ' <span style="color:#8899aa;">(' + srcLabel + ')</span></span><br>' +
                        '<span style="color:#ccc;font-size:12px;">' + geo.latitude + ', ' + geo.longitude + '</span>';
                    this.submitBtn.textContent = '\u2705 \u041f\u043e\u0434\u0442\u0432\u0435\u0440\u0434\u0438\u0442\u044c';
                    this.submitBtn.disabled = false;
                    return;
                }

                // Decimal/DMS/URL → assign immediately (user typed exact coords)
                await this._doAssign(geo);

            } catch (err) {
                this.status.innerHTML = '<span style="color:#e94560;">\u274c ' + err.message + '</span>';
                this.submitBtn.disabled = false;
            }
        }

        async _doAssign(geo) {
            var lat = geo.latitude, lon = geo.longitude;
            this.submitBtn.disabled = true;
            this._pendingGeo = null;
            var src = geo.source === 'gemini' ? ' (AI)' : geo.source === 'nominatim' ? ' (OSM)' : '';
            this.status.innerHTML = '<span style="color:#4ade80;">\ud83d\udccd ' +
                (geo.display || (lat + ', ' + lon)) + src + ' \u2014 \u043d\u0430\u0437\u043d\u0430\u0447\u0430\u044e...</span>';

            try {
                var assignResp = await fetch('/geo/assign', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_ids: this.imageIds, latitude: lat, longitude: lon })
                });
                if (!assignResp.ok) throw new Error('\u041e\u0448\u0438\u0431\u043a\u0430 \u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f \u043a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442');
                var assignData = await assignResp.json();

                this.status.innerHTML = '<span style="color:#4ade80;">\u2705 \u041a\u043e\u043e\u0440\u0434\u0438\u043d\u0430\u0442\u044b \u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d\u044b: ' +
                    (assignData.updated || this.imageIds.length) + ' \u0444\u043e\u0442\u043e \u2192 ' + lat + ', ' + lon + '</span>';

                var ids = this.imageIds.slice();
                var self = this;
                this.onAssigned(ids, lat, lon);
                setTimeout(function () { self.close(); }, 1500);
            } catch (err) {
                this.status.innerHTML = '<span style="color:#e94560;">\u274c ' + err.message + '</span>';
                this.submitBtn.disabled = false;
            }
        }

        destroy() {
            if (this.overlay && this.overlay.parentNode) {
                this.overlay.parentNode.removeChild(this.overlay);
            }
        }
    }

    window.GeoPicker = GeoPicker;
})();
