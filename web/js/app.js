/**
 * FormFlow OCR Enterprise — Interactive Application Engine
 * Supports Tri-Engine Mode, Interactive Glyph Ribbon, Real-time Gating, and Operator Ingestion
 */

(function () {
  'use strict';

  // Application State
  const state = {
    fields: [],
    filteredFields: [],
    currentField: null,
    currentResult: null,
    activeMode: 'tri_engine',
    confThreshold: 0.85,
    showBoundingBoxes: true,
    activeGlyphIdx: null,
    operatorStartTime: Date.now(),
    auditLogs: [],
    customImageBase64: null,
  };

  // DOM Elements Cache
  const el = {
    galleryContainer: document.getElementById('gallery-container'),
    filterType: document.getElementById('filter-type'),
    filterBox: document.getElementById('filter-box'),
    searchInput: document.getElementById('search-input'),
    tabBenchmark: document.getElementById('tab-benchmark'),
    tabCustom: document.getElementById('tab-custom'),
    paneBenchmark: document.getElementById('pane-benchmark'),
    paneCustom: document.getElementById('pane-custom'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('file-input'),
    customFieldType: document.getElementById('custom-field-type'),
    customIsComb: document.getElementById('custom-is-comb'),
    fieldCanvas: document.getElementById('field-canvas'),
    canvasPlaceholder: document.getElementById('canvas-placeholder'),
    fieldMetaBadge: document.getElementById('field-meta-badge'),
    btnToggleBoxes: document.getElementById('btn-toggle-boxes'),
    enginePills: document.getElementById('engine-pills'),
    decisionBanner: document.getElementById('decision-banner'),
    bannerTitle: document.getElementById('banner-title'),
    bannerSubtitle: document.getElementById('banner-subtitle'),
    bannerLatency: document.getElementById('banner-latency'),
    confSlider: document.getElementById('conf-slider'),
    gatingValDisplay: document.getElementById('gating-val-display'),
    teleMinConf: document.getElementById('tele-min-conf'),
    teleMeanConf: document.getElementById('tele-mean-conf'),
    teleSyntax: document.getElementById('tele-syntax'),
    glyphRibbon: document.getElementById('glyph-ribbon'),
    transcriptionInput: document.getElementById('transcription-input'),
    btnRevert: document.getElementById('btn-revert'),
    groundTruthBadge: document.getElementById('ground-truth-badge'),
    repairsBox: document.getElementById('repairs-box'),
    repairsList: document.getElementById('repairs-list'),
    btnAccept: document.getElementById('btn-accept'),
    btnCorrect: document.getElementById('btn-correct'),
    btnReject: document.getElementById('btn-reject'),
    auditCount: document.getElementById('audit-count'),
    auditTableBody: document.getElementById('audit-table-body'),
    btnExportCsv: document.getElementById('btn-export-csv'),
    btnClearAudit: document.getElementById('btn-clear-audit'),
    modalShortcuts: document.getElementById('modal-shortcuts'),
    btnShortcuts: document.getElementById('btn-shortcuts'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    toastContainer: document.getElementById('toast-container'),
  };

  // Canvas Drawing Context
  const ctx = el.fieldCanvas.getContext('2d');
  let loadedImageObj = null;

  // Initialize Application
  async function init() {
    setupEventListeners();
    await loadBenchmarkFields();
    await loadAuditLogs();
    
    // Auto-select first benchmark item
    if (state.fields.length > 0) {
      selectBenchmarkField(state.fields[0]);
    }
  }

  // Event Listeners
  function setupEventListeners() {
    // Tabs
    el.tabBenchmark.addEventListener('click', () => switchTab('benchmark'));
    el.tabCustom.addEventListener('click', () => switchTab('custom'));

    // Filters
    el.filterType.addEventListener('change', applyFilters);
    el.filterBox.addEventListener('change', applyFilters);
    el.searchInput.addEventListener('input', applyFilters);

    // Dropzone & File Upload
    el.dropzone.addEventListener('click', () => el.fileInput.click());
    el.fileInput.addEventListener('change', handleFileUpload);
    el.dropzone.addEventListener('dragover', (e) => { e.preventDefault(); el.dropzone.classList.add('dragover'); });
    el.dropzone.addEventListener('dragleave', () => el.dropzone.classList.remove('dragover'));
    el.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      el.dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });

    // Custom metadata triggers re-prediction
    el.customFieldType.addEventListener('change', () => { if (state.customImageBase64) predictCustomImage(); });
    el.customIsComb.addEventListener('change', () => { if (state.customImageBase64) predictCustomImage(); });

    // Engine Mode Selector
    el.enginePills.querySelectorAll('.engine-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        el.enginePills.querySelectorAll('.engine-pill').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        state.activeMode = btn.dataset.mode;
        reExecuteCurrentField();
      });
    });

    // Confidence Slider
    el.confSlider.addEventListener('input', handleSliderChange);

    // Preset Buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const val = parseFloat(btn.dataset.preset);
        el.confSlider.value = val;
        handleSliderChange();
      });
    });

    // Toggle Bounding Boxes
    el.btnToggleBoxes.addEventListener('click', () => {
      state.showBoundingBoxes = !state.showBoundingBoxes;
      el.btnToggleBoxes.textContent = state.showBoundingBoxes ? 'Hide Boxes' : 'Show Boxes';
      renderCanvas();
    });

    // Revert Transcription
    el.btnRevert.addEventListener('click', () => {
      if (state.currentResult) {
        el.transcriptionInput.value = state.currentResult.text;
        showToast('Transcription reverted to original AI output');
      }
    });

    // Operator Action Buttons
    el.btnAccept.addEventListener('click', () => submitOperatorAction('ACCEPT'));
    el.btnCorrect.addEventListener('click', () => submitOperatorAction('CORRECT'));
    el.btnReject.addEventListener('click', () => submitOperatorAction('REJECT'));

    // Drawer Tabs
    document.querySelectorAll('.drawer-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.drawer-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.dtab-pane').forEach(p => p.style.display = 'none');
        tab.classList.add('active');
        document.getElementById(`dpane-${tab.dataset.dtab}`).style.display = 'block';
      });
    });

    // Audit CSV Export & Clear
    el.btnExportCsv.addEventListener('click', exportAuditCsv);
    el.btnClearAudit.addEventListener('click', clearAuditLogs);

    // Shortcuts Modal
    el.btnShortcuts.addEventListener('click', () => el.modalShortcuts.style.display = 'flex');
    el.btnCloseModal.addEventListener('click', () => el.modalShortcuts.style.display = 'none');
    el.modalShortcuts.addEventListener('click', (e) => {
      if (e.target === el.modalShortcuts) el.modalShortcuts.style.display = 'none';
    });

    // Global Keyboard Shortcuts
    window.addEventListener('keydown', handleGlobalKeydown);
  }

  // Load 150 Benchmark Fields
  async function loadBenchmarkFields() {
    try {
      const res = await fetch('/api/benchmark/fields');
      const data = await res.json();
      state.fields = data.fields || [];
      state.filteredFields = [...state.fields];
      renderBenchmarkGallery();
    } catch (err) {
      console.error('Failed to load benchmark fields:', err);
      el.galleryContainer.innerHTML = '<div class="gallery-loader text-danger">Failed to load benchmark fields.</div>';
    }
  }

  // Render Benchmark Gallery
  function renderBenchmarkGallery() {
    if (state.filteredFields.length === 0) {
      el.galleryContainer.innerHTML = '<div class="gallery-loader">No matching benchmark fields.</div>';
      return;
    }

    el.galleryContainer.innerHTML = state.filteredFields.map(f => `
      <div class="gallery-card ${state.currentField && state.currentField.field_id === f.field_id ? 'active' : ''}" data-id="${f.field_id}">
        <img src="${f.image_url}" alt="Field ${f.field_id}" class="gallery-card-thumb" loading="lazy" />
        <div class="gallery-card-meta">
          <span>#${f.field_id}</span>
          <span>${f.field_type.split(' ')[0]}</span>
        </div>
        <div class="gallery-card-gt" title="${f.ground_truth}">${f.ground_truth}</div>
      </div>
    `).join('');

    // Attach click events
    el.galleryContainer.querySelectorAll('.gallery-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = parseInt(card.dataset.id, 10);
        const field = state.fields.find(item => item.field_id === id);
        if (field) selectBenchmarkField(field);
      });
    });
  }

  // Apply Filter Controls
  function applyFilters() {
    const typeVal = el.filterType.value;
    const boxVal = el.filterBox.value;
    const searchVal = el.searchInput.value.toLowerCase().trim();

    state.filteredFields = state.fields.filter(f => {
      const matchType = (typeVal === 'all') || (f.field_type.toLowerCase() === typeVal.toLowerCase());
      const matchBox = (boxVal === 'all') || (boxVal === 'comb' && f.is_comb_box) || (boxVal === 'freeform' && !f.is_comb_box);
      const matchSearch = !searchVal || 
        f.field_id.toString().includes(searchVal) || 
        f.ground_truth.toLowerCase().includes(searchVal) ||
        f.field_type.toLowerCase().includes(searchVal);
      return matchType && matchBox && matchSearch;
    });

    renderBenchmarkGallery();
  }

  // Switch between Benchmark and Custom tabs
  function switchTab(tab) {
    if (tab === 'benchmark') {
      el.tabBenchmark.classList.add('active');
      el.tabCustom.classList.remove('active');
      el.paneBenchmark.style.display = 'block';
      el.paneCustom.style.display = 'none';
    } else {
      el.tabCustom.classList.add('active');
      el.tabBenchmark.classList.remove('active');
      el.paneCustom.style.display = 'block';
      el.paneBenchmark.style.display = 'none';
    }
  }

  // Handle Custom File Upload
  function handleFileUpload(e) {
    if (e.target.files.length > 0) {
      processUploadedFile(e.target.files[0]);
    }
  }

  function processUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Please upload an image file (PNG, JPG, BMP)', 'danger');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      state.customImageBase64 = e.target.result;
      state.currentField = {
        field_id: 'Upload',
        field_type: el.customFieldType.value,
        is_comb_box: el.customIsComb.value === 'true',
        ground_truth: null
      };
      predictCustomImage();
    };
    reader.readAsDataURL(file);
  }

  async function predictCustomImage() {
    if (!state.customImageBase64) return;
    state.operatorStartTime = Date.now();
    el.canvasPlaceholder.style.display = 'none';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: state.customImageBase64,
          field_type: el.customFieldType.value,
          is_comb_box: el.customIsComb.value === 'true',
          mode: state.activeMode,
          conf_threshold: state.confThreshold
        })
      });
      const data = await res.json();
      handlePredictionResult(data);
    } catch (err) {
      console.error('Custom prediction error:', err);
      showToast('Error processing custom image', 'danger');
    }
  }

  // Select and Run Benchmark Field
  async function selectBenchmarkField(field) {
    state.currentField = field;
    state.operatorStartTime = Date.now();
    el.canvasPlaceholder.style.display = 'none';

    // Highlight card in gallery
    el.galleryContainer.querySelectorAll('.gallery-card').forEach(c => {
      c.classList.toggle('active', parseInt(c.dataset.id, 10) === field.field_id);
    });

    el.fieldMetaBadge.textContent = `Field #${field.field_id} • ${field.field_type} (${field.is_comb_box ? 'Comb-Box' : 'Freeform'})`;

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          field_id: field.field_id,
          mode: state.activeMode,
          conf_threshold: state.confThreshold
        })
      });
      const data = await res.json();
      handlePredictionResult(data);
    } catch (err) {
      console.error('Benchmark prediction error:', err);
      showToast('Error processing benchmark field', 'danger');
    }
  }

  // Re-execute current field (e.g. mode switch)
  function reExecuteCurrentField() {
    if (state.currentField) {
      if (state.currentField.field_id === 'Upload') {
        predictCustomImage();
      } else {
        selectBenchmarkField(state.currentField);
      }
    }
  }

  // Handle Prediction Response
  function handlePredictionResult(result) {
    state.currentResult = result;

    // Load base64 image onto canvas
    loadedImageObj = new Image();
    loadedImageObj.onload = () => {
      renderCanvas();
    };
    loadedImageObj.src = result.field_image_b64;

    // Update Decision Banner
    updateDecisionBanner(result);

    // Update Telemetry
    el.teleMinConf.textContent = `${result.min_conf_pct}%`;
    el.teleMinConf.className = `tele-val ${result.min_conf >= 0.85 ? 'text-success' : (result.min_conf >= 0.75 ? 'text-warning' : 'text-danger')}`;
    el.teleMeanConf.textContent = `${result.mean_conf_pct}%`;
    el.teleSyntax.textContent = result.syntax_valid ? 'Valid' : 'Syntax Issue';
    el.teleSyntax.className = `tele-val ${result.syntax_valid ? 'text-success' : 'text-danger'}`;

    // Update Ground Truth Badge
    if (result.ground_truth) {
      el.groundTruthBadge.style.display = 'inline-block';
      const isMatch = result.is_exact_match;
      el.groundTruthBadge.textContent = `Ground Truth: ${result.ground_truth} (${isMatch ? 'Exact Match ✓' : 'Mismatch ⚠️'})`;
      el.groundTruthBadge.className = `badge ${isMatch ? 'badge-success' : 'badge-warning'}`;
    } else {
      el.groundTruthBadge.style.display = 'none';
    }

    // Update Transcription Input
    el.transcriptionInput.value = result.text;

    // Update Repairs Box
    if (result.corrections && result.corrections.length > 0) {
      el.repairsBox.style.display = 'block';
      el.repairsList.innerHTML = result.corrections.map(c => `<li>• ${c}</li>`).join('');
    } else {
      el.repairsBox.style.display = 'none';
      el.repairsList.innerHTML = '';
    }

    // Render Interactive Glyph Ribbon
    renderGlyphRibbon(result.glyphs);
  }

  // Update Decision Banner
  function updateDecisionBanner(result) {
    const isApproved = (result.min_conf >= state.confThreshold && result.syntax_valid);
    el.bannerLatency.textContent = `${result.latency_ms} ms`;

    if (isApproved) {
      el.decisionBanner.className = 'decision-banner banner-approved';
      el.bannerTitle.textContent = 'AUTOMATED INGESTION APPROVED';
      el.bannerSubtitle.textContent = `Zero-Touch Ingestion Authorized • Confidence ${result.min_conf_pct}% meets threshold ${Math.round(state.confThreshold * 100)}%`;
      document.getElementById('banner-icon').innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
      `;
    } else {
      el.decisionBanner.className = 'decision-banner banner-flagged';
      el.bannerTitle.textContent = 'FLAGGED FOR OPERATOR VERIFICATION';
      el.bannerSubtitle.textContent = `Human-in-the-loop Routing Triggered • Confidence ${result.min_conf_pct}% below threshold ${Math.round(state.confThreshold * 100)}%`;
      document.getElementById('banner-icon').innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
      `;
    }
  }

  // Handle Dynamic Slider Change
  function handleSliderChange() {
    state.confThreshold = parseFloat(el.confSlider.value);
    el.gatingValDisplay.textContent = `θ = ${state.confThreshold.toFixed(2)}`;

    // Re-evaluate current field status instantaneously
    if (state.currentResult) {
      updateDecisionBanner(state.currentResult);
    }
  }

  // Render Canvas with Bounding Boxes
  function renderCanvas() {
    if (!loadedImageObj) return;

    const w = loadedImageObj.naturalWidth || loadedImageObj.width;
    const h = loadedImageObj.naturalHeight || loadedImageObj.height;

    el.fieldCanvas.width = w;
    el.fieldCanvas.height = h;
    ctx.drawImage(loadedImageObj, 0, 0);

    if (state.showBoundingBoxes && state.currentResult && state.currentResult.glyphs) {
      state.currentResult.glyphs.forEach((g, idx) => {
        const [x, y, bw, bh] = g.bbox;
        const isActive = (state.activeGlyphIdx === idx);

        // Box border
        ctx.lineWidth = isActive ? 3 : 1.5;
        if (isActive) {
          ctx.strokeStyle = '#2563eb';
          ctx.fillStyle = 'rgba(37, 99, 235, 0.15)';
        } else if (g.conf >= 0.90) {
          ctx.strokeStyle = '#059669';
          ctx.fillStyle = 'rgba(5, 150, 105, 0.08)';
        } else if (g.conf >= 0.75) {
          ctx.strokeStyle = '#d97706';
          ctx.fillStyle = 'rgba(217, 119, 6, 0.08)';
        } else {
          ctx.strokeStyle = '#dc2626';
          ctx.fillStyle = 'rgba(220, 38, 38, 0.1)';
        }

        ctx.fillRect(x, y, bw, bh);
        ctx.strokeRect(x, y, bw, bh);

        // Character label on top of box
        ctx.fillStyle = isActive ? '#2563eb' : (g.conf >= 0.90 ? '#059669' : '#d97706');
        ctx.font = 'bold 11px monospace';
        ctx.fillText(g.char, x + 2, y > 12 ? y - 3 : y + 12);
      });
    }
  }

  // Render Interactive Glyph Ribbon (Hero Component)
  function renderGlyphRibbon(glyphs) {
    if (!glyphs || glyphs.length === 0) {
      el.glyphRibbon.innerHTML = '<div class="ribbon-empty">No character glyphs segmented.</div>';
      return;
    }

    el.glyphRibbon.innerHTML = glyphs.map((g, idx) => `
      <div class="glyph-card ${state.activeGlyphIdx === idx ? 'active' : ''}" data-idx="${idx}" id="glyph-card-${idx}">
        <img src="${g.patch_b64}" class="glyph-patch-img" alt="Glyph ${idx}" />
        <span class="glyph-main-char">${g.char}</span>
        <span class="glyph-conf-badge ${g.badge_class}">${g.conf_pct}%</span>
        <div class="glyph-alts-list">
          ${g.alts.map((alt, altIdx) => `
            <div class="alt-chip" data-idx="${idx}" data-char="${alt.char}" title="Click to replace with ${alt.char} (${alt.pct}%)">
              <strong>${alt.char}</strong>
              <span>${alt.pct}%</span>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');

    // Attach card focus events
    el.glyphRibbon.querySelectorAll('.glyph-card').forEach(card => {
      card.addEventListener('mouseenter', () => {
        state.activeGlyphIdx = parseInt(card.dataset.idx, 10);
        renderCanvas();
      });
      card.addEventListener('mouseleave', () => {
        state.activeGlyphIdx = null;
        renderCanvas();
      });
      card.addEventListener('click', () => {
        state.activeGlyphIdx = parseInt(card.dataset.idx, 10);
        el.glyphRibbon.querySelectorAll('.glyph-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        renderCanvas();
      });
    });

    // Attach alternative replacement chip clicks
    el.glyphRibbon.querySelectorAll('.alt-chip').forEach(chip => {
      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        const glyphIdx = parseInt(chip.dataset.idx, 10);
        const newChar = chip.dataset.char;
        replaceCharacterInTranscription(glyphIdx, newChar);
      });
    });
  }

  // Replace Character in Transcription Box
  function replaceCharacterInTranscription(idx, newChar) {
    const currentVal = el.transcriptionInput.value;
    if (idx < currentVal.length) {
      const arr = currentVal.split('');
      arr[idx] = newChar;
      el.transcriptionInput.value = arr.join('');

      // Visual feedback
      const card = document.getElementById(`glyph-card-${idx}`);
      if (card) {
        card.querySelector('.glyph-main-char').textContent = newChar;
        card.classList.add('active');
      }

      showToast(`Character #${idx + 1} swapped to '${newChar}'`, 'success');
    }
  }

  // Submit Operator Action (Accept / Correct / Reject)
  async function submitOperatorAction(action) {
    if (!state.currentResult) return;

    const latencySec = (Date.now() - state.operatorStartTime) / 1000.0;
    const originalText = state.currentResult.text;
    const verifiedText = (action === 'REJECT') ? '[REJECTED]' : el.transcriptionInput.value;

    const payload = {
      field_id: state.currentField ? String(state.currentField.field_id) : 'Custom',
      field_type: state.currentResult.field_type || 'Unknown',
      original_text: originalText,
      verified_text: verifiedText,
      action: action,
      status: state.currentResult.status,
      min_conf: state.currentResult.min_conf,
      operator_latency_s: latencySec,
      notes: (action === 'CORRECT') ? `Corrected from ${originalText} to ${verifiedText}` : ''
    };

    try {
      const res = await fetch('/api/audit/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.success) {
        state.auditLogs.unshift(data.record);
        renderAuditTable();
        showToast(
          action === 'ACCEPT' ? `Field #${payload.field_id} ingested successfully ✓` :
          (action === 'CORRECT' ? `Correction submitted for #${payload.field_id} ✏️` : `Field #${payload.field_id} rejected 🚩`),
          action === 'ACCEPT' ? 'success' : (action === 'CORRECT' ? 'warning' : 'danger')
        );
        autoAdvanceNextField();
      }
    } catch (err) {
      console.error('Audit log error:', err);
    }
  }

  // Auto-advance to next benchmark item
  function autoAdvanceNextField() {
    if (!state.currentField || state.currentField.field_id === 'Upload') return;
    const currId = state.currentField.field_id;
    const nextItem = state.fields.find(f => f.field_id === currId + 1);
    if (nextItem) {
      setTimeout(() => selectBenchmarkField(nextItem), 400);
    }
  }

  // Load and Render Audit Logs
  async function loadAuditLogs() {
    try {
      const res = await fetch('/api/audit/logs');
      const data = await res.json();
      state.auditLogs = data.logs || [];
      renderAuditTable();
    } catch (err) {
      console.error('Failed to load audit logs:', err);
    }
  }

  function renderAuditTable() {
    el.auditCount.textContent = state.auditLogs.length;

    if (state.auditLogs.length === 0) {
      el.auditTableBody.innerHTML = '<tr class="empty-row"><td colspan="9">No operator audit events logged yet. Process a field to begin tracking.</td></tr>';
      return;
    }

    el.auditTableBody.innerHTML = state.auditLogs.map(log => `
      <tr>
        <td><code>${log.audit_id}</code></td>
        <td>${log.timestamp}</td>
        <td><strong>#${log.field_id}</strong></td>
        <td>${log.field_type}</td>
        <td><code>${log.original_text}</code></td>
        <td><strong><code>${log.verified_text}</code></strong></td>
        <td>
          <span class="badge ${log.action === 'ACCEPT' ? 'badge-success' : (log.action === 'CORRECT' ? 'badge-warning' : 'badge-danger')}">
            ${log.action}
          </span>
        </td>
        <td>${Math.round(log.min_conf * 100)}%</td>
        <td>${log.operator_latency_s}s</td>
      </tr>
    `).join('');
  }

  // Export Audit Trail as CSV
  function exportAuditCsv() {
    if (state.auditLogs.length === 0) {
      showToast('No audit logs to export', 'warning');
      return;
    }
    const headers = ['Audit ID', 'Timestamp', 'Field ID', 'Type', 'Raw AI Text', 'Verified Text', 'Action', 'Min Conf', 'Review Time (s)'];
    const rows = state.auditLogs.map(l => [
      l.audit_id, l.timestamp, l.field_id, l.field_type, l.original_text, l.verified_text, l.action, l.min_conf, l.operator_latency_s
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.map(v => `"${v}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `formflow_audit_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function clearAuditLogs() {
    state.auditLogs = [];
    renderAuditTable();
    showToast('Audit trail cleared');
  }

  // Global Keyboard Shortcuts
  function handleGlobalKeydown(e) {
    // Ignore if modal open or typing in text input (unless Enter/Esc)
    if (el.modalShortcuts.style.display === 'flex' && e.key === 'Escape') {
      el.modalShortcuts.style.display = 'none';
      return;
    }

    if (e.key === '?' && e.target.tagName !== 'INPUT') {
      e.preventDefault();
      el.modalShortcuts.style.display = 'flex';
      return;
    }

    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      submitOperatorAction('CORRECT');
      return;
    }

    if (e.key === 'Enter' && e.target !== el.searchInput) {
      e.preventDefault();
      submitOperatorAction('ACCEPT');
      return;
    }

    if (e.key === 'Escape') {
      submitOperatorAction('REJECT');
      return;
    }

    // Number keys 1, 2, 3 to swap active glyph alternatives
    if (['1', '2', '3'].includes(e.key) && e.target.tagName !== 'INPUT') {
      const altIdx = parseInt(e.key, 10) - 1;
      const targetGlyph = (state.activeGlyphIdx !== null) ? state.activeGlyphIdx : 0;
      if (state.currentResult && state.currentResult.glyphs && state.currentResult.glyphs[targetGlyph]) {
        const alts = state.currentResult.glyphs[targetGlyph].alts;
        if (alts && alts[altIdx]) {
          replaceCharacterInTranscription(targetGlyph, alts[altIdx].char);
        }
      }
    }
  }

  // Toast Notification
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    el.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 200);
    }, 2500);
  }

  // Start Application
  window.addEventListener('DOMContentLoaded', init);
})();
