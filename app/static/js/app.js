/* ==========================================================================
   Logical-JEPA — web demo client
   Plain ES2020, no build step, no dependencies.
   ========================================================================== */

'use strict';

const $ = (id) => document.getElementById(id);

const el = {
  statusChip:   $('status-chip'),
  headerMeta:   $('header-meta'),

  explToggle:   $('explainer-toggle'),
  explBody:     $('explainer-body'),

  dropzone:     $('dropzone'),
  fileInput:    $('file-input'),
  runBtn:       $('run-btn'),
  clearBtn:     $('clear-btn'),

  samples:      $('samples'),
  samplesCap:   $('samples-caption'),

  progress:     $('progress'),
  progressText: $('progress-text'),
  alert:        $('alert'),

  results:      $('results'),
  verdictBadge: $('verdict-badge'),
  verdictWord:  $('verdict-word'),
  verdictSub:   $('verdict-sub'),
  gaugeFill:    $('gauge-fill'),
  gaugeThresh:  $('gauge-threshold'),
  stats:        $('stats'),

  viewSeg:      $('view-seg'),
  shotA:        $('shot-a'),
  shotB:        $('shot-b'),
  imgA:         $('img-a'),
  imgB:         $('img-b'),
  capA:         $('cap-a'),
  capB:         $('cap-b'),

  scales:       $('scales'),
  cardgrid:     $('cardgrid'),
  calibNote:    $('calib-note'),
};

const state = {
  status:       null,
  pendingFile:  null,
  activeSample: null,
  lastResult:   null,
  view:         'overlay',
  busy:         false,
};

/* ------------------------------------------------------------------ utils */

function fmtInt(n) {
  return Number(n).toLocaleString('en-US');
}

function fmtParams(n) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : `${(n / 1e3).toFixed(0)}K`;
}

function prettyDefect(name) {
  if (name === 'good') return 'normal';
  return String(name).replace(/_anomalies$/, '').replace(/_/g, ' ');
}

function showAlert(message) {
  el.alert.textContent = message;
  el.alert.hidden = false;
}

function clearAlert() {
  el.alert.hidden = true;
}

function setBusy(busy, message) {
  state.busy = busy;
  el.progress.hidden = !busy;
  if (message) el.progressText.textContent = message;
  el.runBtn.disabled = busy || (!state.pendingFile && state.activeSample === null);
}

/* --------------------------------------------------------------- explainer */

el.explToggle.addEventListener('click', () => {
  const open = el.explToggle.getAttribute('aria-expanded') === 'true';
  el.explToggle.setAttribute('aria-expanded', String(!open));
  el.explBody.hidden = open;
});

/* ------------------------------------------------------------------ status */

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    if (!res.ok || !data.ready) throw new Error(data.error || 'Model not ready');
    state.status = data;

    renderHeader(data);
    renderModelCard(data);
  } catch (err) {
    el.statusChip.textContent = 'model unavailable';
    el.statusChip.className = 'chip chip-bad';
    showAlert(`Could not reach the model: ${err.message}`);
  }
}

function renderHeader(s) {
  const chips = [
    { text: s.category, cls: 'chip' },
    { text: s.device, cls: 'chip' },
    { text: `${s.model.trainable_params >= 1e6
              ? (s.model.trainable_params / 1e6).toFixed(1) + 'M'
              : s.model.trainable_params} params`, cls: 'chip' },
    {
      text: s.calibrated ? 'calibrated' : 'uncalibrated',
      cls: s.calibrated ? 'chip chip-ok' : 'chip chip-bad',
    },
  ];

  el.headerMeta.innerHTML = '';
  for (const c of chips) {
    const span = document.createElement('span');
    span.className = c.cls;
    span.textContent = c.text;
    el.headerMeta.appendChild(span);
  }
}

function renderModelCard(s) {
  const rows = [
    ['Category',        s.category],
    ['Patch grid',      `${s.model.grid} @ ${s.model.patch_size}px`],
    ['Input size',      `${s.model.img_size}×${s.model.img_size}`],
    ['Embedding dim',   s.model.embed_dim],
    ['Context encoder', `${fmtParams(s.model.encoder_params)} params`],
    ['Predictor',       `${fmtParams(s.model.predictor_params)} params`],
    ['Training masks',  `${s.masking.strategy}${s.masking.mode ? ` / ${s.masking.mode}` : ''}`],
    ['Sweep windows',   s.inference.sweep_windows.map((w) => `${w}×${w}`).join(', ')],
    ['Sweep configs',   `${fmtInt(s.inference.sweep_configs)} per image`],
    ['Distance',        s.inference.distance],
    ['Scale fusion',    s.inference.fusion],
    ['Image score',     s.inference.aggregation],
  ];

  el.cardgrid.innerHTML = '';
  for (const [k, v] of rows) {
    const div = document.createElement('div');
    const kEl = document.createElement('div');
    const vEl = document.createElement('div');
    kEl.className = 'k'; kEl.textContent = k;
    vEl.className = 'v'; vEl.textContent = v;
    div.append(kEl, vEl);
    el.cardgrid.appendChild(div);
  }

  const c = s.calibration;
  el.calibNote.textContent = s.calibrated
    ? `Threshold ${c.threshold.toFixed(4)} = mean + 3σ of the score on ${c.n_samples} `
      + `held-out NORMAL images (mean ${c.score_mean.toFixed(4)}, σ ${c.score_std.toFixed(4)}). `
      + `No anomalous image informs this threshold.`
    : 'No calibration loaded — run evaluate.py to fit a decision threshold on normal images. '
      + 'Scores are still valid for ranking, but the NORMAL/ANOMALOUS verdict is not meaningful.';
}

/* ----------------------------------------------------------------- samples */

async function loadSamples() {
  try {
    const res = await fetch('/api/samples?limit=6');
    const data = await res.json();

    const groups = data.groups || {};
    const names = Object.keys(groups);

    if (!names.length) {
      el.samplesCap.textContent = 'No test split found for this category.';
      return;
    }

    el.samplesCap.textContent =
      'Click a sample to inspect it. Its true label is revealed after inference.';
    el.samples.innerHTML = '';

    // good first, then logical, then structural
    const order = ['good', 'logical_anomalies', 'structural_anomalies'];
    names.sort((a, b) => order.indexOf(a) - order.indexOf(b));

    for (const name of names) {
      const group = document.createElement('div');
      group.className = 'sample-group';

      const h3 = document.createElement('h3');
      h3.textContent = `${prettyDefect(name)} · ${groups[name].length} shown`;
      group.appendChild(h3);

      const row = document.createElement('div');
      row.className = 'sample-row';

      for (const s of groups[name]) {
        row.appendChild(makeSampleTile(s, name));
      }
      group.appendChild(row);
      el.samples.appendChild(group);
    }
  } catch (err) {
    el.samplesCap.textContent = `Could not load samples: ${err.message}`;
  }
}

function makeSampleTile(sample, family) {
  const btn = document.createElement('button');
  btn.className = `sample ${family.replace('_anomalies', '')}`;
  btn.type = 'button';
  btn.title = `${sample.name} — ${prettyDefect(family)}`;

  const img = document.createElement('img');
  img.src = `/api/sample-image?index=${sample.index}`;
  img.alt = `${prettyDefect(family)} sample ${sample.name}`;
  img.loading = 'lazy';

  const tag = document.createElement('span');
  tag.className = 'tag';
  tag.textContent = sample.name;

  btn.append(img, tag);
  btn.addEventListener('click', () => selectSample(sample.index, btn));
  return btn;
}

function selectSample(index, node) {
  document.querySelectorAll('.sample.is-active')
    .forEach((n) => n.classList.remove('is-active'));
  node.classList.add('is-active');

  state.activeSample = index;
  state.pendingFile = null;
  resetDropzone();

  el.runBtn.disabled = false;
  el.clearBtn.disabled = false;
  clearAlert();

  predict();   // one click is enough
}

/* ---------------------------------------------------------------- dropzone */

function resetDropzone() {
  el.dropzone.classList.remove('has-file');
  const preview = el.dropzone.querySelector('.dz-preview');
  if (preview) preview.remove();
  el.dropzone.querySelector('.dz-icon').hidden = false;
}

function acceptFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showAlert('That file is not an image.');
    return;
  }

  clearAlert();
  state.pendingFile = file;
  state.activeSample = null;
  document.querySelectorAll('.sample.is-active')
    .forEach((n) => n.classList.remove('is-active'));

  resetDropzone();
  el.dropzone.classList.add('has-file');
  el.dropzone.querySelector('.dz-icon').hidden = true;

  const img = document.createElement('img');
  img.className = 'dz-preview';
  img.alt = 'Selected image preview';
  img.src = URL.createObjectURL(file);
  img.onload = () => URL.revokeObjectURL(img.src);
  el.dropzone.prepend(img);

  el.dropzone.querySelector('.dz-main').textContent = file.name;
  el.dropzone.querySelector('.dz-sub').textContent =
    `${(file.size / 1024).toFixed(0)} KB — ready to inspect`;

  el.runBtn.disabled = false;
  el.clearBtn.disabled = false;
}

el.dropzone.addEventListener('click', () => el.fileInput.click());
el.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); }
});
el.fileInput.addEventListener('change', (e) => acceptFile(e.target.files[0]));

['dragenter', 'dragover'].forEach((type) =>
  el.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    el.dropzone.classList.add('is-drag');
  }));

['dragleave', 'drop'].forEach((type) =>
  el.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    el.dropzone.classList.remove('is-drag');
  }));

el.dropzone.addEventListener('drop', (e) => {
  acceptFile(e.dataTransfer.files[0]);
});

// Paste an image straight from the clipboard.
document.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])]
    .find((i) => i.type.startsWith('image/'));
  if (item) acceptFile(item.getAsFile());
});

el.clearBtn.addEventListener('click', () => {
  state.pendingFile = null;
  state.activeSample = null;
  state.lastResult = null;

  resetDropzone();
  el.dropzone.querySelector('.dz-main').innerHTML =
    'Drop an image, or <span class="link">browse</span>';
  el.dropzone.querySelector('.dz-sub').textContent =
    'PNG / JPG · resized to 256×256 · max 16 MB';

  el.fileInput.value = '';
  document.querySelectorAll('.sample.is-active')
    .forEach((n) => n.classList.remove('is-active'));

  el.results.hidden = true;
  el.runBtn.disabled = true;
  el.clearBtn.disabled = true;
  clearAlert();
});

el.runBtn.addEventListener('click', predict);

/* ---------------------------------------------------------------- predict */

async function predict() {
  if (state.busy) return;
  if (!state.pendingFile && state.activeSample === null) return;

  clearAlert();
  setBusy(true, 'Sweeping masks across the image…');

  try {
    let res;
    if (state.pendingFile) {
      const form = new FormData();
      form.append('file', state.pendingFile);
      res = await fetch('/api/predict', { method: 'POST', body: form });
    } else {
      res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: state.activeSample }),
      });
    }

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    state.lastResult = data;
    renderResult(data);
    el.results.hidden = false;
    el.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    showAlert(`Inference failed — ${err.message}`);
  } finally {
    setBusy(false);
  }
}

/* ----------------------------------------------------------------- render */

function renderResult(r) {
  // --- verdict -------------------------------------------------------- //
  el.verdictBadge.classList.toggle('is-anom', r.is_anomalous);
  el.verdictBadge.classList.toggle('is-normal', !r.is_anomalous);
  el.verdictWord.textContent = r.verdict;

  if (!r.calibrated) {
    el.verdictSub.textContent = 'uncalibrated — score only';
  } else if (r.ground_truth !== undefined) {
    el.verdictSub.textContent =
      `truth: ${prettyDefect(r.ground_truth)} — ${r.correct ? 'correct' : 'incorrect'}`;
  } else {
    el.verdictSub.textContent = `${(r.confidence * 100).toFixed(0)}% confidence`;
  }

  // --- gauge ---------------------------------------------------------- //
  // The threshold sits at the mid-point; the score is placed relative to it in
  // units of the normal-score standard deviation, clamped to the visible span.
  const span = 6;                                  // ±6σ maps to the full bar
  const pos = 50 + (r.z_score / span) * 50;
  el.gaugeFill.style.width = `${Math.max(2, Math.min(100, pos))}%`;
  el.gaugeThresh.style.left = '50%';

  // --- stats ---------------------------------------------------------- //
  const stats = [
    ['Anomaly score', r.score.toFixed(4), 'top-1% of the heatmap'],
    ['Deviation', `${r.z_score >= 0 ? '+' : ''}${r.z_score.toFixed(2)}σ`, 'vs normal images'],
    ['Threshold', r.threshold.toFixed(4), 'mean + 3σ of normals'],
    ['Peak location', `${r.peak.x}, ${r.peak.y}`, 'x, y in pixels'],
    ['Inference', `${r.elapsed_ms} ms`, 'full mask sweep'],
  ];

  el.stats.innerHTML = '';
  for (const [k, v, note] of stats) {
    const wrap = document.createElement('div');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = k;
    dd.textContent = v;
    if (note) {
      const small = document.createElement('small');
      small.textContent = ` ${note}`;
      dd.appendChild(document.createElement('br'));
      dd.appendChild(small);
    }
    wrap.append(dt, dd);
    el.stats.appendChild(wrap);
  }

  applyView();
  renderScales(r.scales);
}

function applyView() {
  const r = state.lastResult;
  if (!r) return;

  const v = state.view;

  if (v === 'split') {
    el.shotB.hidden = false;
    el.imgA.src = r.images.input;    el.capA.textContent = 'Input';
    el.imgB.src = r.images.overlay;  el.capB.textContent = 'Anomaly overlay';
  } else {
    el.shotB.hidden = true;
    el.imgA.src = r.images[v];
    el.capA.textContent =
      v === 'overlay' ? 'Anomaly overlay' : v === 'heatmap' ? 'Anomaly heatmap' : 'Input';
  }
}

el.viewSeg.addEventListener('click', (e) => {
  const btn = e.target.closest('.seg-btn');
  if (!btn) return;

  el.viewSeg.querySelectorAll('.seg-btn').forEach((b) => {
    b.classList.remove('is-active');
    b.setAttribute('aria-selected', 'false');
  });
  btn.classList.add('is-active');
  btn.setAttribute('aria-selected', 'true');

  state.view = btn.dataset.view;
  applyView();
});

function renderScales(scales) {
  el.scales.innerHTML = '';
  if (!scales) return;

  const windows = Object.keys(scales).map(Number).sort((a, b) => a - b);

  for (const w of windows) {
    const s = scales[String(w)];
    const card = document.createElement('div');
    card.className = `scale-card ${w <= 3 ? 'is-small' : 'is-large'}`;

    const img = document.createElement('img');
    img.src = s.image;
    img.alt = `Anomaly grid at ${w}×${w} window scale`;

    const body = document.createElement('div');
    body.className = 'scale-body';

    const head = document.createElement('div');
    head.className = 'scale-head';

    const name = document.createElement('span');
    name.className = 'scale-name';
    name.textContent = `${w}×${w} window`;

    const peak = document.createElement('span');
    peak.className = 'scale-peak';
    peak.textContent = `peak ${s.max.toFixed(3)}`;

    const role = document.createElement('p');
    role.className = 'scale-role';
    role.textContent = s.role;

    head.append(name, peak);
    body.append(head, role);
    card.append(img, body);
    el.scales.appendChild(card);
  }
}

/* -------------------------------------------------------------------- init */

loadStatus();
loadSamples();
