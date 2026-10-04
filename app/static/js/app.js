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

  sweepGrid:    $('sweep-grid'),
};

const state = {
  status:       null,
  pendingFile:  null,
  activeSample: null,
  lastResult:   null,
  view:         'overlay',
  busy:         false,
  sweepTimer:   null,
  elapsedTimer: null,
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

/** Remove + reflow + re-add a class so a CSS animation replays on re-run. */
function retrigger(node, cls) {
  if (!node) return;
  node.classList.remove(cls);
  // eslint-disable-next-line no-unused-expressions
  void node.offsetWidth;
  node.classList.add(cls);
}

/** Animate a number counting from its previous shown value to `target`. */
function animateNumber(node, target, { decimals = 4, prefix = '', suffix = '', duration = 700 } = {}) {
  const from = Number(node.dataset.value || 0);
  const start = performance.now();

  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);              // ease-out-cubic
    const value = from + (target - from) * eased;
    node.textContent = `${prefix}${value.toFixed(decimals)}${suffix}`;
    if (t < 1) requestAnimationFrame(tick);
    else node.dataset.value = String(target);
  }
  requestAnimationFrame(tick);
}

function setBusy(busy, message) {
  state.busy = busy;
  el.progress.hidden = !busy;
  if (message) el.progressText.textContent = message;
  el.runBtn.disabled = busy || (!state.pendingFile && state.activeSample === null);
  el.runBtn.classList.toggle('is-busy', busy);

  const label = el.runBtn.querySelector('.btn-label');
  if (busy) {
    label.innerHTML = '<span class="spinner"></span>Inspecting…';
    startSweepAnimation();
  } else {
    label.textContent = 'Run inspection';
    stopSweepAnimation();
  }
}

/* Lights random cells of the 16x16 grid and cycles status copy while the
   real request is in flight, so the wait itself explains the mechanism
   (a window sweeping over every region of the patch grid) instead of being
   dead time. */
const SWEEP_MESSAGES = [
  'Sweeping masks across the patch grid…',
  'Predicting hidden-region embeddings…',
  'Comparing predicted vs observed latents…',
  'Fusing evidence across scales…',
];

function startSweepAnimation() {
  if (!el.sweepGrid) return;

  if (!el.sweepGrid.childElementCount) {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 256; i++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      frag.appendChild(cell);
    }
    el.sweepGrid.appendChild(frag);
  }

  const cells = el.sweepGrid.children;
  let msgIndex = 0;
  let tick = 0;

  state.sweepTimer = setInterval(() => {
    for (const c of cells) c.classList.remove('lit');
    for (let i = 0; i < 10; i++) {
      cells[Math.floor(Math.random() * cells.length)].classList.add('lit');
    }
    tick++;
    if (tick % 6 === 0) {
      msgIndex = (msgIndex + 1) % SWEEP_MESSAGES.length;
      el.progressText.textContent = SWEEP_MESSAGES[msgIndex];
    }
  }, 140);
}

function stopSweepAnimation() {
  if (state.sweepTimer) {
    clearInterval(state.sweepTimer);
    state.sweepTimer = null;
  }
  if (el.sweepGrid) {
    for (const c of el.sweepGrid.children) c.classList.remove('lit');
  }
}

/* --------------------------------------------------------------- explainer */

el.explToggle.addEventListener('click', () => {
  const open = el.explToggle.getAttribute('aria-expanded') === 'true';
  el.explToggle.setAttribute('aria-expanded', String(!open));
  el.explBody.classList.toggle('is-collapsed', open);
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
  el.verdictBadge.classList.remove('is-anom', 'is-normal');
  void el.verdictBadge.offsetWidth;                 // reflow so the animation replays
  el.verdictBadge.classList.add(r.is_anomalous ? 'is-anom' : 'is-normal');
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
  el.gaugeFill.style.width = '0%';
  // Force the browser to register the 0% start before animating to the real
  // width, otherwise the transition has no starting point to animate from.
  void el.gaugeFill.offsetWidth;
  requestAnimationFrame(() => {
    el.gaugeFill.style.width = `${Math.max(2, Math.min(100, pos))}%`;
  });
  retrigger(el.gaugeFill, 'gauge-fill');
  el.gaugeThresh.style.left = '50%';

  // --- stats ---------------------------------------------------------- //
  const statDefs = [
    { k: 'Anomaly score', value: r.score, decimals: 4, note: 'top-1% of the heatmap' },
    { k: 'Deviation', value: r.z_score, decimals: 2, suffix: 'σ',
      prefix: r.z_score >= 0 ? '+' : '', note: 'vs normal images' },
    { k: 'Threshold', value: r.threshold, decimals: 4, note: 'mean + 3σ of normals' },
    { k: 'Peak location', text: `${r.peak.x}, ${r.peak.y}`, note: 'x, y in pixels' },
    { k: 'Inference', text: `${r.elapsed_ms} ms`, note: 'full mask sweep' },
  ];

  el.stats.innerHTML = '';
  for (const def of statDefs) {
    const wrap = document.createElement('div');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = def.k;

    if (def.text !== undefined) {
      dd.textContent = def.text;
    } else {
      const numSpan = document.createElement('span');
      numSpan.dataset.value = '0';
      numSpan.textContent = '0';
      dd.appendChild(numSpan);
      animateNumber(numSpan, def.value, {
        decimals: def.decimals, prefix: def.prefix || '', suffix: def.suffix || '',
      });
    }

    if (def.note) {
      const small = document.createElement('small');
      small.textContent = ` ${def.note}`;
      dd.appendChild(document.createElement('br'));
      dd.appendChild(small);
    }

    wrap.append(dt, dd);
    el.stats.appendChild(wrap);
  }

  applyView();
  renderScales(r.scales, r.cardinality);
}

function applyView() {
  const r = state.lastResult;
  if (!r) return;

  const v = state.view;
  retrigger(el.shotA, 'anim-in');

  if (v === 'split') {
    el.shotB.hidden = false;
    retrigger(el.shotB, 'anim-in');
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

function scaleCard(s, cls, title, alt) {
  const card = document.createElement('div');
  card.className = `scale-card ${cls}`;

  const img = document.createElement('img');
  img.src = s.image;
  img.alt = alt;

  const body = document.createElement('div');
  body.className = 'scale-body';

  const head = document.createElement('div');
  head.className = 'scale-head';

  const name = document.createElement('span');
  name.className = 'scale-name';
  name.textContent = title;

  const peak = document.createElement('span');
  peak.className = 'scale-peak';
  peak.textContent = `peak ${s.max.toFixed(3)}`;

  const role = document.createElement('p');
  role.className = 'scale-role';
  role.textContent = s.role;

  head.append(name, peak);
  body.append(head, role);
  card.append(img, body);
  return card;
}

function renderScales(scales, cardinality) {
  el.scales.innerHTML = '';
  if (!scales) return;

  const windows = Object.keys(scales).map(Number).sort((a, b) => a - b);

  for (const w of windows) {
    el.scales.appendChild(scaleCard(
      scales[String(w)], w <= 3 ? 'is-small' : 'is-large',
      `${w}×${w} window`, `Anomaly grid at ${w}×${w} window scale`));
  }

  if (cardinality) {
    el.scales.appendChild(scaleCard(
      cardinality, 'is-cardinality', `cardinality · ${cardinality.mode}`,
      'Cardinality mismatch: expected vs observed component mass'));
  }
}

/* -------------------------------------------------------------------- init */

loadStatus();
loadSamples();
