/**
 * script.js — ORION Mission Control Dashboard
 * AI-Powered Satellite Drag Prediction and Orbital Decay Simulator
 *
 * Modules:
 *   Starfield      — particle background animation
 *   OrbitCanvas    — Earth + satellite orbit visualization
 *   API            — fetch() wrappers for Flask backend
 *   PredictUI      — /predict endpoint handling
 *   SimulateUI     — /simulate endpoint handling
 *   Charts         — Plotly chart rendering
 *   Telemetry      — right-sidebar live card updates
 *   UI             — toasts, loading overlay, nav, clock
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════
   STARFIELD — Particle background
═══════════════════════════════════════════════════════════════ */
const Starfield = (() => {
  const canvas = document.getElementById('starfield');
  const ctx    = canvas.getContext('2d');
  let stars    = [];

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function initStars(count = 280) {
    stars = [];
    for (let i = 0; i < count; i++) {
      stars.push({
        x:        Math.random() * canvas.width,
        y:        Math.random() * canvas.height,
        r:        Math.random() * 1.4 + 0.2,
        alpha:    Math.random(),
        speed:    Math.random() * 0.004 + 0.001,
        twinkle:  Math.random() * Math.PI * 2,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const now = performance.now() / 1000;
    stars.forEach(s => {
      s.twinkle += s.speed;
      const a = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(s.twinkle));
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(200,220,255,${a})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', () => { resize(); initStars(); });
  resize();
  initStars();
  draw();
})();


/* ═══════════════════════════════════════════════════════════════
   ORBIT CANVAS — Earth + satellite animation
═══════════════════════════════════════════════════════════════ */
const OrbitCanvas = (() => {
  const canvas = document.getElementById('orbit-canvas');
  const ctx    = canvas.getContext('2d');

  // State
  let angle       = 0;         // current satellite angle [radians]
  let orbitRadius = 0.38;      // as fraction of canvas short-side
  let trajectory  = null;      // array from simulation result
  let trajIndex   = 0;
  let animFrame   = null;
  let baseAlt     = 408;       // km — used for orbit radius scaling
  const MIN_ALT   = 100;       // reentry km
  const MAX_ALT   = 800;       // reference max km

  function resize() {
    const parent = canvas.parentElement;
    canvas.width  = parent.clientWidth;
    canvas.height = parent.clientHeight;
  }

  /** Map altitude to orbit radius (fraction of canvas half-size) */
  function altToRadius(alt_km) {
    const t = Math.max(0, Math.min(1, (alt_km - MIN_ALT) / (MAX_ALT - MIN_ALT)));
    return 0.18 + t * 0.30;   // 0.18 → 0.48
  }

  /** Draw a glowing circle */
  function glowCircle(x, y, r, color, glowSize = 12) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, glowSize);
    g.addColorStop(0,   color);
    g.addColorStop(0.4, color.replace('1)', '0.4)'));
    g.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.arc(x, y, glowSize, 0, Math.PI * 2);
    ctx.fillStyle = g;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  function drawEarth(cx, cy, er) {
    // Atmosphere glow
    const atmGrad = ctx.createRadialGradient(cx, cy, er * 0.88, cx, cy, er * 1.3);
    atmGrad.addColorStop(0,   'rgba(56,189,248,0.12)');
    atmGrad.addColorStop(1,   'rgba(56,189,248,0)');
    ctx.beginPath();
    ctx.arc(cx, cy, er * 1.3, 0, Math.PI * 2);
    ctx.fillStyle = atmGrad;
    ctx.fill();

    // Earth body
    const earthGrad = ctx.createRadialGradient(cx - er*0.3, cy - er*0.3, er*0.05, cx, cy, er);
    earthGrad.addColorStop(0,   '#1e4d8c');
    earthGrad.addColorStop(0.45,'#1a3a6e');
    earthGrad.addColorStop(0.7, '#0f2a52');
    earthGrad.addColorStop(1,   '#060e20');
    ctx.beginPath();
    ctx.arc(cx, cy, er, 0, Math.PI * 2);
    ctx.fillStyle = earthGrad;
    ctx.fill();

    // Land masses (stylized blobs)
    ctx.save();
    ctx.clip();
    ctx.fillStyle = 'rgba(34,197,94,0.22)';
    [[cx-er*0.2, cy-er*0.1, er*0.28],
     [cx+er*0.1, cy+er*0.2, er*0.22],
     [cx-er*0.3, cy+er*0.3, er*0.18]].forEach(([lx,ly,lr]) => {
      ctx.beginPath();
      ctx.arc(lx, ly, lr, 0, Math.PI*2);
      ctx.fill();
    });
    ctx.restore();

    // Rim highlight
    ctx.beginPath();
    ctx.arc(cx, cy, er, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(56,189,248,0.25)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawOrbitPath(cx, cy, or_) {
    ctx.beginPath();
    ctx.arc(cx, cy, or_, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(56,189,248,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 8]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function frame() {
    resize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cx  = canvas.width  / 2;
    const cy  = canvas.height / 2;
    const ref = Math.min(canvas.width, canvas.height) / 2;
    const er  = ref * 0.32;    // Earth radius in px
    const or_ = ref * orbitRadius; // orbit ring radius in px

    // Draw Earth
    drawEarth(cx, cy, er);

    // Draw orbit ring
    drawOrbitPath(cx, cy, or_);

    // Advance satellite angle
    if (trajectory && trajIndex < trajectory.length) {
      const pt     = trajectory[trajIndex];
      orbitRadius  = altToRadius(pt.altitude_km);
      trajIndex   += 1;
      // Update overlay readouts
      document.getElementById('ov-altitude').querySelector('.om-val').textContent =
        pt.altitude_km.toFixed(1);
      document.getElementById('ov-velocity').querySelector('.om-val').textContent =
        Math.round(pt.velocity_ms).toLocaleString();
    }

    angle += 0.012;   // radians per frame — approx visual orbit speed

    // Satellite position
    const sx = cx + or_ * Math.cos(angle);
    const sy = cy + or_ * Math.sin(angle);

    // Draw satellite trail
    for (let i = 1; i <= 18; i++) {
      const ta = angle - i * 0.035;
      const tx = cx + or_ * Math.cos(ta);
      const ty = cy + or_ * Math.sin(ta);
      ctx.beginPath();
      ctx.arc(tx, ty, 1.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(249,115,22,${0.35 * (1 - i/18)})`;
      ctx.fill();
    }

    // Draw satellite
    glowCircle(sx, sy, 4, 'rgba(249,115,22,1)', 16);

    animFrame = requestAnimationFrame(frame);
  }

  /** Start a simulation playback */
  function playTrajectory(points) {
    trajectory = points;
    trajIndex  = 0;
  }

  /** Reset to default orbit */
  function reset(alt = 408) {
    trajectory  = null;
    trajIndex   = 0;
    orbitRadius = altToRadius(alt);
  }

  window.addEventListener('resize', resize);
  resize();
  frame();

  return { playTrajectory, reset, altToRadius };
})();


/* ═══════════════════════════════════════════════════════════════
   UI HELPERS — toasts, loading, nav, clock
═══════════════════════════════════════════════════════════════ */
const UI = (() => {
  const overlay   = document.getElementById('loading-overlay');
  const container = document.getElementById('toast-container');
  let   startTime = null;
  let   clockInt  = null;

  function showLoading()  { overlay.classList.remove('d-none'); }
  function hideLoading()  { overlay.classList.add('d-none'); }

  function toast(msg, type = 'info', duration = 3000) {
    const el = document.createElement('div');
    el.className = `toast-item ${type}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => el.remove(), duration + 300);
  }

  function animateCounter(el, from, to, decimals = 0, duration = 800) {
    const start = performance.now();
    el.classList.add('counting');
    const step = now => {
      const t = Math.min((now - start) / duration, 1);
      const v = from + (to - from) * (t < 0.5 ? 2*t*t : -1+(4-2*t)*t);
      el.textContent = decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString();
      if (t < 1) requestAnimationFrame(step);
      else el.classList.remove('counting');
    };
    requestAnimationFrame(step);
  }

  // Navigation tab switching
  document.querySelectorAll('.nav-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-pill').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.input-panel').forEach(p => p.classList.remove('active-panel'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.panel).classList.add('active-panel');
    });
  });

  // Mission clock
  function startClock() {
    startTime = Date.now();
    clockInt = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
      const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
      const s = String(elapsed % 60).padStart(2, '0');
      document.getElementById('mission-clock').textContent = `T+${h}:${m}:${s}`;
    }, 1000);
  }
  startClock();

  return { showLoading, hideLoading, toast, animateCounter };
})();


/* ═══════════════════════════════════════════════════════════════
   API — fetch() wrappers
═══════════════════════════════════════════════════════════════ */
const API = (() => {
  const BASE = '';   // same origin as Flask

  async function getHealth() {
    const r = await fetch(`${BASE}/health`);
    return r.json();
  }

  async function postPredict(payload) {
    const r = await fetch(`${BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Prediction failed');
    return data;
  }

  async function postSimulate(payload) {
    const r = await fetch(`${BASE}/simulate`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Simulation failed');
    return data;
  }

  return { getHealth, postPredict, postSimulate };
})();


/* ═══════════════════════════════════════════════════════════════
   TELEMETRY — right-sidebar card updates
═══════════════════════════════════════════════════════════════ */
const Telemetry = (() => {
  function setDrag(force_N) {
    const el = document.getElementById('val-drag');
    UI.animateCounter(el, 0, force_N, 4);
    // bar: scale log10 against typical LEO range (1e-6 to 1e-1)
    const pct = Math.max(0, Math.min(100,
      (Math.log10(force_N + 1e-20) + 6) / 5 * 100
    ));
    document.getElementById('bar-drag').style.width = pct + '%';
  }

  function setAltitude(km) {
    UI.animateCounter(document.getElementById('val-altitude'), 408, km, 2);
    const pct = Math.max(0, Math.min(100, (km - 100) / 700 * 100));
    document.getElementById('bar-altitude').style.width = pct + '%';
    document.getElementById('ov-altitude').querySelector('.om-val').textContent = km.toFixed(1);
  }

  function setVelocity(ms) {
    UI.animateCounter(document.getElementById('val-velocity'), 7660, ms, 0);
    const pct = Math.max(0, Math.min(100, (ms - 6000) / 3000 * 100));
    document.getElementById('bar-velocity').style.width = pct + '%';
    document.getElementById('ov-velocity').querySelector('.om-val').textContent = Math.round(ms).toLocaleString();
  }

  function setDensity(rho) {
    if (rho !== undefined) {
      document.getElementById('val-density').textContent = rho.toExponential(3);
    }
  }

  function setWeather(f107, kp) {
    document.getElementById('wx-f107').textContent = f107;
    document.getElementById('wx-kp').textContent   = parseFloat(kp).toFixed(1);
  }

  function setReentry(isReentry) {
    const card = document.getElementById('reentry-card');
    const text = document.getElementById('reentry-text');
    if (isReentry) {
      card.classList.add('danger');
      text.textContent = 'RE-ENTRY DETECTED';
    } else {
      card.classList.remove('danger');
      text.textContent = 'NOMINAL ORBIT';
    }
  }

  function setSummary({ steps, time_s, alt, vel }) {
    document.getElementById('sum-steps').textContent = steps !== undefined ? steps.toLocaleString() : '—';
    document.getElementById('sum-time').textContent  = time_s !== undefined ? `${time_s.toFixed(0)}s` : '—';
    document.getElementById('sum-alt').textContent   = alt !== undefined ? `${alt.toFixed(2)} km` : '—';
    document.getElementById('sum-vel').textContent   = vel !== undefined ? `${Math.round(vel)} m/s` : '—';
  }

  function addHistory(alt, force, label = 'PREDICT') {
    const list = document.getElementById('history-list');
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
      <div>
        <div class="hi-label">${label} · ${alt} km</div>
      </div>
      <div class="hi-val">${force.toExponential(3)} N</div>
    `;
    list.prepend(item);
    // Cap at 12 history items
    while (list.children.length > 12) list.removeChild(list.lastChild);
  }

  return { setDrag, setAltitude, setVelocity, setDensity, setWeather, setReentry, setSummary, addHistory };
})();


/* ═══════════════════════════════════════════════════════════════
   CHARTS — Plotly rendering
═══════════════════════════════════════════════════════════════ */
const Charts = (() => {
  const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font:          { family: 'Share Tech Mono', color: '#64748b', size: 10 },
    margin:        { t: 36, r: 12, b: 36, l: 48 },
    xaxis: {
      gridcolor: 'rgba(56,189,248,0.07)',
      linecolor: 'rgba(56,189,248,0.15)',
      tickfont:  { size: 9 },
    },
    yaxis: {
      gridcolor: 'rgba(56,189,248,0.07)',
      linecolor: 'rgba(56,189,248,0.15)',
      tickfont:  { size: 9 },
    },
    showlegend: false,
  };

  const CONFIG = { displayModeBar: false, responsive: true };

  function makeTrace(x, y, color, name, fill = false) {
    return {
      x, y, name,
      type: 'scatter',
      mode: 'lines',
      line:  { color, width: 1.5, shape: 'spline' },
      fill:  fill ? 'tozeroy' : 'none',
      fillcolor: fill ? color.replace('1)', '0.06)') : undefined,
    };
  }

  function render(trajectory) {
    const times    = trajectory.map(p => p.time_s);
    const alts     = trajectory.map(p => p.altitude_km);
    const drags    = trajectory.map(p => p.drag_force_N);
    const vels     = trajectory.map(p => p.velocity_ms);

    const grid = document.getElementById('chart-grid');
    grid.classList.remove('d-none');

    // Staggered card reveal
    const cards = grid.querySelectorAll('.chart-card');
    cards.forEach((c, i) => {
      setTimeout(() => c.classList.add('visible'), i * 120);
    });

    const plots = [
      {
        el: 'chart-altitude', title: 'ALTITUDE vs TIME',
        trace: makeTrace(times, alts, 'rgba(56,189,248,1)', 'Altitude (km)', true),
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'km', font: { size: 9 } } },
      },
      {
        el: 'chart-drag', title: 'DRAG FORCE vs TIME',
        trace: makeTrace(times, drags, 'rgba(249,115,22,1)', 'Drag (N)', true),
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'N', font: { size: 9 } }, type: 'log' },
      },
      {
        el: 'chart-velocity', title: 'VELOCITY vs TIME',
        trace: makeTrace(times, vels, 'rgba(34,211,238,1)', 'Velocity (m/s)', false),
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'm/s', font: { size: 9 } } },
      },
      {
        el: 'chart-density', title: 'DRAG ACCELERATION',
        trace: makeTrace(times, drags.map((d,i) => d / (vels[i] || 1) * 1e6),
          'rgba(167,139,250,1)', 'accel (µm/s²)', true),
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'µm/s²', font: { size: 9 } } },
      },
    ];

    plots.forEach(({ el, title, trace, yaxis }) => {
      Plotly.newPlot(el, [trace], {
        ...PLOTLY_LAYOUT,
        title: { text: title, font: { size: 10, color: '#64748b', family: 'Share Tech Mono' }, x: 0.02 },
        yaxis,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'TIME (s)', font: { size: 9 } } },
      }, CONFIG);
    });
  }

  return { render };
})();


/* ═══════════════════════════════════════════════════════════════
   VALIDATION — client-side input checks
═══════════════════════════════════════════════════════════════ */
function validate(fields) {
  // fields: [{id, min, max, label}]
  for (const f of fields) {
    const el  = document.getElementById(f.id);
    const val = parseFloat(el.value);
    if (isNaN(val)) {
      UI.toast(`${f.label}: enter a valid number`, 'error');
      el.focus();
      return false;
    }
    if (f.min !== undefined && val < f.min) {
      UI.toast(`${f.label}: must be ≥ ${f.min}`, 'error');
      el.focus();
      return false;
    }
    if (f.max !== undefined && val > f.max) {
      UI.toast(`${f.label}: must be ≤ ${f.max}`, 'error');
      el.focus();
      return false;
    }
  }
  return true;
}

function updateReentryStatus(altitude, dragForce) {
  const card = document.getElementById("reentry-card");
  const text = document.getElementById("reentry-text");
  const icon = card.querySelector(".reentry-icon");

  card.classList.remove("safe", "warning", "critical", "danger");

  if (altitude <= 180 || dragForce >= 0.3) {
      text.textContent = "RE-ENTRY IMMINENT";
      icon.textContent = "🔴";
      card.classList.add("danger");
  }
  else if (altitude <= 250 || dragForce >= 0.1) {
      text.textContent = "CRITICAL DECAY";
      icon.textContent = "⚠";
      card.classList.add("critical");
  }
  else if (altitude <= 400 || dragForce >= 0.01) {
      text.textContent = "ELEVATED DRAG";
      icon.textContent = "🟠";
      card.classList.add("warning");
  }
  else {
      text.textContent = "NOMINAL ORBIT";
      icon.textContent = "🟢";
      card.classList.add("safe");
  }
}

/* ═══════════════════════════════════════════════════════════════
   PREDICT UI
═══════════════════════════════════════════════════════════════ */
document.getElementById('btn-predict').addEventListener('click', async () => {
  const fields = [
    { id: 'p-altitude', min: 100, max: 2000,  label: 'Altitude' },
    { id: 'p-velocity', min: 0.1, max: 12,    label: 'Velocity' },
    { id: 'p-f107',     min: 60,  max: 300,   label: 'F10.7' },
    { id: 'p-kp',       min: 0,   max: 9,     label: 'Kp' },
    { id: 'p-ap',       min: 0,               label: 'Ap' },
  ];
  if (!validate(fields)) return;

  const payload = {
    altitude_km:          parseFloat(document.getElementById('p-altitude').value),
    orbital_velocity_kms: parseFloat(document.getElementById('p-velocity').value),
    f107:                 parseFloat(document.getElementById('p-f107').value),
    kp:                   parseFloat(document.getElementById('p-kp').value),
    ap:                   parseFloat(document.getElementById('p-ap').value),
  };

  UI.showLoading();
  try {
    const result = await API.postPredict(payload);
    const force  = result.predicted_drag_force_N;

    updateReentryStatus(payload.altitude_km, force);

    Telemetry.setDrag(force);
    Telemetry.setAltitude(payload.altitude_km);
    Telemetry.setWeather(payload.f107, payload.kp);
    Telemetry.addHistory(payload.altitude_km, force, 'PREDICT');

    OrbitCanvas.reset(payload.altitude_km);

    UI.toast(`Drag force: ${force.toExponential(4)} N`, 'success');
  } catch (err) {
    UI.toast(`Error: ${err.message}`, 'error');
  } finally {
    UI.hideLoading();
  }
});


/* ═══════════════════════════════════════════════════════════════
   SIMULATE UI
═══════════════════════════════════════════════════════════════ */
document.getElementById('btn-simulate').addEventListener('click', async () => {
  const fields = [
    { id: 's-altitude', min: 100, max: 2000,  label: 'Altitude' },
    { id: 's-steps',    min: 1,   max: 10000, label: 'Steps' },
    { id: 's-dt',       min: 1,   max: 300,   label: 'Timestep' },
    { id: 's-mass',     min: 1,               label: 'Mass' },
    { id: 's-cd',       min: 0.1,             label: 'Cd' },
    { id: 's-area',     min: 0.1,             label: 'Area' },
    { id: 's-f107',     min: 60,  max: 300,   label: 'F10.7' },
    { id: 's-kp',       min: 0,   max: 9,     label: 'Kp' },
  ];
  if (!validate(fields)) return;

  const payload = {
    altitude_km: parseFloat(document.getElementById('s-altitude').value),
    steps:       parseInt(document.getElementById('s-steps').value),
    dt:          parseFloat(document.getElementById('s-dt').value),
    mass:        parseFloat(document.getElementById('s-mass').value),
    cd:          parseFloat(document.getElementById('s-cd').value),
    area:        parseFloat(document.getElementById('s-area').value),
    f107:        parseFloat(document.getElementById('s-f107').value),
    kp:          parseFloat(document.getElementById('s-kp').value),
  };

  UI.showLoading();
  try {
    const result = await API.postSimulate(payload);

    // Update telemetry cards
    Telemetry.setAltitude(result.final_altitude_km);
    Telemetry.setVelocity(result.final_velocity_ms);
    Telemetry.setReentry(result.reentry);
    Telemetry.setWeather(payload.f107, payload.kp);
    Telemetry.setSummary({
      steps:  result.steps_completed,
      time_s: result.total_time_s,
      alt:    result.final_altitude_km,
      vel:    result.final_velocity_ms,
    });

    // Last drag force reading
    const pts = result.trajectory_points;
    if (pts && pts.length > 0) {

      const lastDrag = pts[pts.length - 1].drag_force_N;
  
      updateReentryStatus(
          result.final_altitude_km,
          lastDrag
      );
  
      Telemetry.setDrag(lastDrag);
      Telemetry.addHistory(result.final_altitude_km, lastDrag, 'SIM');
  
      // Animate orbit canvas through trajectory
      OrbitCanvas.playTrajectory(pts);
  
      // Render Plotly charts
      Charts.render(pts);
    }

    const msg = result.reentry
      ? `RE-ENTRY after ${result.steps_completed} steps!`
      : `Sim complete — final alt ${result.final_altitude_km.toFixed(2)} km`;
    UI.toast(msg, result.reentry ? 'error' : 'success');

  } catch (err) {
    UI.toast(`Error: ${err.message}`, 'error');
  } finally {
    UI.hideLoading();
  }
});


/* ═══════════════════════════════════════════════════════════════
   HEALTH CHECK on load
═══════════════════════════════════════════════════════════════ */
(async () => {
  const dot   = document.getElementById('api-dot');
  const label = document.getElementById('api-label');
  try {
    const data = await API.getHealth();
    if (data.status === 'ok') {
      dot.classList.add('ok');
      label.textContent = 'API ONLINE';
      UI.toast('Backend connected', 'success', 2500);
    } else {
      throw new Error('not healthy');
    }
  } catch {
    dot.classList.add('err');
    label.textContent = 'API OFFLINE';
    UI.toast('Cannot reach Flask backend', 'error', 4000);
  }
})();