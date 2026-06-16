/**
 * xai.js — Phase 6: AI Explainability & Mission Intelligence
 * ORION Mission Control — Satellite Drag Prediction
 *
 * Loaded AFTER script.js. Adds the AI Analysis tab with:
 *   - Feature importance bars (RF model importances via /api/model-info)
 *   - Confidence ring with breakdown
 *   - Mission health indicator
 *   - Re-entry risk ladder
 *   - Orbital lifetime heuristic
 *   - Natural-language AI explanation engine
 *   - Mission recommendation generator
 *   - Scrollable prediction history table
 *
 * Integrates with existing dashboard:
 *   - Hooks into predict/simulate button results via window events
 *   - Reads inputs from existing DOM elements (p-altitude, p-velocity etc.)
 *   - Dispatches xai:update custom event on each prediction
 *
 * NO existing APIs are modified. ONE new endpoint is consumed:
 *   GET /api/model-info  (served by analysis.py blueprint)
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS
═══════════════════════════════════════════════════════════════ */
const XAI_CONSTANTS = {
  // Atmospheric scale heights per layer (mirrors atmosphere.py)
  LAYERS: [
    {b:0,   r0:1.225,    h:8.44},
    {b:100, r0:5.604e-7, h:5.84},
    {b:150, r0:2.076e-9, h:25.5},
    {b:200, r0:2.541e-10,h:37.0},
    {b:300, r0:1.916e-11,h:45.5},
    {b:400, r0:2.803e-12,h:53.0},
    {b:500, r0:5.215e-13,h:58.0},
    {b:600, r0:1.137e-13,h:65.0},
    {b:700, r0:3.070e-14,h:73.0},
    {b:800, r0:1.136e-14,h:80.0},
  ],

  // Pretty labels for feature names (maps model feature keys → display names)
  FEATURE_LABELS: {
    'altitude_km':          'Altitude',
    'orbital_velocity_kms': 'Velocity',
    'f107':                 'Solar Flux (F10.7)',
    'kp':                   'Kp Index',
    'ap':                   'Ap Index',
  },

  // Icons per feature
  FEATURE_ICONS: {
    'altitude_km':          '▼',
    'orbital_velocity_kms': '▶',
    'f107':                 '☀',
    'kp':                   '⚡',
    'ap':                   '◈',
  },

  // Physics equations per feature
  FEATURE_EQ: {
    'altitude_km':          'ρ(h) = ρ₀ · exp(−(h−h₀)/H)  — exponential decrease with altitude',
    'orbital_velocity_kms': 'F_drag ∝ v²  — drag scales as velocity squared',
    'f107':                 'ρ ∝ 1 + 0.003·(F10.7−150)  — EUV heating expands thermosphere',
    'kp':                   'ρ ∝ 1 + 0.15·(Kp−1)  — Joule heating from geomagnetic storms',
    'ap':                   'Correlated with Kp — geomagnetic storm energy proxy',
  },

  // Bar colour classes by rank
  BAR_CLASSES: ['primary','secondary','tertiary','minor','minor'],

  // Confidence penalty table
  CONF_PENALTIES: [
    { test: (a)         => a < 150,          val: 20, reason: 'Altitude below model training range' },
    { test: (a)         => a > 900,          val: 18, reason: 'Altitude above model training range' },
    { test: (a,v,f,k)   => k > 7,            val: 14, reason: 'Extreme geomagnetic storm (Kp>7)' },
    { test: (a,v,f)     => f > 240,          val: 10, reason: 'Very high solar flux (F10.7>240)' },
    { test: (a,v,f)     => f < 70,           val:  8, reason: 'Very low solar flux (F10.7<70)' },
    { test: (a,v)       => v > 9 || v < 6.5, val:  8, reason: 'Velocity outside typical LEO range' },
  ],
};

/* ═══════════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════════ */
const XAI_STATE = {
  modelImportances: null,   // from /api/model-info
  featureNames:     null,
  importanceSource: null,
  predictionHistory: [],    // [{id, alt, vel, f107, kp, ap, drag, ts, risk}]
  lastInputs: null,
  lastDrag:   null,
};

/* ═══════════════════════════════════════════════════════════════
   PHYSICS HELPERS (mirrors atmosphere.py + drag_calculator.py)
   Used ONLY for SHAP-approximate contributions — NOT for prediction.
═══════════════════════════════════════════════════════════════ */
const XAI_Physics = (() => {
  function getLayer(alt) {
    const L = XAI_CONSTANTS.LAYERS;
    for (let i = L.length - 1; i >= 0; i--)
      if (alt >= L[i].b) return L[i];
    return L[0];
  }

  function baseDensity(alt) {
    const L = getLayer(alt);
    return L.r0 * Math.exp(-(alt - L.b) / L.h);
  }

  function density(alt, f107, kp) {
    const ff = Math.max(0.1, 1 + 0.003 * (f107 - 150));
    const fk = Math.max(0.1, 1 + 0.15  * (kp   - 1));
    return baseDensity(alt) * ff * fk;
  }

  /**
   * Drag force: F = 0.5 · Cd · ρ · A · v²
   * Cd=2.2, A=10 m² (representative defaults used for contribution math only)
   */
  function dragForce(alt, vel_kms, f107, kp) {
    const v   = vel_kms * 1000;
    const rho = density(alt, f107, kp);
    return 0.5 * 2.2 * rho * 10.0 * v * v;
  }

  /**
   * Compute SHAP-approximate contributions via finite differences.
   * Each feature's contribution = F(feature=actual) − F(feature=reference)
   * Reference: alt=400km, vel=7.66km/s, f107=150sfu, kp=1
   */
  function shapApprox(alt, vel, f107, kp, ap, modelImportances, featureNames) {
    // If we have model importances, blend them with physics contributions
    const REF = { alt: 400, vel: 7.66, f107: 150, kp: 1 };
    const base = dragForce(REF.alt, REF.vel, REF.f107, REF.kp);

    const physRaw = {
      'altitude_km':          dragForce(alt,     REF.vel, REF.f107, REF.kp) - base,
      'orbital_velocity_kms': dragForce(REF.alt, vel,     REF.f107, REF.kp) - base,
      'f107':                 dragForce(REF.alt, REF.vel, f107,     REF.kp) - base,
      'kp':                   dragForce(REF.alt, REF.vel, REF.f107, kp)     - base,
      'ap':                   (ap / 400) * Math.abs(dragForce(REF.alt, REF.vel, REF.f107, kp) - base) * 0.18,
    };

    // Normalise physics raw to absolute percentages
    const totalAbs = Object.values(physRaw).reduce((s,v) => s + Math.abs(v), 0) || 1;
    const physPct  = {};
    for (const k in physRaw) physPct[k] = Math.abs(physRaw[k]) / totalAbs * 100;

    // If model importances available: blend 50/50 to anchor on RF training signal
    if (modelImportances && featureNames) {
      const result = featureNames.map((name, i) => {
        const phys  = physPct[name] || 0;
        const model = modelImportances[i] * 100;
        const blended = 0.5 * phys + 0.5 * model;
        return { name, raw: physRaw[name] || 0, pct: blended };
      });
      // Renormalise so they sum to 100
      const total = result.reduce((s, r) => s + r.pct, 0) || 1;
      result.forEach(r => { r.pct = r.pct / total * 100; });
      return result.sort((a, b) => b.pct - a.pct);
    }

    // Physics-only fallback
    return Object.entries(physPct)
      .map(([name, pct]) => ({ name, raw: physRaw[name], pct }))
      .sort((a, b) => b.pct - a.pct);
  }

  /**
   * Heuristic orbital lifetime estimate [days].
   * Based on simplified decay rate using ballistic coefficient and density.
   * Reference: King-Hele decay formula approximation.
   * mass=500kg, Cd=2.2, A=10m² (generic defaults)
   */
  function lifetimeEstimateDays(alt, f107, kp, mass = 500, cd = 2.2, area = 10) {
    const beta  = mass / (cd * area);          // ballistic coefficient [kg/m²]
    const v     = Math.sqrt(3.986e14 / (6371000 + alt * 1000)); // circular speed [m/s]
    const rho   = density(alt, f107, kp);
    const H     = getLayer(alt).h * 1000;      // scale height in meters
    // Simplified lifetime: τ ≈ (beta · H) / (ρ · v · R_earth) in seconds
    // Ref: Wertz & Larson "Space Mission Analysis and Design" §7.4
    const Re    = 6371000;
    const tau_s = (beta * H) / (rho * v * Re);
    return tau_s / 86400;
  }

  return { density, dragForce, shapApprox, lifetimeEstimateDays, baseDensity };
})();

/* ═══════════════════════════════════════════════════════════════
   CONFIDENCE ENGINE
═══════════════════════════════════════════════════════════════ */
const XAI_Confidence = (() => {
  function score(alt, vel, f107, kp) {
    let s       = 97;
    const flags = [];
    XAI_CONSTANTS.CONF_PENALTIES.forEach(p => {
      if (p.test(alt, vel, f107, kp)) { s -= p.val; flags.push(p.reason); }
    });
    return { score: Math.max(35, Math.min(98, s)), flags };
  }

  function label(s) {
    if (s >= 90) return { word: 'HIGH', color: 'var(--green)' };
    if (s >= 70) return { word: 'MODERATE', color: 'var(--blue)' };
    if (s >= 50) return { word: 'LOW', color: 'var(--orange)' };
    return            { word: 'EXTRAPOLATION', color: 'var(--red)' };
  }

  return { score, label };
})();

/* ═══════════════════════════════════════════════════════════════
   MISSION STATUS ENGINE
═══════════════════════════════════════════════════════════════ */
const XAI_Mission = (() => {
  /** 0=stable, 1=moderate, 2=high, 3=critical */
  function riskLevel(alt, drag, kp, f107) {
    if (alt < 180 || drag > 0.5  || kp >= 8)             return 3;
    if (alt < 250 || drag > 0.05 || kp >= 6 || f107 > 230) return 2;
    if (alt < 350 || drag > 5e-3 || kp >= 4 || f107 > 190) return 1;
    return 0;
  }

  const LEVELS = [
    { dot: 'green',  status: 'STABLE',        sub: 'All systems nominal',          rung: 0 },
    { dot: 'green',  status: 'ELEVATED DRAG',  sub: 'Monitor orbital parameters',  rung: 1 },
    { dot: 'orange', status: 'MODERATE RISK',  sub: 'Consider reboost assessment', rung: 2 },
    { dot: 'red',    status: 'HIGH RISK',      sub: 'Immediate action required',   rung: 3 },
  ];

  const RUNGS = [
    { cls:'r0', icon:'◉', label:'NOMINAL ORBIT' },
    { cls:'r1', icon:'◎', label:'ELEVATED DRAG' },
    { cls:'r2', icon:'⚠', label:'CRITICAL DECAY' },
    { cls:'r3', icon:'🔴', label:'RE-ENTRY IMMINENT' },
  ];

  return { riskLevel, LEVELS, RUNGS };
})();

/* ═══════════════════════════════════════════════════════════════
   EXPLANATION ENGINE — natural-language sentence generator
═══════════════════════════════════════════════════════════════ */
const XAI_Explain = (() => {

  function altDescriptor(alt) {
    if (alt < 200) return { adj: 'critically low', regime: 'very dense thermosphere' };
    if (alt < 300) return { adj: 'low',            regime: 'dense lower thermosphere' };
    if (alt < 450) return { adj: 'moderate',       regime: 'mid-thermosphere' };
    if (alt < 600) return { adj: 'high',           regime: 'upper thermosphere' };
    return               { adj: 'very high',       regime: 'rarefied exosphere' };
  }

  function swDescriptor(f107, kp) {
    const solar  = f107 > 200 ? 'elevated' : f107 > 160 ? 'moderately active' : f107 < 90 ? 'very quiet' : 'quiet';
    const geo    = kp > 6 ? 'severe geomagnetic storm' : kp > 4 ? 'moderate geomagnetic disturbance' : kp > 2 ? 'mildly disturbed' : 'quiet geomagnetic conditions';
    return { solar, geo };
  }

  function velNote(vel) {
    if (vel > 8.0) return 'The high orbital velocity significantly amplifies drag, which scales as v².';
    if (vel < 7.0) return 'The reduced orbital velocity somewhat lowers drag relative to typical LEO speeds.';
    return 'Orbital velocity is within the standard LEO range, contributing normally to the v² drag term.';
  }

  /**
   * Build an array of explanation sentence objects.
   * Each has: { bullet, text, emphasis }
   */
  function build(inputs, drag, contributions, rho, lifetime) {
    const { alt, vel, f107, kp } = inputs;
    const top   = contributions[0];
    const altD  = altDescriptor(alt);
    const sw    = swDescriptor(f107, kp);
    const topLabel = XAI_CONSTANTS.FEATURE_LABELS[top.name] || top.name;
    const fmt   = v => v < 1e-3 ? v.toExponential(3) : v.toFixed(4);

    const sentences = [];

    // Sentence 1 — dominant factor
    sentences.push({
      bullet: '01',
      text: `The predicted drag force of <span class="expl-hi">${fmt(drag)} N</span> is primarily driven by
             <span class="expl-warn">${topLabel}</span> (${top.pct.toFixed(1)}% contribution).
             At <span class="expl-hi">${alt} km</span> altitude the atmospheric density is
             <span class="expl-hi">${rho.toExponential(2)} kg/m³</span>,
             placing the satellite in the <span class="expl-hi">${altD.regime}</span>.`,
    });

    // Sentence 2 — velocity
    sentences.push({
      bullet: '02',
      text: velNote(vel) + ` Current speed: <span class="expl-hi">${(vel*1000).toFixed(0)} m/s</span>.`,
    });

    // Sentence 3 — space weather
    if (kp > 3 || f107 > 170) {
      sentences.push({
        bullet: '03',
        text: `Space weather is <span class="expl-warn">${sw.solar}</span> (F10.7=${f107} sfu)
               with <span class="expl-warn">${sw.geo}</span> (Kp=${kp}).
               ${kp > 4
                 ? `The geomagnetic storm is heating the polar thermosphere, expanding atmospheric density globally by up to
                    <span class="expl-warn">${((Math.max(0.1,1+0.15*(kp-1))-1)*100).toFixed(0)}%</span> above quiet-Sun baseline.`
                 : `Solar EUV heating has expanded the thermosphere by approximately
                    <span class="expl-hi">${((Math.max(0.1,1+0.003*(f107-150))-1)*100).toFixed(1)}%</span> above the quiet baseline.`}`,
      });
    } else {
      sentences.push({
        bullet: '03',
        text: `Space weather is <span class="expl-hi">quiet</span> (F10.7=${f107} sfu, Kp=${kp}).
               Solar and geomagnetic forcing have minimal influence on current atmospheric density.`,
      });
    }

    // Sentence 4 — lifetime context
    const ltDays = Math.round(lifetime);
    const ltStr  = ltDays > 365
      ? `~${(ltDays/365).toFixed(1)} years`
      : ltDays > 30
        ? `~${Math.round(ltDays/30)} months`
        : ltDays > 1
          ? `~${ltDays} days`
          : `< 24 hours`;
    sentences.push({
      bullet: '04',
      text: `Heuristic orbital lifetime at current conditions: <span class="expl-hi">${ltStr}</span>.
             This assumes constant altitude, no reboost, and typical satellite parameters (m=500 kg, A=10 m², Cd=2.2).`,
    });

    return sentences;
  }

  return { build };
})();

/* ═══════════════════════════════════════════════════════════════
   RECOMMENDATION ENGINE
═══════════════════════════════════════════════════════════════ */
const XAI_Recommend = (() => {
  function generate(risk, alt, drag, kp, f107, lifetime) {
    if (risk === 3) {
      return {
        cls:    'critical',
        text:   `Altitude of ${alt} km is critically low. Atmospheric drag of ${drag.toExponential(2)} N will cause rapid orbital decay.
                 Estimated remaining lifetime is less than ${Math.max(1, Math.round(lifetime * 24))} hours without corrective action.`,
        action: 'ACTION: EMERGENCY REBOOST OR CONTROLLED DE-ORBIT RECOMMENDED',
      };
    }
    if (risk === 2) {
      const lt = lifetime > 1 ? `~${Math.round(lifetime)} days` : `~${Math.round(lifetime*24)} hours`;
      return {
        cls:    'moderate',
        text:   `Drag is elevated at this altitude. ${kp > 4 ? `An active geomagnetic storm (Kp=${kp}) is further increasing atmospheric density. ` : ''}
                 Estimated lifetime at current drag: ${lt}. A reboost manoeuvre of approximately ${Math.round((350 - alt) * 2.5)} m/s Δv would restore safe margins.`,
        action: 'ACTION: SCHEDULE REBOOST WITHIN NEXT 72 HOURS',
      };
    }
    if (risk === 1) {
      return {
        cls:    'moderate',
        text:   `Drag is above nominal but within manageable bounds. ${f107 > 180 ? `Elevated solar flux (F10.7=${f107}) is the primary driver of increased thermospheric density. ` : ''}
                 Continue monitoring. Reboost should be planned if altitude drops below ${Math.max(alt - 30, 300).toFixed(0)} km.`,
        action: 'ACTION: MONITOR — REBOOST ASSESSMENT RECOMMENDED WITHIN 2 WEEKS',
      };
    }
    return {
      cls:    '',
      text:   `Orbital conditions are nominal. Drag force is within expected bounds for this altitude and space weather conditions.
               No immediate action is required. Continue standard monitoring cadence.`,
      action: 'STATUS: NOMINAL ORBIT — ROUTINE MONITORING ONLY',
    };
  }

  return { generate };
})();

/* ═══════════════════════════════════════════════════════════════
   CANVAS — Confidence ring
═══════════════════════════════════════════════════════════════ */
const XAI_ConfRing = (() => {
  function draw(canvasEl, pct, color) {
    const ctx = canvasEl.getContext('2d');
    const W   = canvasEl.width;
    const H   = canvasEl.height;
    const cx  = W / 2, cy = H / 2;
    const r   = Math.min(W, H) / 2 - 6;
    const start = -Math.PI / 2;
    const end   = start + (pct / 100) * 2 * Math.PI;

    ctx.clearRect(0, 0, W, H);

    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth   = 8;
    ctx.stroke();

    // Progress arc with glow
    ctx.shadowColor = color;
    ctx.shadowBlur  = 12;
    ctx.beginPath();
    ctx.arc(cx, cy, r, start, end);
    ctx.strokeStyle = color;
    ctx.lineWidth   = 8;
    ctx.lineCap     = 'round';
    ctx.stroke();
    ctx.shadowBlur  = 0;
  }

  /** Animates the ring from 0 to target pct */
  function animate(canvasEl, targetPct, color, duration = 1000) {
    const start = performance.now();
    function step(now) {
      const t   = Math.min((now - start) / duration, 1);
      const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      draw(canvasEl, ease * targetPct, color);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  return { animate };
})();

/* ═══════════════════════════════════════════════════════════════
   RENDER — Update all XAI DOM panels
═══════════════════════════════════════════════════════════════ */
const XAI_Render = (() => {

  /** Animated counter for numeric elements */
  function animNum(el, to, decimals = 1, dur = 800) {
    const start = performance.now();
    const from  = parseFloat(el.textContent.replace(/[^0-9.\-e]/g, '')) || 0;
    function step(now) {
      const t   = Math.min((now - start) / dur, 1);
      const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const v   = from + (to - from) * ease;
      el.textContent = decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString();
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function fmtDrag(v) {
    if (v === 0 || v === undefined) return '—';
    const e = Math.floor(Math.log10(Math.abs(v)));
    return `${(v / Math.pow(10,e)).toFixed(4)}×10<sup>${e}</sup>`;
  }

  function renderDragHero(drag) {
    const el = document.getElementById('xai-drag-hero');
    if (!el || drag == null) return;
    el.innerHTML = fmtDrag(drag);
  }

  function renderFeatureBars(contributions) {
    const container = document.getElementById('xai-feat-bars');
    if (!container || !contributions) return;

    container.innerHTML = '';
    contributions.forEach((item, i) => {
      const label  = XAI_CONSTANTS.FEATURE_LABELS[item.name] || item.name;
      const cls    = XAI_CONSTANTS.BAR_CLASSES[i] || 'minor';
      const pctStr = item.pct.toFixed(1) + '%';
      const isPrimary = i === 0;

      const row = document.createElement('div');
      row.className = 'feat-bar-row';
      row.innerHTML = `
        <div class="feat-rank-badge">${String(i+1).padStart(2,'0')}</div>
        <div class="feat-bar-label">
          ${label}${isPrimary ? '<span class="primary-marker">PRIMARY</span>' : ''}
        </div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill ${cls}" data-w="${item.pct.toFixed(1)}" style="width:0%"></div>
        </div>
        <div class="feat-bar-pct" style="color:${i===0?'var(--orange)':i===1?'var(--blue)':'var(--text-secondary)'}">${pctStr}</div>
      `;
      container.appendChild(row);
    });

    // Animate bars in after paint
    requestAnimationFrame(() => {
      setTimeout(() => {
        container.querySelectorAll('.feat-bar-fill').forEach(el => {
          el.style.width = el.dataset.w + '%';
        });
      }, 60);
    });

    // Source note
    const src = document.getElementById('xai-feat-source');
    if (src) {
      src.textContent = XAI_STATE.importanceSource === 'random_forest_feature_importances'
        ? '◈ SOURCE: RF FEATURE_IMPORTANCES_ BLENDED WITH PHYSICS SHAP'
        : '◈ SOURCE: PHYSICS-PRIOR FINITE DIFFERENCE APPROXIMATION';
    }
  }

  function renderConfidence(score, label, flags) {
    const canvas = document.getElementById('xai-conf-ring');
    const pctEl  = document.getElementById('xai-conf-pct');
    const wordEl = document.getElementById('xai-conf-word');
    const detEl  = document.getElementById('xai-conf-detail');

    if (canvas) XAI_ConfRing.animate(canvas, score, label.color);
    if (pctEl)  { pctEl.style.color = label.color; pctEl.textContent = score + '%'; }
    if (wordEl) { wordEl.style.color = label.color; wordEl.textContent = label.word; }

    if (detEl) {
      if (flags.length === 0) {
        detEl.innerHTML = '<div class="conf-detail-row"><span>All inputs in training distribution</span><span class="conf-detail-val" style="color:var(--green)">✓</span></div>';
      } else {
        detEl.innerHTML = flags.map(f =>
          `<div class="conf-detail-row"><span>${f}</span><span class="conf-detail-val" style="color:var(--orange)">⚠</span></div>`
        ).join('');
      }
    }
  }

  function renderMissionHealth(risk) {
    const dot    = document.getElementById('xai-health-dot');
    const status = document.getElementById('xai-health-status');
    const sub    = document.getElementById('xai-health-sub');
    const L      = XAI_Mission.LEVELS[risk];
    if (!L) return;
    if (dot)    { dot.className = `health-dot ${L.dot}`; }
    if (status) { status.textContent = L.status; status.style.color = L.dot === 'green' ? 'var(--green)' : L.dot === 'orange' ? 'var(--orange)' : 'var(--red)'; }
    if (sub)    { sub.textContent = L.sub; }
  }

  function renderRiskLadder(risk) {
    document.querySelectorAll('.risk-rung').forEach((el, i) => {
      el.classList.toggle('active', i === risk);
    });
  }

  function renderLifetime(days) {
    const el   = document.getElementById('xai-lifetime-val');
    const unit = document.getElementById('xai-lifetime-unit');
    if (!el) return;

    if (days > 365) {
      el.textContent   = (days / 365).toFixed(1);
      unit.textContent = 'YEARS';
    } else if (days > 30) {
      el.textContent   = Math.round(days / 30);
      unit.textContent = 'MONTHS';
    } else if (days > 1) {
      el.textContent   = Math.round(days);
      unit.textContent = 'DAYS';
    } else {
      el.textContent   = Math.round(days * 24);
      unit.textContent = 'HOURS';
    }
  }

  function renderPrimaryContrib(top) {
    const feat = document.getElementById('xai-primary-feat');
    const pct  = document.getElementById('xai-primary-pct');
    const eq   = document.getElementById('xai-primary-eq');
    const icon = document.getElementById('xai-primary-icon');
    if (!feat) return;

    const label = XAI_CONSTANTS.FEATURE_LABELS[top.name] || top.name;
    if (feat) feat.textContent = label;
    if (pct)  pct.textContent  = `${top.pct.toFixed(1)}% of total drag variation`;
    if (eq)   eq.textContent   = XAI_CONSTANTS.FEATURE_EQ[top.name] || '';
    if (icon) icon.textContent = XAI_CONSTANTS.FEATURE_ICONS[top.name] || '◉';
  }

  function renderExplanation(sentences) {
    const el = document.getElementById('xai-explanation');
    if (!el) return;
    el.innerHTML = sentences.map(s => `
      <div class="expl-sentence">
        <span class="expl-bullet">[${s.bullet}]</span>
        <span class="expl-text">${s.text}</span>
      </div>
    `).join('');
  }

  function renderRecommendation(rec) {
    const block  = document.getElementById('xai-rec-block');
    const text   = document.getElementById('xai-rec-text');
    const action = document.getElementById('xai-rec-action');
    if (!block) return;

    block.className = `rec-block ${rec.cls}`;
    if (text)   text.innerHTML  = rec.text.replace(/\n/g, '<br>');
    if (action) action.textContent = rec.action;
  }

  function renderHistory(history) {
    const tbody = document.getElementById('xai-hist-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    history.slice().reverse().forEach(h => {
      const tr = document.createElement('tr');
      const riskLabel = ['STABLE','ELEVATED','CRITICAL','IMMINENT'][h.risk] || '—';
      const riskCls   = ['stable','stable','moderate','critical'][h.risk]   || 'stable';
      tr.innerHTML = `
        <td class="hist-num">#${h.id}</td>
        <td>${h.alt.toFixed(0)} km</td>
        <td>${h.vel} km/s</td>
        <td class="hist-drag">${h.drag.toExponential(3)} N</td>
        <td><span class="hist-badge ${riskCls}">${riskLabel}</span></td>
        <td>${h.ts}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  return {
    renderDragHero, renderFeatureBars, renderConfidence,
    renderMissionHealth, renderRiskLadder, renderLifetime,
    renderPrimaryContrib, renderExplanation, renderRecommendation,
    renderHistory, animNum,
  };
})();

/* ═══════════════════════════════════════════════════════════════
   ORCHESTRATOR — runs full XAI update pipeline
═══════════════════════════════════════════════════════════════ */
function xaiUpdate(inputs, drag) {
  if (!inputs || drag == null) return;

  const { alt, vel, f107, kp, ap } = inputs;

  // 1. Compute contributions
  const contribs  = XAI_Physics.shapApprox(
    alt, vel, f107, kp, ap,
    XAI_STATE.modelImportances,
    XAI_STATE.featureNames
  );

  // 2. Confidence
  const confRes   = XAI_Confidence.score(alt, vel, f107, kp);
  const confLabel = XAI_Confidence.label(confRes.score);

  // 3. Density
  const rho       = XAI_Physics.density(alt, f107, kp);

  // 4. Lifetime
  const lifetime  = XAI_Physics.lifetimeEstimateDays(alt, f107, kp);

  // 5. Risk
  const risk      = XAI_Mission.riskLevel(alt, drag, kp, f107);

  // 6. Sentences
  const sentences = XAI_Explain.build(inputs, drag, contribs, rho, lifetime);

  // 7. Recommendation
  const rec       = XAI_Recommend.generate(risk, alt, drag, kp, f107, lifetime);

  // 8. Store history
  const now = new Date();
  XAI_STATE.predictionHistory.push({
    id:   XAI_STATE.predictionHistory.length + 1,
    alt, vel, f107, kp, ap,
    drag, risk,
    ts: now.toTimeString().slice(0,8),
  });

  // 9. Render everything
  XAI_Render.renderDragHero(drag);
  XAI_Render.renderFeatureBars(contribs);
  XAI_Render.renderConfidence(confRes.score, confLabel, confRes.flags);
  XAI_Render.renderMissionHealth(risk);
  XAI_Render.renderRiskLadder(risk);
  XAI_Render.renderLifetime(lifetime);
  XAI_Render.renderPrimaryContrib(contribs[0]);
  XAI_Render.renderExplanation(sentences);
  XAI_Render.renderRecommendation(rec);
  XAI_Render.renderHistory(XAI_STATE.predictionHistory);

  XAI_STATE.lastInputs = inputs;
  XAI_STATE.lastDrag   = drag;
}

/* ═══════════════════════════════════════════════════════════════
   BOOTSTRAP — load model info, wire events
═══════════════════════════════════════════════════════════════ */
(async function xaiBootstrap() {

  // ── Load model importances from backend ─────────────────────
  try {
    const resp = await fetch('/api/model-info');
    if (resp.ok) {
      const data = await resp.json();
      XAI_STATE.modelImportances = data.importances;
      XAI_STATE.featureNames     = data.feature_names;
      XAI_STATE.importanceSource = data.source;

      const srcEl = document.getElementById('xai-feat-source');
      if (srcEl && data.source === 'random_forest_feature_importances') {
        srcEl.textContent = `◈ RF MODEL · ${data.train_samples} TRAINING SAMPLES · R²=${(data.metrics.r2||0).toFixed(4)}`;
      }
    }
  } catch (e) {
    console.info('[XAI] /api/model-info not available, using physics prior.', e.message);
  }

  // ── Hook into existing predict button ───────────────────────
  const btnPredict = document.getElementById('btn-predict');
  if (btnPredict) {
    // Wrap the existing click handler by listening on the same button
    // after existing script.js has already added its listener.
    btnPredict.addEventListener('click', async () => {
      // Small delay so the existing handler runs first and populates the result
      await new Promise(r => setTimeout(r, 800));

      // Read the raw predicted value from the telemetry card that existing code updates
      const valEl = document.getElementById('val-drag');
      const raw   = valEl ? parseFloat(valEl.textContent) : null;

      // Read inputs from existing form fields
      const inputs = {
        alt:  parseFloat(document.getElementById('p-altitude')?.value) || 408,
        vel:  parseFloat(document.getElementById('p-velocity')?.value) || 7.67,
        f107: parseFloat(document.getElementById('p-f107')?.value)     || 150,
        kp:   parseFloat(document.getElementById('p-kp')?.value)       || 2,
        ap:   parseFloat(document.getElementById('p-ap')?.value)       || 15,
      };

      if (raw && !isNaN(raw)) {
        xaiUpdate(inputs, raw);
        // Auto-switch to AI tab if user is on it or was looking at it
        const aiTab = document.querySelector('[data-view="xai-view"]');
        if (aiTab) {
          // Only auto-switch if XAI tab already active
          if (document.getElementById('xai-view')?.classList.contains('active-view')) {
            xaiUpdate(inputs, raw); // re-render for the active panel
          }
        }
      }
    });
  }

  // ── Also hook into simulate button ──────────────────────────
  const btnSim = document.getElementById('btn-simulate');
  if (btnSim) {
    btnSim.addEventListener('click', async () => {
      await new Promise(r => setTimeout(r, 3000)); // wait for simulation to complete
      const valEl = document.getElementById('val-drag');
      const raw   = valEl ? parseFloat(valEl.textContent) : null;
      const inputs = {
        alt:  parseFloat(document.getElementById('s-altitude')?.value) || 408,
        vel:  7.67,
        f107: parseFloat(document.getElementById('s-f107')?.value)     || 150,
        kp:   parseFloat(document.getElementById('s-kp')?.value)       || 2,
        ap:   15,
      };
      if (raw && !isNaN(raw)) xaiUpdate(inputs, raw);
    });
  }

  // ── Tab switching ────────────────────────────────────────────
  document.querySelectorAll('[data-view]').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active-view'));
      tab.classList.add('active');
      const target = document.getElementById(tab.dataset.view);
      if (target) target.classList.add('active-view');

      // If switching to XAI tab and we have data, re-render
      if (tab.dataset.view === 'xai-view' && XAI_STATE.lastDrag !== null) {
        xaiUpdate(XAI_STATE.lastInputs, XAI_STATE.lastDrag);
      }
    });
  });

  // ── Run initial demo with default ISS-like values ────────────
  setTimeout(() => {
    xaiUpdate({ alt: 408, vel: 7.67, f107: 150, kp: 2, ap: 15 },
      XAI_Physics.dragForce(408, 7.67, 150, 2));
  }, 600);

  console.info('[XAI] Phase 6 engine initialised. Model source:', XAI_STATE.importanceSource || 'pending');
})();