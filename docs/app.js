/* RoboPRO project page — vanilla JS for tab navigation and dropdown-driven viewers. */

(() => {
  const TAB_IDS = ['overview', 'tasks', 'collision', 'vision', 'language', 'leaderboard', 'rollouts'];
  let manifest = null;

  // ---------- Helpers ----------

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'html') node.innerHTML = v;
      else if (v != null) node.setAttribute(k, v);
    }
    for (const c of [].concat(children || [])) {
      if (c == null) continue;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return node;
  };

  const setVideo = (slotId, src, posterText) => {
    const slot = document.getElementById(slotId);
    if (!slot) return;
    slot.innerHTML = '';
    if (!src) {
      slot.appendChild(el('div', { class: 'video-fallback' },
        posterText || 'Rollout coming soon'));
      return;
    }
    const v = el('video', {
      controls: '', muted: '', loop: '', playsinline: '', preload: 'metadata',
      autoplay: ''
    });
    v.appendChild(el('source', { src, type: 'video/mp4' }));
    slot.appendChild(v);
  };

  const fillSelect = (select, options) => {
    select.innerHTML = '';
    for (const { value, label } of options) {
      const opt = el('option', { value }, label);
      select.appendChild(opt);
    }
  };

  // ---------- Tabs ----------

  const initTabs = () => {
    const links = document.querySelectorAll('.tab-nav a[data-tab]');
    const showTab = (id) => {
      if (!TAB_IDS.includes(id)) id = 'overview';
      links.forEach(a => a.classList.toggle('active', a.dataset.tab === id));
      TAB_IDS.forEach(t => {
        const panel = document.getElementById(t);
        if (panel) panel.classList.toggle('active', t === id);
      });
    };
    links.forEach(a => a.addEventListener('click', (e) => {
      e.preventDefault();
      const id = a.dataset.tab;
      history.replaceState(null, '', '#' + id);
      showTab(id);
      window.scrollTo({ top: document.querySelector('.tab-nav').offsetTop, behavior: 'smooth' });
    }));
    const initial = (location.hash || '').replace('#', '') || 'overview';
    showTab(initial);
  };

  // ---------- Task Gallery ----------

  const initTasks = () => {
    const sceneSel = $('#task-scene');
    const taskSel  = $('#task-slug');
    const scenes = Object.keys(manifest.scenes);

    fillSelect(sceneSel, scenes.map(k => ({ value: k, label: manifest.scenes[k].label })));

    const populateTasks = (scene) => {
      const list = manifest.tasks[scene] || [];
      fillSelect(taskSel, list.map(t => ({ value: t.slug, label: t.label + (t.kind === 'compositional' ? '  ·  comp.' : '') })));
    };

    const renderTask = () => {
      const scene = sceneSel.value;
      const slug  = taskSel.value;
      const list  = manifest.tasks[scene] || [];
      const t = list.find(x => x.slug === slug) || list[0];
      if (!t) return;
      $('#task-title').textContent = t.label;
      $('#task-desc').textContent = manifest.scenes[scene].blurb;
      $('#task-scene-label').textContent = manifest.scenes[scene].label;
      $('#task-kind').innerHTML = '';
      $('#task-kind').appendChild(el('span', {
        class: 'chip ' + (t.kind === 'compositional' ? 'compositional' : 'atomic')
      }, t.kind || 'atomic'));
      $('#task-slug-text').textContent = t.slug;
      setVideo('task-video-slot', t.video,
        'No rollout staged for ' + t.slug + ' yet');
    };

    sceneSel.addEventListener('change', () => { populateTasks(sceneSel.value); renderTask(); });
    taskSel.addEventListener('change', renderTask);

    sceneSel.value = scenes[0];
    populateTasks(scenes[0]);
    renderTask();
  };

  // ---------- Collision ----------

  const initCollision = () => {
    const sel = $('#collision-density');
    const levels = manifest.collision.levels;
    fillSelect(sel, levels.map(l => ({ value: l.key, label: l.label })));

    const render = () => {
      const lvl = levels.find(l => l.key === sel.value) || levels[0];
      $('#collision-title').textContent = lvl.label;
      $('#collision-count').textContent = String(lvl.obstacles);
      $('#collision-task').textContent = manifest.collision.task;
      $('#collision-scene').textContent =
        manifest.scenes[manifest.collision.scene]?.label || manifest.collision.scene;
      setVideo('collision-video-slot', lvl.video);
    };

    sel.addEventListener('change', render);
    sel.value = 'd10';
    render();
  };

  // ---------- Vision ----------

  const initVision = () => {
    const sel = $('#vision-axis');
    const axes = manifest.vision.axes;
    fillSelect(sel, axes.map(a => ({ value: a.key, label: a.label })));

    const render = () => {
      const a = axes.find(x => x.key === sel.value) || axes[0];
      $('#vision-title').textContent = a.label;
      $('#vision-blurb').textContent = a.blurb;
      setVideo('vision-video-slot', a.video);
    };

    sel.addEventListener('change', render);
    sel.value = 'combined';
    render();
  };

  // ---------- Language ----------

  const initLanguage = () => {
    const sel = $('#language-axis');
    const axes = manifest.language.axes;
    fillSelect(sel, axes.map(a => ({ value: a.key, label: a.label })));

    const render = () => {
      const a = axes.find(x => x.key === sel.value) || axes[0];
      $('#language-title').textContent = a.label;
      $('#language-sample').textContent = a.sample || '';
      setVideo('language-video-slot', a.video);
    };

    sel.addEventListener('change', render);
    sel.value = axes[0].key;
    render();
  };

  // ---------- Leaderboard ----------

  const initLeaderboard = () => {
    const fmt = (s) => s == null ? '—' : s;
    const num = (n, digits = 2) => n == null ? '—' : Number(n).toFixed(digits);

    // Table 1 — Clutter
    const clutter = $('#lb-clutter');
    clutter.innerHTML = '';
    for (const m of manifest.leaderboard.models) {
      const c = m.clutter;
      const row = el('tr', {}, [
        el('td', {}, m.name),
        el('td', { class: 'num' }, fmt(c.clean.office)),
        el('td', { class: 'num' }, fmt(c.clean.study)),
        el('td', { class: 'num' }, fmt(c.clean.kitchens)),
        el('td', { class: 'num' }, fmt(c.clean.kitchenl)),
        el('td', { class: 'num' }, fmt(c.clean.avg)),
        el('td', { class: 'num' }, fmt(c.cluttered.office)),
        el('td', { class: 'num' }, fmt(c.cluttered.study)),
        el('td', { class: 'num' }, fmt(c.cluttered.kitchens)),
        el('td', { class: 'num' }, fmt(c.cluttered.kitchenl)),
        el('td', { class: 'num' }, fmt(c.cluttered.avg)),
        el('td', { class: 'num' }, fmt(c.overall))
      ]);
      clutter.appendChild(row);
    }

    // Table 2 — Perturbation
    const perturb = $('#lb-perturb');
    perturb.innerHTML = '';
    for (const m of manifest.leaderboard.models) {
      const p = m.perturbation;
      const rows = [
        ['SR ↑',  p.original.sr,  p.object.sr,  p.visual.sr,  p.language.sr,  p.average.sr,  +1],
        ['HSR ↑', p.original.hsr, p.object.hsr, p.visual.hsr, p.language.hsr, p.average.hsr, +1],
        ['CR ↓',  p.original.cr,  p.object.cr,  p.visual.cr,  p.language.cr,  p.average.cr, -1]
      ];
      rows.forEach((r, i) => {
        const tr = el('tr');
        if (i === 0) {
          const modelTd = el('td', { rowspan: '3', style: 'border-right:1px solid var(--rule);' }, m.name);
          tr.appendChild(modelTd);
        }
        const orig = r[1];
        tr.appendChild(el('td', {}, r[0]));
        for (let col = 1; col <= 5; col++) {
          const v = r[col];
          const td = el('td', { class: 'num' }, num(v));
          if (col >= 2 && col <= 5 && orig != null && v != null) {
            const delta = (v - orig);
            const isImprovement = (r[6] === +1) ? delta > 0 : delta < 0;
            const sign = delta >= 0 ? '+' : '−';
            const span = el('span', {
              class: isImprovement ? 'delta-up' : 'delta-down'
            }, ' ' + sign + Math.abs(delta).toFixed(2));
            td.appendChild(span);
          }
          tr.appendChild(td);
        }
        perturb.appendChild(tr);
      });
    }

    // Table 3 — Collision
    const coll = $('#lb-collision');
    coll.innerHTML = '';
    for (const m of manifest.leaderboard.models) {
      const c = m.collision;
      const row = el('tr', {}, [
        el('td', {}, m.name),
        el('td', { class: 'num' }, num(c.clean.fcr)),
        el('td', { class: 'num' }, num(c.clean.ocr)),
        el('td', { class: 'num' }, num(c.clean.cr)),
        el('td', { class: 'num' }, num(c.cluttered.fcr)),
        el('td', { class: 'num' }, num(c.cluttered.ocr)),
        el('td', { class: 'num' }, num(c.cluttered.cr))
      ]);
      coll.appendChild(row);
    }
  };

  // ---------- Rollouts ----------

  const initRollouts = () => {
    const grid = $('#rollouts-grid');
    grid.innerHTML = '';
    for (const r of manifest.rollouts) {
      const v = el('video', {
        controls: '', muted: '', loop: '', playsinline: '', preload: 'metadata'
      });
      v.appendChild(el('source', { src: r.video, type: 'video/mp4' }));
      const fig = el('figure', { class: 'rollout' }, [
        v,
        el('figcaption', {}, [
          el('span', { class: 'policy' }, r.policy),
          el('span', { class: 'task' }, r.task),
          el('span', { class: 'scene' }, r.scene)
        ])
      ]);
      grid.appendChild(fig);
    }
  };

  // ---------- Boot ----------

  const boot = async () => {
    initTabs();
    try {
      const res = await fetch('manifest.json?v=6', { cache: 'no-cache' });
      manifest = await res.json();
    } catch (e) {
      console.error('Failed to load manifest.json:', e);
      return;
    }
    initTasks();
    initCollision();
    initVision();
    initLanguage();
    initLeaderboard();
    initRollouts();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
