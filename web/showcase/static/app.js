// app.js — tab orchestration + initial data fetch.

const STATE = {
  features: null,        // { categories: [], features: [] }
  milestones: null,      // { milestones: [] }
  docsIndex: null,       // { groups: [] }
  activeCategoryId: '__ALL__',
  activeDocName: null,
};

// ─── Tab switching ─────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => {
    c.classList.toggle('active', c.id === `tab-${name}`);
  });
}

// ─── Boot ──────────────────────────────────────────────────────────────
async function boot() {
  setLang('zh');

  try {
    const [features, milestones, docsIndex] = await Promise.all([
      fetchJSON('/api/features'),
      fetchJSON('/api/milestones'),
      fetchJSON('/api/docs/index'),
    ]);
    STATE.features  = features;
    STATE.milestones = milestones;
    STATE.docsIndex = docsIndex;
  } catch (e) {
    console.error('failed to load content', e);
    return;
  }

  renderStats();
  renderCategoryTabs();
  renderFeatures();
  renderTimeline();
  renderDocsIndex();
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}

function renderStats() {
  if (!STATE.features || !STATE.milestones || !STATE.docsIndex) return;
  document.getElementById('stat-features').textContent =
      STATE.features.features.filter(f => f.status === 'DONE').length;
  document.getElementById('stat-milestones').textContent =
      STATE.milestones.milestones.length;
  document.getElementById('stat-docs').textContent =
      STATE.docsIndex.groups.reduce((n, g) => n + g.docs.length, 0);
  document.getElementById('stat-categories').textContent =
      STATE.features.categories.length;
}

window.addEventListener('DOMContentLoaded', boot);
