// feature_matrix.js — render the Capabilities tab.

function renderCategoryTabs() {
  if (!STATE.features) return;
  const bar = document.getElementById('cat-tabs');
  const cats = STATE.features.categories;
  const allCount = STATE.features.features.length;

  const html = [
    `<button class="cat-tab ${STATE.activeCategoryId === '__ALL__' ? 'active' : ''}" data-cat="__ALL__">
       ${escapeHtml(t('cat_all'))}<span class="count">${allCount}</span>
     </button>`,
    ...cats.map(c => {
      const n = STATE.features.features.filter(f => f.category === c.id).length;
      const active = STATE.activeCategoryId === c.id ? 'active' : '';
      const name = currentLang === 'en' ? c.name_en : c.name_zh;
      return `<button class="cat-tab ${active}" data-cat="${escapeHtml(c.id)}">
                ${escapeHtml(name)}<span class="count">${n}</span>
              </button>`;
    })
  ].join('');

  bar.innerHTML = html;
  bar.querySelectorAll('.cat-tab').forEach(btn => {
    btn.onclick = () => {
      STATE.activeCategoryId = btn.dataset.cat;
      renderCategoryTabs();
      renderFeatures();
    };
  });
}

function renderFeatures() {
  if (!STATE.features) return;
  renderCategoryTabs();
  const grid = document.getElementById('feature-grid');
  const all = STATE.features.features;
  const items = STATE.activeCategoryId === '__ALL__'
    ? all
    : all.filter(f => f.category === STATE.activeCategoryId);

  // DONE first, then IN-PROGRESS, then PLANNED; within group sort by date desc (null last)
  const order = { 'DONE': 0, 'IN-PROGRESS': 1, 'PLANNED': 2 };
  items.sort((a, b) => {
    if (order[a.status] !== order[b.status]) return order[a.status] - order[b.status];
    const da = a.completed_date || '';
    const db = b.completed_date || '';
    return db.localeCompare(da);
  });

  grid.innerHTML = items.map(f => featureCard(f)).join('');
  grid.querySelectorAll('.doc-pill').forEach(pill => {
    pill.onclick = (e) => {
      e.preventDefault();
      const doc = pill.dataset.doc;
      switchTab('docs');
      if (typeof openDoc === 'function') openDoc(doc);
    };
  });
}

function featureCard(f) {
  const title   = currentLang === 'en' ? (f.title_en || f.title_zh) : (f.title_zh || f.title_en);
  const summary = currentLang === 'en' ? (f.summary_en || f.summary_zh) : (f.summary_zh || f.summary_en);
  const date    = f.completed_date ? f.completed_date : '—';

  const docs = (f.docs || []).map(d =>
    `<span class="doc-pill" data-doc="${escapeHtml(d)}" title="${escapeHtml(d)}">${escapeHtml(d.replace(/\.md$/, ''))}</span>`
  ).join('');

  return `
    <div class="feature-card" data-id="${escapeHtml(f.id)}">
      <div class="feature-head">
        <div class="feature-title">${escapeHtml(title)}</div>
        <span class="status-badge ${f.status}">${f.status}</span>
      </div>
      <div class="feature-summary">${escapeHtml(summary)}</div>
      <div class="feature-meta">
        <span>${escapeHtml(date)}</span>
        <div class="feature-docs">${docs}</div>
      </div>
    </div>
  `;
}
