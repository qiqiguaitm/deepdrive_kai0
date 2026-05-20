// docs.js — render docs sidebar + fetch & render markdown.

function renderDocsIndex() {
  if (!STATE.docsIndex) return;
  const sidebar = document.getElementById('docs-sidebar');
  const html = STATE.docsIndex.groups.map(g => {
    const groupName = currentLang === 'en' ? g.name_en : g.name_zh;
    const items = g.docs.map(d => {
      const title = currentLang === 'en' ? (d.title_en || d.title_zh) : (d.title_zh || d.title_en);
      const active = STATE.activeDocName === d.file ? 'active' : '';
      return `<li class="${active}" data-file="${escapeHtml(d.file)}"
                  title="${escapeHtml(d.file)}">${escapeHtml(title)}</li>`;
    }).join('');
    return `
      <div class="docs-group">
        <h4>${escapeHtml(groupName)}</h4>
        <ul>${items}</ul>
      </div>
    `;
  }).join('');
  sidebar.innerHTML = html;

  sidebar.querySelectorAll('li').forEach(li => {
    li.onclick = () => openDoc(li.dataset.file);
  });
}

async function openDoc(file) {
  STATE.activeDocName = file;
  renderDocsIndex();
  const viewer = document.getElementById('doc-content');
  viewer.innerHTML = `<div class="placeholder">${escapeHtml(t('docs_loading'))} <code>${escapeHtml(file)}</code></div>`;

  try {
    const r = await fetch(`/api/doc/${encodeURIComponent(file)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const text = await r.text();
    // marked v4+ uses marked.parse; older uses marked()
    const html = (typeof marked === 'function' ? marked(text) : marked.parse(text));
    viewer.innerHTML = html;
  } catch (e) {
    viewer.innerHTML = `<div class="placeholder">${escapeHtml(t('docs_error'))}: ${escapeHtml(String(e))}</div>`;
  }
}
