// timeline.js — render milestones on Overview tab.

function renderTimeline() {
  if (!STATE.milestones) return;
  const el = document.getElementById('timeline');
  const items = [...STATE.milestones.milestones];
  // newest first
  items.sort((a, b) => b.date.localeCompare(a.date));

  el.innerHTML = items.map(m => {
    const title   = currentLang === 'en' ? (m.title_en || m.title_zh) : (m.title_zh || m.title_en);
    const summary = currentLang === 'en' ? (m.summary_en || m.summary_zh) : (m.summary_zh || m.summary_en);
    return `
      <div class="tl-item">
        <div class="tl-date">${escapeHtml(m.date)}</div>
        <div class="tl-title">${escapeHtml(title)}</div>
        <div class="tl-summary">${escapeHtml(summary)}</div>
      </div>
    `;
  }).join('');
}
