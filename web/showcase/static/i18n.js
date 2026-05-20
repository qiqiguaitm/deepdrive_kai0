// Static UI strings. Feature/milestone content has its own _zh/_en fields in JSON.
const I18N = {
  zh: {
    repo_link: "DEEPDIVE_KAI0",
    header_sub: "· π0.5 双臂操作部署",

    tab_overview:     "概览",
    tab_capabilities: "能力矩阵",
    tab_pipeline:     "端到端管线",
    tab_docs:         "文档",

    ov_title: "deepdive_kai0 — π0.5 双臂操作部署",
    ov_desc:  "基于 Physical Intelligence 开源的 π0.5 视觉-语言-动作模型，在 Agilex 双 Piper 机械臂 + 三路 RealSense 相机平台上的端到端部署项目。覆盖训练数据采集 (遥操作 / DAgger / autonomy 回放) → 多机训练集群 (gf / uc / js) → 单工控机一体推理 (sim01) → 实时机械臂控制的完整闭环。",

    stat_features:    "已交付功能",
    stat_milestones:  "里程碑",
    stat_docs:        "部署文档",
    stat_categories:  "能力分类",

    arch_title: "系统架构",
    ms_title:   "最近里程碑",

    cap_title: "能力矩阵",
    cap_desc:  "按 5 大类别梳理已交付的子系统能力。每张卡片标注完成日期 + 状态 + 关联文档。点击文档徽章可在「文档」tab 直接打开对应说明。",
    cat_all:   "全部",

    pipe_title:          "端到端管线",
    pipe_desc:           "从数据采集到真机执行的三大阶段。每个阶段列举本项目实现的关键子系统。",
    pipe_stage_collect:  "数据采集",
    pipe_stage_train:    "模型训练",
    pipe_stage_deploy:   "真机部署",
    pipe_collect_items: [
      "Master-Slave 双臂遥操作 (4 路 CAN)",
      "3 路 RealSense (1×D435 + 2×D405)",
      "React + FastAPI 采集 UI",
      "DAgger 在线纠错采集",
      "Autonomy 自主回放记录",
      "LeRobot v2.1 统一格式落盘",
      "TOS 跨机增量同步"
    ],
    pipe_train_items: [
      "gf 集群 (vePFS 共享)",
      "uc 集群 (RoCEv2 多机)",
      "js 集群 (200 Gb/s InfiniBand)",
      "π0 / π0.5 / π0_fast / π0_rtc 变体",
      "Sidecar config + override_asset_id",
      "动态数据集 watcher 续训",
      "norm_stats md5 一致性校验"
    ],
    pipe_deploy_items: [
      "sim01 单机一体推理 (双 RTX 5090)",
      "ROS2-native policy node (JAX in-process)",
      "稳态 66 ms / V1 Triton 路径 32 ms",
      "RTC chunk-boundary 平滑",
      "多模态: depth + EE pose I/O",
      "动作安全限位 (URDF clamp)",
      "Rerun 3D mesh 实时可视化",
      "OBSERVE / EXECUTE 运行时切换"
    ],

    docs_title:       "部署文档浏览器",
    docs_desc:        "24 份 docs/deployment 文档按主题分组。左侧选择，右侧渲染 markdown。",
    docs_placeholder: "← 在左侧选择一份文档查看",
    docs_loading:     "加载中…",
    docs_error:       "加载失败"
  },
  en: {
    repo_link: "DEEPDIVE_KAI0",
    header_sub: "· π0.5 dual-arm deployment",

    tab_overview:     "Overview",
    tab_capabilities: "Capabilities",
    tab_pipeline:     "Pipeline",
    tab_docs:         "Docs",

    ov_title: "deepdive_kai0 — π0.5 dual-arm manipulation deployment",
    ov_desc:  "End-to-end deployment of Physical Intelligence's open-source π0.5 vision-language-action model on the Agilex dual-Piper arm platform with three RealSense cameras. Covers the full loop: training data collection (teleop / DAgger / autonomy replay) → multi-cluster training (gf / uc / js) → single-workstation unified inference (sim01) → real-time arm control.",

    stat_features:    "Features shipped",
    stat_milestones:  "Milestones",
    stat_docs:        "Deployment docs",
    stat_categories:  "Categories",

    arch_title: "System architecture",
    ms_title:   "Recent milestones",

    cap_title: "Capability matrix",
    cap_desc:  "Delivered subsystem capabilities grouped into five categories. Each card shows completion date + status + related docs. Click a doc badge to jump straight to it in the Docs tab.",
    cat_all:   "All",

    pipe_title:          "End-to-end pipeline",
    pipe_desc:           "Three stages from raw data collection to live hardware control, each listing the key subsystems implemented in this project.",
    pipe_stage_collect:  "Data Collection",
    pipe_stage_train:    "Model Training",
    pipe_stage_deploy:   "Live Deployment",
    pipe_collect_items: [
      "Master-Slave dual-arm teleoperation (4-bus CAN)",
      "3 × RealSense cameras (1 × D435 + 2 × D405)",
      "React + FastAPI collection UI",
      "DAgger online correction collection",
      "Autonomy auto-recording",
      "LeRobot v2.1 unified on-disk format",
      "Cross-host incremental TOS sync"
    ],
    pipe_train_items: [
      "gf cluster (shared vePFS)",
      "uc cluster (RoCEv2 multi-host)",
      "js cluster (200 Gb/s InfiniBand)",
      "π0 / π0.5 / π0_fast / π0_rtc variants",
      "Sidecar config + override_asset_id",
      "Dynamic-dataset watcher for resumed training",
      "norm_stats md5 consistency checks"
    ],
    pipe_deploy_items: [
      "sim01 single-host unified inference (dual RTX 5090)",
      "ROS2-native policy node (JAX in-process)",
      "66 ms steady-state / 32 ms via V1 Triton path",
      "RTC chunk-boundary smoothing",
      "Multimodal: depth + EE pose I/O",
      "Action safety clamp (URDF-based)",
      "Live Rerun 3D mesh visualization",
      "Runtime OBSERVE / EXECUTE toggle"
    ],

    docs_title:       "Documentation browser",
    docs_desc:        "24 docs from docs/deployment grouped by topic. Pick one on the left to render its markdown on the right.",
    docs_placeholder: "← pick a doc on the left",
    docs_loading:     "loading…",
    docs_error:       "failed to load"
  }
};

let currentLang = 'zh';

function setLang(lang) {
  if (lang !== 'zh' && lang !== 'en') return;
  currentLang = lang;
  document.documentElement.lang = lang;
  document.getElementById('lang-zh').classList.toggle('active', lang === 'zh');
  document.getElementById('lang-en').classList.toggle('active', lang === 'en');

  const dict = I18N[lang];

  // Plain text: data-i18n="key"
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key] !== undefined) {
      el.textContent = dict[key];
    }
  });

  // List: data-i18n-list="key" → rebuild <li> children
  document.querySelectorAll('[data-i18n-list]').forEach(ul => {
    const key = ul.getAttribute('data-i18n-list');
    const items = dict[key];
    if (Array.isArray(items)) {
      ul.innerHTML = items.map(t => `<li>${escapeHtml(t)}</li>`).join('');
    }
  });

  // Re-render dynamic content (features, milestones, docs index) in new language.
  if (typeof renderFeatures === 'function') renderFeatures();
  if (typeof renderTimeline === 'function') renderTimeline();
  if (typeof renderDocsIndex === 'function') renderDocsIndex();
}

function t(key) {
  return I18N[currentLang][key] || key;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}
