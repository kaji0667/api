const state = {
  tasks: [],
  selectedTask: null,
  imageFile: null,
  imageUrl: null,
};

const taskGrid = document.querySelector("#task-grid");
const workspace = document.querySelector("#workspace");
const selectedBadge = document.querySelector("#selected-task-badge");
const inputSummary = document.querySelector("#input-summary");
const dynamicFields = document.querySelector("#dynamic-fields");
const runButton = document.querySelector("#run-button");
const runLabel = document.querySelector("#run-label");
const formMessage = document.querySelector("#form-message");
const resultHeading = document.querySelector("#result-heading");
const resultHelp = document.querySelector("#result-help");
const resultStage = document.querySelector("#result-stage");
const outputTags = document.querySelector("#output-tags");
const interfaceStatusChip = document.querySelector("#interface-status-chip");
const imageInput = document.querySelector("#image-input");
const uploadZone = document.querySelector("#upload-zone");
const uploadEmpty = document.querySelector("#upload-empty");
const previewWrap = document.querySelector("#image-preview-wrap");
const imagePreview = document.querySelector("#image-preview");
const imageName = document.querySelector("#image-name");
const imageSize = document.querySelector("#image-size");
const replaceImage = document.querySelector("#replace-image");
const heroCta = document.querySelector("#hero-cta");
const workspaceNavLinks = document.querySelectorAll("[data-workspace-link]");
const menuToggle = document.querySelector("#menu-toggle");
const mobileMenu = document.querySelector("#mobile-menu");
const quoteSection = document.querySelector("#quote-section");
const pxRing = document.querySelector("#px-ring");
const pxLeft = document.querySelector("#px-left");
const pxRight = document.querySelector("#px-right");
const heroEarthVideos = Array.from(document.querySelectorAll(".hero-earth-video"));
const aeroShardsCanvas = document.querySelector("#aero-shards-canvas");
const ghostFibersCanvas = document.querySelector("#ghost-fibers-canvas");
const cursorGridCanvas = document.querySelector("#cursor-grid-canvas");
const aboutSection = document.querySelector("#about-section");
const aboutCursorGridCanvas = document.querySelector("#about-cursor-grid-canvas");
let refreshCursorGrid = () => {};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setMessage(message = "", kind = "") {
  formMessage.textContent = message;
  formMessage.className = `form-message${kind ? ` ${kind}` : ""}`;
}

function renderPlaceholder(message, detail) {
  resultStage.innerHTML = `
    <div class="result-placeholder">
      <div class="placeholder-orbit" aria-hidden="true">
        <span class="orbit-ring"></span>
        <span class="orbit-dot"></span>
        <span class="orbit-center"></span>
      </div>
      <strong></strong>
      <p></p>
    </div>
  `;
  resultStage.querySelector("strong").textContent = message;
  resultStage.querySelector("p").textContent = detail;
}

function renderLoading(message) {
  resultStage.innerHTML = `
    <div class="result-placeholder">
      <div class="loading-orbit" aria-hidden="true">
        <span class="orbit-ring"></span>
        <span class="orbit-dot"></span>
        <span class="orbit-center"></span>
      </div>
      <strong></strong>
      <p>模型首次使用时需要加载 checkpoint，请稍候…</p>
    </div>
  `;
  resultStage.querySelector("strong").textContent = message;
}

function setInterfaceStatus(task) {
  const available = ["ready", "loaded"].includes(task.interface_status);
  interfaceStatusChip.textContent = available ? "模型可用" : "接口待接入";
  interfaceStatusChip.classList.toggle("ready", available);
}

const taskVisuals = {
  scene: {
    tag: "SCENE",
    shapes: [{ cls: "visual-shape-two" }, { cls: "visual-shape-three" }, { cls: "visual-shape-four" }],
  },
  detection: {
    tag: "DETECT",
    shapes: [{ cls: "visual-shape-two" }, { cls: "visual-shape-three" }, { cls: "visual-shape-four", text: "0.97" }],
  },
  counting: {
    tag: "COUNT",
    shapes: [{ cls: "visual-shape-two", text: "12" }],
  },
  refseg: {
    tag: "SEG",
    shapes: [{ cls: "visual-shape-two" }, { cls: "visual-shape-three", text: "MASK" }],
  },
};

function buildTaskVisual(icon) {
  const config = taskVisuals[icon] || taskVisuals.scene;
  const wrap = document.createElement("div");
  wrap.className = `task-card-visual visual-${icon}`;
  wrap.setAttribute("aria-hidden", "true");
  config.shapes.forEach((shape) => {
    const el = document.createElement("span");
    el.className = shape.cls;
    if (shape.text) el.textContent = shape.text;
    wrap.appendChild(el);
  });
  return wrap;
}

function renderTasks() {
  taskGrid.innerHTML = "";
  state.tasks.forEach((task, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "task-card reveal";
    card.dataset.taskId = task.task_id;
    card.style.setProperty("--accent", task.accent_color);
    card.setAttribute("aria-pressed", "false");
    const number = String(index + 1).padStart(2, "0");
    const tag = (taskVisuals[task.icon] || {}).tag || task.icon;

    const top = document.createElement("div");
    top.className = "task-card-top";
    top.innerHTML = `<span class="task-card-number">${number}</span><span class="task-card-tag">${tag}</span>`;

    const body = document.createElement("div");
    body.className = "task-card-body";
    const title = document.createElement("strong");
    title.textContent = task.title;
    const desc = document.createElement("small");
    desc.textContent = task.description;
    body.append(title, desc);

    const arrow = document.createElement("span");
    arrow.className = "task-card-arrow";
    arrow.innerHTML = `
      <span>${task.action_label}</span>
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <path d="M3 7h8M8 4l3 3-3 3" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    `;

    card.append(top, buildTaskVisual(task.icon), body, arrow);
    card.addEventListener("click", () => selectTask(task.task_id));
    taskGrid.appendChild(card);
  });
  observeRevealElements();
}

function renderDynamicFields(task) {
  dynamicFields.innerHTML = "";
  task.inputs.filter((field) => field.kind !== "image").forEach((field) => {
    const wrapper = document.createElement("div");
    wrapper.className = "dynamic-field";
    wrapper.innerHTML = `
      <label class="field-label" for="field-${field.key}">
        ${field.label}
        <span>${field.required ? "必填" : "选填"}</span>
      </label>
      <textarea id="field-${field.key}" data-field-key="${field.key}" placeholder="${field.placeholder}"></textarea>
      <p class="field-help">${field.help_text}</p>
    `;
    dynamicFields.appendChild(wrapper);
  });
}

function selectTask(taskId) {
  const task = state.tasks.find((item) => item.task_id === taskId);
  if (!task) return;
  state.selectedTask = task;
  document.documentElement.style.setProperty("--task-accent", task.accent_color);
  document.querySelectorAll(".task-card").forEach((card) => {
    const selected = card.dataset.taskId === taskId;
    card.classList.toggle("selected", selected);
    card.setAttribute("aria-pressed", String(selected));
  });
  workspace.hidden = false;
  requestAnimationFrame(refreshCursorGrid);
  workspaceNavLinks.forEach((link) => {
    link.hidden = false;
  });
  closeMobileMenu();
  selectedBadge.textContent = `${task.title} · ${task.technical_name}`;
  selectedBadge.classList.add("active");
  inputSummary.textContent = task.inputs.map((field) => field.label.replace("上传", "")).join(" + ");
  resultHeading.textContent = task.result_title;
  resultHelp.textContent = task.detail;
  runLabel.textContent = task.action_label;
  runButton.disabled = false;
  setMessage("");
  setInterfaceStatus(task);
  renderPlaceholder("结果将在这里呈现", task.detail);
  renderDynamicFields(task);
  outputTags.innerHTML = task.outputs.map((output) => `<span class="output-tag">${output.label}</span>`).join("");
  workspace.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") && !/\.(tif|tiff)$/i.test(file.name)) {
    setMessage("请选择 JPG、PNG、WEBP 或 TIFF 图像。", "error");
    return;
  }
  if (file.size > 40 * 1024 * 1024) {
    setMessage("图像文件过大，请上传不超过 40 MiB 的图像。", "error");
    return;
  }
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageFile = file;
  state.imageUrl = URL.createObjectURL(file);
  imagePreview.src = state.imageUrl;
  imageName.textContent = file.name;
  imageSize.textContent = formatBytes(file.size);
  uploadEmpty.hidden = true;
  previewWrap.hidden = false;
  setMessage("");
}

imageInput.addEventListener("change", (event) => setImageFile(event.target.files[0]));
replaceImage.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  imageInput.click();
});

["dragenter", "dragover"].forEach((name) => {
  uploadZone.addEventListener(name, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  uploadZone.addEventListener(name, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragging");
  });
});

uploadZone.addEventListener("drop", (event) => setImageFile(event.dataTransfer.files[0]));

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
    reader.readAsDataURL(file);
  });
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatLatency(value) {
  const latency = Number(value);
  return Number.isFinite(latency) ? `${latency.toFixed(1)} ms` : "—";
}

function resultFilename(suffix) {
  const stem = (state.imageFile?.name || "analysis-result").replace(/\.[^.]+$/, "");
  return `${stem}_${suffix}.png`;
}

function createResultImage(label, source, className = "") {
  const figure = document.createElement("figure");
  figure.className = `result-image ${className}`.trim();
  const image = document.createElement("img");
  image.src = source;
  image.alt = label;
  const caption = document.createElement("figcaption");
  caption.textContent = label;
  figure.append(image, caption);
  return figure;
}

function renderRefsegResult(result) {
  const images = result?.images || {};
  const summary = result?.summary || {};
  if (!images.overlay || !images.mask || !images.probability) {
    throw new Error("语义分割结果缺少可视化图像。");
  }

  resultStage.innerHTML = "";
  const content = document.createElement("div");
  content.className = "refseg-result";

  if (summary.found_target === false || Number(summary.foreground_pixels) === 0) {
    const warning = document.createElement("div");
    warning.className = "result-warning";
    warning.textContent = "未找到符合描述的目标。请补充位置、颜色或大小信息后重试。";
    content.appendChild(warning);
  }

  if (summary.prompt_translated && summary.model_prompt) {
    const translation = document.createElement("div");
    translation.className = "translation-note";
    const label = document.createElement("span");
    const value = document.createElement("strong");
    label.textContent = "中文提示已转译为";
    value.textContent = summary.model_prompt;
    translation.append(label, value);
    content.appendChild(translation);
  }

  const gallery = document.createElement("div");
  gallery.className = "result-gallery";
  gallery.append(
    createResultImage("目标叠加效果", images.overlay, "result-image-main"),
    createResultImage("二值 Mask", images.mask),
    createResultImage("像素概率图", images.probability),
  );

  const metrics = [
    ["分割阈值", Number(summary.threshold).toFixed(2)],
    ["前景占比", formatPercent(summary.foreground_ratio)],
    ["模型耗时", formatLatency(summary.latency_ms)],
    ["原图尺寸", Array.isArray(summary.original_size) ? `${summary.original_size[1]} × ${summary.original_size[0]}` : "—"],
  ];
  const metricGrid = document.createElement("div");
  metricGrid.className = "result-metrics";
  metrics.forEach(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("span");
    const amount = document.createElement("strong");
    name.textContent = label;
    amount.textContent = value;
    item.append(name, amount);
    metricGrid.appendChild(item);
  });

  const downloads = document.createElement("div");
  downloads.className = "result-downloads";
  [
    ["下载叠加图", "overlay"],
    ["下载 Mask", "mask"],
    ["下载概率图", "probability"],
  ].forEach(([label, key]) => {
    const link = document.createElement("a");
    link.href = images[key];
    link.download = resultFilename(key);
    link.textContent = label;
    downloads.appendChild(link);
  });

  content.append(gallery, metricGrid, downloads);
  resultStage.appendChild(content);
}

function appendMetricGrid(content, metrics) {
  const metricGrid = document.createElement("div");
  metricGrid.className = "result-metrics";
  metrics.forEach(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("span");
    const amount = document.createElement("strong");
    name.textContent = label;
    amount.textContent = value;
    item.append(name, amount);
    metricGrid.appendChild(item);
  });
  content.appendChild(metricGrid);
}

function renderClassificationResult(result) {
  const predictions = Array.isArray(result?.predictions) ? result.predictions : [];
  const summary = result?.summary || {};
  if (!predictions.length) throw new Error("场景分类结果缺少候选类别。");
  resultStage.innerHTML = "";
  const content = document.createElement("div");
  content.className = "classification-result";

  const winner = document.createElement("div");
  winner.className = "classification-winner";
  const winnerLabel = document.createElement("span");
  const winnerClass = document.createElement("strong");
  const winnerProbability = document.createElement("em");
  winnerLabel.textContent = "最可能场景";
  winnerClass.textContent = predictions[0].class_name;
  winnerProbability.textContent = formatPercent(predictions[0].probability);
  winner.append(winnerLabel, winnerClass, winnerProbability);

  const ranking = document.createElement("div");
  ranking.className = "classification-ranking";
  predictions.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "ranking-row";
    const header = document.createElement("div");
    const label = document.createElement("strong");
    const value = document.createElement("span");
    label.textContent = `${index + 1}. ${item.class_name}`;
    value.textContent = formatPercent(item.probability);
    header.append(label, value);
    const track = document.createElement("div");
    track.className = "ranking-track";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(0, Math.min(100, Number(item.probability) * 100))}%`;
    track.appendChild(fill);
    row.append(header, track);
    ranking.appendChild(row);
  });
  content.append(winner, ranking);
  appendMetricGrid(content, [
    ["Top-1 概率", formatPercent(summary.top1_probability)],
    ["候选类别", `${predictions.length} 个`],
    ["模型耗时", formatLatency(summary.latency_ms)],
    ["原图尺寸", Array.isArray(summary.original_size) ? `${summary.original_size[1]} × ${summary.original_size[0]}` : "—"],
  ]);
  resultStage.appendChild(content);
}

function renderBoxTaskResult(result, taskId) {
  const summary = result?.summary || {};
  const images = result?.images || {};
  const boxes = Array.isArray(result?.boxes) ? result.boxes : [];
  const scores = Array.isArray(result?.scores) ? result.scores : [];
  if (!images.visualization || boxes.length !== scores.length) {
    throw new Error("检测结果缺少标注图或框置信度不匹配。");
  }
  const isCounting = taskId === "counting";
  const count = Number(isCounting ? summary.count : summary.detection_count) || 0;
  resultStage.innerHTML = "";
  const content = document.createElement("div");
  content.className = "box-task-result";
  if (count === 0) {
    const warning = document.createElement("div");
    warning.className = "result-warning";
    warning.textContent = isCounting ? "没有检测到要统计的目标。" : "没有检测到符合描述的目标。";
    content.appendChild(warning);
  }
  content.appendChild(createResultImage("目标标注结果", images.visualization, "result-image-main"));
  appendMetricGrid(content, [
    [isCounting ? "目标数量" : "检测数量", `${count}`],
    [isCounting ? "目标类别" : "目标描述", summary.target_class || summary.prompt || "—"],
    ["模型耗时", formatLatency(summary.latency_ms)],
    ["置信阈值", Number(summary.conf_thres).toFixed(2)],
  ]);

  if (boxes.length) {
    const tableWrap = document.createElement("div");
    tableWrap.className = "box-table-wrap";
    const table = document.createElement("table");
    table.className = "box-table";
    table.innerHTML = "<thead><tr><th>#</th><th>置信度</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr></thead>";
    const body = document.createElement("tbody");
    boxes.slice(0, 50).forEach((box, index) => {
      const row = document.createElement("tr");
      const values = [index + 1, Number(scores[index]).toFixed(3), ...box.map((value) => Number(value).toFixed(1))];
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
    table.appendChild(body);
    tableWrap.appendChild(table);
    content.appendChild(tableWrap);
  }

  const downloads = document.createElement("div");
  downloads.className = "result-downloads";
  const link = document.createElement("a");
  link.href = images.visualization;
  link.download = resultFilename(isCounting ? "counting" : "detection");
  link.textContent = "下载标注图";
  downloads.appendChild(link);
  content.appendChild(downloads);
  resultStage.appendChild(content);
}

function renderResult(taskId, result) {
  if (taskId === "classification") {
    renderClassificationResult(result);
    return;
  }
  if (taskId === "detection" || taskId === "counting") {
    renderBoxTaskResult(result, taskId);
    return;
  }
  if (taskId === "refseg") {
    renderRefsegResult(result);
    return;
  }
  renderPlaceholder("任务执行完成", "该任务结果渲染器将在对应模型接口接入时补充。");
}

runButton.addEventListener("click", async () => {
  if (!state.selectedTask) {
    setMessage("请先选择一项分析任务。", "error");
    return;
  }
  if (!state.imageFile) {
    setMessage("请先上传一张遥感图像。", "error");
    return;
  }

  const inputs = {};
  for (const field of state.selectedTask.inputs.filter((item) => item.kind !== "image")) {
    const value = document.querySelector(`[data-field-key="${field.key}"]`)?.value.trim() || "";
    if (field.required && !value) {
      setMessage(`请填写"${field.label}"。`, "error");
      return;
    }
    inputs[field.key] = value;
  }

  runButton.disabled = true;
  setMessage("正在读取图像并准备推理…");
  renderLoading("正在分析图像");
  try {
    inputs.image = {
      name: state.imageFile.name,
      data_url: await fileToDataUrl(state.imageFile),
    };
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: state.selectedTask.task_id, inputs }),
    });
    const payload = await response.json();
    if (response.ok) {
      renderResult(state.selectedTask.task_id, payload.result);
      const emptyRefseg = state.selectedTask.task_id === "refseg"
        && Number(payload.result?.summary?.foreground_pixels) === 0;
      const emptyBoxes = ["detection", "counting"].includes(state.selectedTask.task_id)
        && Number(payload.result?.summary?.detection_count ?? payload.result?.summary?.count) === 0;
      setMessage(
        emptyRefseg || emptyBoxes ? "未找到符合描述的目标，请调整输入后重试。" : "任务执行完成。",
        emptyRefseg || emptyBoxes ? "error" : "success",
      );
      interfaceStatusChip.textContent = "模型已加载";
      interfaceStatusChip.classList.add("ready");
    } else if (payload.status === "interface_pending") {
      setMessage(payload.message, "error");
      resultHelp.textContent = `${state.selectedTask.title}的界面、输入校验和路由已经连通；下一步接入实际模型 Adapter。`;
      renderPlaceholder("该任务暂未接入模型", resultHelp.textContent);
    } else {
      const message = payload.message || payload.error || "任务请求失败。";
      setMessage(message, "error");
      renderPlaceholder("分析未完成", message);
    }
  } catch (error) {
    const message = error instanceof Error && error.message.includes("结果缺少")
      ? error.message
      : "无法完成请求，请确认本地服务正在运行且图像可以读取。";
    setMessage(message, "error");
    renderPlaceholder("分析未完成", message);
  } finally {
    runButton.disabled = false;
  }
});

let revealObserver = null;

function observeRevealElements() {
  if (revealObserver) revealObserver.disconnect();
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReducedMotion) {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("visible"));
    return;
  }
  revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });
  document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));
}

observeRevealElements();

if (heroCta) {
  heroCta.addEventListener("click", () => {
    const tasksSection = document.querySelector("#tasks-section");
    if (tasksSection) tasksSection.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function closeMobileMenu() {
  if (!menuToggle || !mobileMenu) return;
  document.body.classList.remove("menu-open");
  menuToggle.classList.remove("open");
  menuToggle.setAttribute("aria-expanded", "false");
  mobileMenu.setAttribute("aria-hidden", "true");
}

if (menuToggle && mobileMenu) {
  menuToggle.addEventListener("click", () => {
    const open = document.body.classList.toggle("menu-open");
    menuToggle.classList.toggle("open", open);
    menuToggle.setAttribute("aria-expanded", String(open));
    mobileMenu.setAttribute("aria-hidden", String(!open));
  });
  mobileMenu.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeMobileMenu);
  });
}

function initQuoteParallax() {
  if (!quoteSection || !pxRing || !pxLeft || !pxRight) return;
  const layers = [pxRing, pxLeft, pxRight];
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    layers.forEach((layer) => {
      layer.style.opacity = "1";
      layer.style.transform = "none";
    });
    return;
  }
  const lerp = (from, to, amount) => from + (to - from) * amount;
  const clamp01 = (value) => Math.min(1, Math.max(0, value));
  let progress = 0;
  let cloudProgress = 0;
  function update() {
    const rect = quoteSection.getBoundingClientRect();
    const viewport = window.innerHeight;
    const target = clamp01((viewport - rect.top) / (viewport + rect.height));
    progress = lerp(progress, target, 0.06);
    cloudProgress = lerp(cloudProgress, target, 0.04);
    const ringY = 120 + (-160 - 120) * progress;
    pxRing.style.transform = `translate3d(0, ${ringY.toFixed(2)}px, 0)`;
    pxRing.style.opacity = clamp01(progress * 3).toFixed(3);
    const slide = clamp01((cloudProgress - 0.12) / 0.8);
    const cloudY = cloudProgress * -50;
    const cloudOpacity = clamp01(slide * 1.8).toFixed(3);
    pxLeft.style.opacity = cloudOpacity;
    pxLeft.style.transform = `translate3d(${(-200 + 200 * slide).toFixed(2)}px, ${cloudY.toFixed(2)}px, 0)`;
    pxRight.style.opacity = cloudOpacity;
    pxRight.style.transform = `translate3d(${(200 - 200 * slide).toFixed(2)}px, ${cloudY.toFixed(2)}px, 0) scaleX(-1)`;
  }
  function frame() {
    update();
    requestAnimationFrame(frame);
  }
  update();
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  requestAnimationFrame(frame);
}

initQuoteParallax();

function initHeroEarthVideo() {
  if (heroEarthVideos.length < 2) return;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    heroEarthVideos.forEach((video) => video.pause());
    return;
  }

  const crossfadeSeconds = 1.2;
  let activeIndex = 0;
  let transitioning = false;

  heroEarthVideos.forEach((video) => {
    video.muted = true;
    video.playbackRate = 0.72;
  });

  function startCrossfade() {
    const current = heroEarthVideos[activeIndex];
    if (transitioning) return;
    transitioning = true;
    const nextIndex = (activeIndex + 1) % heroEarthVideos.length;
    const next = heroEarthVideos[nextIndex];
    next.currentTime = 0;
    next.playbackRate = 0.72;
    next.play().catch(() => {});
    next.classList.add("is-active");
    current.classList.remove("is-active");

    window.setTimeout(() => {
      current.pause();
      current.currentTime = 0;
      activeIndex = nextIndex;
      transitioning = false;
    }, crossfadeSeconds * 1000);
  }

  function maybeCrossfade() {
    const current = heroEarthVideos[activeIndex];
    if (transitioning || !Number.isFinite(current.duration)) return;
    if (current.duration - current.currentTime <= crossfadeSeconds + 0.15) startCrossfade();
  }

  heroEarthVideos.forEach((video) => {
    video.addEventListener("timeupdate", maybeCrossfade);
    video.addEventListener("ended", () => {
      if (video === heroEarthVideos[activeIndex]) startCrossfade();
    });
  });
  heroEarthVideos[0].play().catch(() => {});
}

initHeroEarthVideo();

function initGhostFibers() {
  if (!ghostFibersCanvas) return;
  const gl = ghostFibersCanvas.getContext("webgl2", {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    powerPreference: "low-power",
  });
  if (!gl) return;

  const vertexSource = `#version 300 es
  precision highp float;
  const vec2 POSITIONS[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
  );
  void main() {
    gl_Position = vec4(POSITIONS[gl_VertexID], 0.0, 1.0);
  }
  `;

  const fragmentSource = `#version 300 es
  precision highp float;

  uniform vec2 uResolution;
  uniform float uTime;
  uniform float uSpeed;
  uniform float uScale;
  uniform float uRotation;
  uniform float uLayers;
  uniform float uWaveAmplitude;
  uniform float uWaveFrequency;
  uniform float uWaveSpeed;
  uniform float uLayerSpeed;
  uniform float uTwist;
  uniform float uTwistFrequency;
  uniform float uTwistSpeed;
  uniform float uLineFrequency;
  uniform float uLineSpacing;
  uniform float uLineSharpness;
  uniform float uGlowFalloff;
  uniform float uGlowIntensity;
  uniform float uBrightness;
  uniform float uBlueBoost;
  uniform float uVignette;
  uniform float uGrain;
  uniform float uRotationSpeed;
  uniform vec3 uLineColor;
  uniform vec3 uGlowColor;

  out vec4 fragColor;

  #define MAX_LAYERS 10

  mat2 rotate2d(float angle) {
    float sine = sin(angle);
    float cosine = cos(angle);
    return mat2(cosine, -sine, sine, cosine);
  }

  float grainHash(vec2 point) {
    point = floor(point);
    float hash = 52.9829189 * fract(dot(point, vec2(0.065, 0.005)));
    return fract(hash);
  }

  float layeredGrain(vec2 fragmentPixel) {
    vec2 point = mod(fragmentPixel + vec2(uTime * 30.0, -uTime * 21.0), 1024.0);
    vec2 rotated = mat2(0.8, -0.5, 0.5, 0.8) * point;
    float grain = 0.0;
    grain += 0.40 * grainHash(rotated);
    grain += 0.25 * grainHash(rotated * 2.0 + 17.0);
    grain += 0.20 * grainHash(rotated * 4.0 + 47.0);
    grain += 0.10 * grainHash(rotated * 8.0 + 113.0);
    grain += 0.05 * grainHash(rotated * 16.0 + 191.0);
    return grain;
  }

  void main() {
    vec2 resolution = max(uResolution, vec2(1.0));
    vec2 uv = (2.0 * gl_FragCoord.xy - resolution) / resolution.y;
    float time = uTime * uSpeed;
    vec3 centerTone = max(uLineColor * 0.85567 - uGlowColor * 0.06186, vec3(0.0));
    vec3 cloudTone = uLineColor * 0.19588 + uGlowColor * 0.2268;
    vec2 p = uv;
    p /= max(uScale, 0.05);
    p = rotate2d(radians(uRotation) + time * uRotationSpeed) * p;
    vec3 color = vec3(0.0);

    for (int index = 0; index < MAX_LAYERS; index++) {
      float fi = float(index) + 1.0;
      if (fi > uLayers) break;

      p += uWaveAmplitude * sin(p.yx * fi * uWaveFrequency + time * (uWaveSpeed + fi * uLayerSpeed));

      float radius = length(p);
      float polarAngle = atan(p.y, p.x);
      polarAngle += sin(radius * uTwistFrequency - time * uTwistSpeed + fi) * uTwist;
      p = vec2(cos(polarAngle), sin(polarAngle)) * radius;

      float lines = abs(sin(p.x * (uLineFrequency + fi * uLineSpacing) + sin(p.y * 3.0 + time)));
      lines = pow(max(0.0, 1.0 - lines), uLineSharpness);
      color += uLineColor * lines / fi;

      float glow = exp(-uGlowFalloff * abs(sin(p.x * 3.0 + time + fi)));
      color += uGlowColor * glow * uGlowIntensity / (fi * 2.0);
    }

    float center = exp(-2.2 * dot(uv, uv));
    color += centerTone * center;

    float cloud = exp(-1.5 * length(uv + vec2(sin(time * 0.3) * 0.25, cos(time * 0.25) * 0.18)));
    color += cloudTone * cloud;

    float vignette = 1.0 - smoothstep(0.35, 1.45, length(uv));
    color *= mix(1.0 - uVignette, 1.0, vignette);
    color = 1.0 - exp(-color * uBrightness);
    color.b *= uBlueBoost;

    vec3 outputColor = color;
    float fiberAlpha = clamp(max(max(color.r, color.g), color.b) * 0.82, 0.0, 0.62);
    float noise = (layeredGrain(gl_FragCoord.xy) - 0.5) * uGrain;
    outputColor = clamp(outputColor + noise * 0.35, 0.0, 1.0);
    fragColor = vec4(outputColor, fiberAlpha);
  }
  `;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) return;

  const program = gl.createProgram();
  if (!program) return;
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return;
  }

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.useProgram(program);

  const uniforms = {};
  [
    "uResolution", "uTime", "uSpeed", "uScale", "uRotation", "uLayers",
    "uWaveAmplitude", "uWaveFrequency", "uWaveSpeed", "uLayerSpeed", "uTwist",
    "uTwistFrequency", "uTwistSpeed", "uLineFrequency", "uLineSpacing",
    "uLineSharpness", "uGlowFalloff", "uGlowIntensity", "uBrightness", "uBlueBoost",
    "uVignette", "uGrain", "uRotationSpeed", "uLineColor", "uGlowColor",
  ].forEach((name) => {
    uniforms[name] = gl.getUniformLocation(program, name);
  });

  gl.uniform1f(uniforms.uSpeed, 0.2);
  gl.uniform1f(uniforms.uScale, 2);
  gl.uniform1f(uniforms.uRotation, 0);
  gl.uniform1f(uniforms.uRotationSpeed, 0.25);
  gl.uniform1f(uniforms.uLayers, 4);
  gl.uniform1f(uniforms.uWaveAmplitude, 0.015);
  gl.uniform1f(uniforms.uWaveFrequency, 3);
  gl.uniform1f(uniforms.uWaveSpeed, 0.15);
  gl.uniform1f(uniforms.uLayerSpeed, 0.08);
  gl.uniform1f(uniforms.uTwist, 0.1);
  gl.uniform1f(uniforms.uTwistFrequency, 5);
  gl.uniform1f(uniforms.uTwistSpeed, 1.2);
  gl.uniform1f(uniforms.uLineFrequency, 5);
  gl.uniform1f(uniforms.uLineSpacing, 2);
  gl.uniform1f(uniforms.uLineSharpness, 16);
  gl.uniform1f(uniforms.uGlowFalloff, 10);
  gl.uniform1f(uniforms.uGlowIntensity, 0.95);
  gl.uniform1f(uniforms.uBrightness, 1.25);
  gl.uniform1f(uniforms.uBlueBoost, 1.06);
  gl.uniform1f(uniforms.uVignette, 0.88);
  gl.uniform1f(uniforms.uGrain, 0.035);
  gl.uniform3f(uniforms.uLineColor, 0.066667, 0.086275, 0.227451);
  gl.uniform3f(uniforms.uGlowColor, 0.192157, 0.278431, 0.662745);

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let width = 1;
  let height = 1;
  let visible = true;
  let frameId = 0;
  let elapsed = 0;
  let previousTime = performance.now();
  let lastRenderTime = 0;

  function resize() {
    const bounds = ghostFibersCanvas.getBoundingClientRect();
    const dpr = Math.min(Math.max(window.devicePixelRatio || 1, 0.5), 1);
    width = Math.max(1, Math.floor(bounds.width));
    height = Math.max(1, Math.floor(bounds.height));
    ghostFibersCanvas.width = Math.max(1, Math.floor(width * dpr));
    ghostFibersCanvas.height = Math.max(1, Math.floor(height * dpr));
    gl.viewport(0, 0, ghostFibersCanvas.width, ghostFibersCanvas.height);
    gl.useProgram(program);
    gl.uniform2f(uniforms.uResolution, ghostFibersCanvas.width, ghostFibersCanvas.height);
    draw();
  }

  function draw() {
    if (gl.isContextLost()) return;
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniform1f(uniforms.uTime, elapsed);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function canAnimate() {
    return visible && !document.hidden && !reducedMotion.matches;
  }

  function loop(now) {
    frameId = 0;
    if (!canAnimate()) return;
    const delta = Math.min((now - previousTime) / 1000, 0.1);
    previousTime = now;
    elapsed += delta;
    if (now - lastRenderTime >= 1000 / 60 - 0.5) {
      draw();
      lastRenderTime = now;
    }
    frameId = requestAnimationFrame(loop);
  }

  function wake() {
    if (!canAnimate() || frameId) return;
    previousTime = performance.now();
    frameId = requestAnimationFrame(loop);
  }

  function pause() {
    if (!frameId) return;
    cancelAnimationFrame(frameId);
    frameId = 0;
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(ghostFibersCanvas);

  const visibilityObserver = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (canAnimate()) wake();
    else pause();
  }, { threshold: 0.02 });
  visibilityObserver.observe(ghostFibersCanvas);

  document.addEventListener("visibilitychange", () => {
    if (canAnimate()) wake();
    else pause();
  });
  reducedMotion.addEventListener("change", () => {
    if (canAnimate()) wake();
    else {
      pause();
      draw();
    }
  });
  ghostFibersCanvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    pause();
  });

  resize();
  draw();
  wake();
}

initGhostFibers();

function mountCursorGrid(container, canvas) {
  if (!container || !canvas) return null;
  const context = canvas.getContext("2d", { alpha: true });
  if (!context) return;

  const cellSize = 70;
  const radius = 130;
  const holdTime = 400;
  const fadeDuration = 800;
  const lineWidth = 1.05;
  const maxOpacity = 0.42;
  const clickOpacity = 0.2;
  const gridOpacity = 0;
  const color = [49, 95, 159];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let width = 1;
  let height = 1;
  let cols = 1;
  let rows = 1;
  let offsetX = 0;
  let offsetY = 0;
  let alphas = new Float32Array(1);
  let touched = new Float64Array(1);
  let frameId = 0;
  let previousTime = performance.now();
  let visible = false;

  function rebuild() {
    const bounds = container.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = Math.max(1, Math.round(bounds.width));
    height = Math.max(1, Math.round(bounds.height));
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(width / cellSize) + 1;
    rows = Math.ceil(height / cellSize) + 1;
    offsetX = (width - cols * cellSize) / 2;
    offsetY = (height - rows * cellSize) / 2;
    alphas = new Float32Array(cols * rows);
    touched = new Float64Array(cols * rows);
    draw(performance.now());
  }

  function drawLattice() {
    if (gridOpacity <= 0) return;
    context.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${gridOpacity})`;
    context.lineWidth = 1;
    context.beginPath();
    for (let column = 0; column <= cols; column += 1) {
      const x = Math.round(offsetX + column * cellSize) + 0.5;
      context.moveTo(x, 0);
      context.lineTo(x, height);
    }
    for (let row = 0; row <= rows; row += 1) {
      const y = Math.round(offsetY + row * cellSize) + 0.5;
      context.moveTo(0, y);
      context.lineTo(width, y);
    }
    context.stroke();
  }

  function energize(pointerX, pointerY) {
    if (reducedMotion.matches || width <= 1 || height <= 1) return;
    const now = performance.now();
    const minCol = Math.max(0, Math.floor((pointerX - radius - offsetX) / cellSize));
    const maxCol = Math.min(cols - 1, Math.floor((pointerX + radius - offsetX) / cellSize));
    const minRow = Math.max(0, Math.floor((pointerY - radius - offsetY) / cellSize));
    const maxRow = Math.min(rows - 1, Math.floor((pointerY + radius - offsetY) / cellSize));

    for (let row = minRow; row <= maxRow; row += 1) {
      for (let column = minCol; column <= maxCol; column += 1) {
        const index = row * cols + column;
        const centerX = offsetX + column * cellSize + cellSize / 2;
        const centerY = offsetY + row * cellSize + cellSize / 2;
        const distance = Math.hypot(centerX - pointerX, centerY - pointerY);
        if (distance > radius) continue;
        const normalized = 1 - distance / radius;
        const smooth = normalized * normalized * (3 - 2 * normalized);
        const level = smooth * maxOpacity;
        if (level > alphas[index]) alphas[index] = level;
        touched[index] = now;
      }
    }
  }

  function revealAll() {
    if (reducedMotion.matches || width <= 1 || height <= 1) return;
    const now = performance.now();
    for (let index = 0; index < alphas.length; index += 1) {
      alphas[index] = Math.max(alphas[index], clickOpacity);
      touched[index] = now;
    }
  }

  function draw(now) {
    frameId = 0;
    const delta = Math.min(now - previousTime, 50);
    previousTime = now;
    context.clearRect(0, 0, width, height);
    drawLattice();

    const fadeStep = delta / Math.max(fadeDuration, 16);
    const half = cellSize / 2;
    let anyVisible = false;

    for (let index = 0; index < alphas.length; index += 1) {
      let alpha = alphas[index];
      if (alpha <= 0) continue;
      if (now - touched[index] > holdTime) {
        alpha = Math.max(0, alpha - fadeStep * maxOpacity);
        alphas[index] = alpha;
        if (alpha <= 0) continue;
      }
      anyVisible = true;

      const column = index % cols;
      const row = Math.floor(index / cols);
      const centerX = offsetX + column * cellSize + half;
      const centerY = offsetY + row * cellSize + half;
      const gradient = context.createRadialGradient(centerX, centerY, 1, centerX, centerY, cellSize);
      gradient.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`);
      gradient.addColorStop(1, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0)`);
      context.beginPath();
      context.rect(centerX - half + 0.5, centerY - half + 0.5, cellSize - 1, cellSize - 1);
      context.strokeStyle = gradient;
      context.lineWidth = lineWidth;
      context.stroke();
    }

    if (anyVisible && visible && !document.hidden && !reducedMotion.matches) {
      frameId = requestAnimationFrame(draw);
    }
  }

  function wake() {
    if (frameId || !visible || document.hidden || reducedMotion.matches) return;
    previousTime = performance.now();
    frameId = requestAnimationFrame(draw);
  }

  function stop() {
    if (!frameId) return;
    cancelAnimationFrame(frameId);
    frameId = 0;
  }

  container.addEventListener("pointermove", (event) => {
    const bounds = container.getBoundingClientRect();
    energize(event.clientX - bounds.left, event.clientY - bounds.top);
    wake();
  }, { passive: true });
  container.addEventListener("pointerdown", () => {
    revealAll();
    wake();
  }, { passive: true });

  const resizeObserver = new ResizeObserver(rebuild);
  resizeObserver.observe(container);
  const visibilityObserver = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!visible) stop();
  }, { threshold: 0.02 });
  visibilityObserver.observe(container);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
    else wake();
  });
  reducedMotion.addEventListener("change", () => {
    stop();
    rebuild();
  });

  rebuild();
  return rebuild;
}

function initCursorGrid() {
  const rebuildWorkspaceGrid = mountCursorGrid(workspace, cursorGridCanvas);
  if (rebuildWorkspaceGrid) refreshCursorGrid = rebuildWorkspaceGrid;
  mountCursorGrid(aboutSection, aboutCursorGridCanvas);
}

initCursorGrid();

function initAeroShards() {
  if (!aeroShardsCanvas) return;
  const context = aeroShardsCanvas.getContext("2d", { alpha: true });
  if (!context) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const glowCanvas = document.createElement("canvas");
  const glowContext = glowCanvas.getContext("2d", { alpha: true });
  if (!glowContext) return;

  let randomState = 0x202603;
  function seededRandom() {
    randomState += 0x6D2B79F5;
    let value = randomState;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  const shards = Array.from({ length: 760 }, (_, index) => {
    const lane = ((seededRandom() + seededRandom() + seededRandom()) / 3) * 2 - 1;
    return {
      phase: seededRandom(),
      lane,
      depth: seededRandom() * 2 - 1,
      size: 0.5 + seededRandom(),
      spin: seededRandom() * Math.PI * 2,
      spinRate: (seededRandom() - 0.5) * 1.4,
      wave: seededRandom() * Math.PI * 2,
      tone: seededRandom(),
      accent: index % 19 === 0,
    };
  });

  let width = 1;
  let height = 1;
  let dpr = 1;
  let visible = true;
  let frameId = 0;
  let previousTime = performance.now();

  function resizeCanvas() {
    const bounds = aeroShardsCanvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = Math.max(1, Math.round(bounds.width));
    height = Math.max(1, Math.round(bounds.height));
    aeroShardsCanvas.width = Math.round(width * dpr);
    aeroShardsCanvas.height = Math.round(height * dpr);
    glowCanvas.width = aeroShardsCanvas.width;
    glowCanvas.height = aeroShardsCanvas.height;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    glowContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function mix(start, end, amount) {
    return Math.round(start + (end - start) * amount);
  }

  function drawShard(target, shard, time, glowPass = false) {
    const progress = shard.phase;
    const tau = Math.PI * 2;
    const primary = progress * tau * 1.12 + 0.45;
    const secondary = progress * tau * 3.1 - 0.8;
    let x = width * (1.16 - progress * 1.34);
    const centerY = height * (0.49 + Math.sin(primary) * 0.115 + Math.sin(secondary) * 0.052);
    const tangentX = -width * 1.34;
    const tangentY = height * (
      Math.cos(primary) * 0.115 * tau * 1.12
      + Math.cos(secondary) * 0.052 * tau * 3.1
    );
    const travelAngle = Math.atan2(tangentY, tangentX);
    const normalX = -Math.sin(travelAngle);
    const normalY = Math.cos(travelAngle);
    const ribbonWidth = height * (0.026 + 0.19 * Math.pow(Math.sin(Math.PI * progress), 1.35));
    const turbulence = Math.sin(progress * 37.699 + shard.wave + time * 0.00028) * height * 0.013;
    const offset = shard.lane * ribbonWidth + shard.depth * height * 0.024 + turbulence;
    x += normalX * offset + shard.depth * 22;
    const y = centerY + normalY * offset;
    const depthAmount = (shard.depth + 1) * 0.5;
    const perspective = 0.56 + depthAmount * 1.02;
    const length = (4 + shard.size * 10) * perspective;
    const thickness = (1.2 + shard.size * 2.5) * perspective;
    const edgeFade = Math.min(1, progress * 10, (1 - progress) * 10);
    const alpha = edgeFade * (0.18 + depthAmount * 0.48) * (0.72 + shard.tone * 0.28);
    const rotation = travelAngle + shard.spin + Math.sin(shard.wave + time * 0.00018) * 0.18;
    const accentMix = shard.accent ? 0.76 : 0.18 + shard.tone * 0.45;
    const red = mix(25, 132, accentMix);
    const green = mix(52, 151, accentMix);
    const blue = mix(110, 232, accentMix);

    target.save();
    target.translate(x, y);
    target.rotate(rotation);
    target.globalAlpha = glowPass ? alpha * 0.72 : alpha;
    target.fillStyle = `rgb(${red} ${green} ${blue})`;
    target.beginPath();
    target.moveTo(-length * 0.58, 0);
    target.lineTo(-length * 0.08, -thickness);
    target.lineTo(length * 0.58, -thickness * 0.12);
    target.lineTo(length * 0.12, thickness);
    target.closePath();
    target.fill();

    if (!glowPass && depthAmount > 0.46) {
      target.globalAlpha = alpha * 0.52;
      target.strokeStyle = `rgb(${mix(red, 218, 0.62)} ${mix(green, 232, 0.62)} 255)`;
      target.lineWidth = Math.max(0.45, perspective * 0.52);
      target.beginPath();
      target.moveTo(-length * 0.42, -thickness * 0.08);
      target.lineTo(length * 0.42, -thickness * 0.12);
      target.stroke();
    }
    target.restore();
  }

  function render(time) {
    frameId = 0;
    context.clearRect(0, 0, width, height);
    glowContext.clearRect(0, 0, width, height);
    const elapsed = Math.min(40, Math.max(0, time - previousTime));
    previousTime = time;
    shards.forEach((shard) => {
      if (!reduceMotion.matches) {
        shard.phase = (shard.phase + elapsed * (0.000014 + (shard.depth + 1) * 0.000004)) % 1;
        shard.spin += elapsed * 0.000035 * shard.spinRate;
      }
      drawShard(glowContext, shard, time, true);
    });

    context.save();
    context.globalCompositeOperation = "source-over";
    context.globalAlpha = 0.66;
    context.filter = "blur(9px)";
    context.drawImage(glowCanvas, 0, 0, glowCanvas.width, glowCanvas.height, 0, 0, width, height);
    context.restore();

    context.globalCompositeOperation = "screen";
    shards.forEach((shard) => drawShard(context, shard, time, false));
    context.globalCompositeOperation = "source-over";
    if (visible && !document.hidden && !reduceMotion.matches) frameId = requestAnimationFrame(render);
  }

  function wake() {
    if (frameId || !visible || document.hidden) return;
    previousTime = performance.now();
    frameId = requestAnimationFrame(render);
  }

  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
    wake();
  });
  resizeObserver.observe(aeroShardsCanvas);

  const visibilityObserver = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!visible && frameId) {
      cancelAnimationFrame(frameId);
      frameId = 0;
    } else {
      wake();
    }
  }, { threshold: 0.05 });
  visibilityObserver.observe(aeroShardsCanvas);

  document.addEventListener("visibilitychange", wake);
  reduceMotion.addEventListener("change", () => {
    if (frameId) cancelAnimationFrame(frameId);
    frameId = 0;
    render(performance.now());
  });
  resizeCanvas();
  render(performance.now());
}

initAeroShards();

async function initialize() {
  try {
    const response = await fetch("/api/tasks");
    if (!response.ok) throw new Error("Task API unavailable");
    const payload = await response.json();
    state.tasks = payload.tasks || [];
    renderTasks();
  } catch (error) {
    taskGrid.innerHTML = '<div class="loading-card">任务配置读取失败，请重新启动服务。</div>';
  }
}

initialize();
