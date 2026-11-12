const byId = (id) => document.getElementById(id);
const feedback = byId("feedback");
const run = byId("run");
const override = byId("override");
const error = byId("error");
const hosted = location.hostname === "aul-ravens.github.io";
const connectPanel = byId("connect-panel");
const connectButton = byId("connect");
const connectError = byId("connect-error");
const status = byId("connection");
let config;
let current;
let modelRoute;
let submittedFeedback;

function updateCount() {
  byId("count").textContent = `${[...feedback.value].length} / ${feedback.maxLength}`;
}

function updateProgress() {
  const steps = byId("steps").querySelectorAll("input");
  const done = [...steps].filter((step) => step.checked).length;
  byId("progress").textContent = `已完成 ${done} / ${steps.length}`;
}

function renderWorkflow(route) {
  current = route;
  byId("decision-title").textContent = route.title;
  byId("criterion").textContent = route.criterion;
  override.value = route.code;
  byId("steps").replaceChildren(...route.steps.map((step) => {
    const label = document.createElement("label");
    label.className = "step";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.addEventListener("change", updateProgress);
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = step.title;
    const detail = document.createElement("small");
    detail.textContent = step.detail;
    copy.append(title, detail);
    label.append(input, copy);
    return label;
  }));
  updateProgress();
  byId("empty").hidden = true;
  byId("workflow").hidden = false;
}

function showError(message) {
  error.textContent = message;
  error.hidden = !message;
}

function apiFetch(path, options = {}) {
  const request = {...options, credentials: "omit", redirect: "error"};
  if (hosted) {
    request.targetAddressSpace = "loopback";
    return fetch(`http://127.0.0.1:8765${path}`, request);
  }
  return fetch(path, request);
}

async function initialize() {
  connectButton.disabled = true;
  connectError.hidden = true;
  status.textContent = "正在连接本地模型";
  try {
    const response = await apiFetch("/api/config");
    if (!response.ok) throw new Error();
    config = await response.json();
    feedback.maxLength = config.max_feedback;
    updateCount();
    byId("samples").replaceChildren();
    for (const [title, content] of config.samples) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = title;
      button.addEventListener("click", () => {
        feedback.value = content;
        updateCount();
        feedback.focus();
      });
      byId("samples").append(button);
    }
    override.replaceChildren();
    for (const route of config.routes) {
      override.add(new Option(route.title, route.code));
    }
    if (current) override.value = current.code;
    status.classList.add("ready");
    status.textContent = `本地模型 · ${config.device === "cuda" ? "GPU" : "CPU"}`;
    const evaluation = config.evaluation;
    byId("local-evaluation").hidden = !evaluation;
    if (evaluation) {
      const p = document.createElement("p");
      p.textContent = `${evaluation.eval_examples} 条合成测试反馈，基座答对 ${evaluation.baseline.correct} 条，LoRA 答对 ${evaluation.lora.correct} 条。`;
      byId("metrics").replaceChildren(p);
    }
    connectPanel.hidden = true;
    showError("");
    run.disabled = false;
  } catch {
    disconnect();
  } finally {
    connectButton.disabled = false;
  }
}

function disconnect() {
  status.classList.remove("ready");
  status.textContent = "本地模型未连接";
  connectPanel.hidden = false;
  connectError.textContent = hosted
    ? "请确认本机服务已启动，且当前站点可访问本地网络。也可直接打开本机入口。"
    : "请启动 WorkPattern 本地服务，然后重新连接。";
  connectError.hidden = false;
  run.disabled = true;
}

connectButton.addEventListener("click", initialize);
feedback.addEventListener("input", updateCount);
byId("clear").addEventListener("click", () => {
  feedback.value = "";
  updateCount();
  showError("");
  feedback.focus();
});
byId("triage-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = feedback.value.trim();
  if (!input) {
    showError("请输入反馈内容。");
    return;
  }
  showError("");
  run.disabled = true;
  run.firstChild.textContent = "正在分流 ";
  try {
    const response = await apiFetch("/api/triage", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({feedback: input}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "本地服务返回了错误。");
    submittedFeedback = input;
    modelRoute = payload.workflow;
    renderWorkflow(modelRoute);
    byId("latency").textContent = `${payload.elapsed_ms} ms`;
  } catch (exception) {
    if (exception instanceof TypeError) disconnect();
    else showError(exception.message);
  } finally {
    run.firstChild.textContent = "分析反馈 ";
    run.disabled = !connectPanel.hidden;
  }
});

override.addEventListener("change", () => {
  renderWorkflow(config.routes.find((route) => route.code === override.value));
});

byId("download").addEventListener("click", () => {
  const checked = [...byId("steps").querySelectorAll("input")];
  const steps = current.steps.map((step, index) => `- [${checked[index].checked ? "x" : " "}] ${step.title}：${step.detail}`);
  const escaped = submittedFeedback.split("\n").map((line) => `> ${line}`).join("\n");
  const content = [
    "# 反馈处理记录", "## 反馈内容", escaped, "## 分流结果",
    `模型建议：${modelRoute.title}\n\n复核结果：${current.title}`, "## 处理步骤", steps.join("\n"),
  ].join("\n\n");
  const url = URL.createObjectURL(new Blob([content], {type: "text/markdown;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "feedback-workflow.md";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

if (hosted) connectPanel.hidden = false;
else initialize();
