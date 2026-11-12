const workflowCopy = {
  A: {
    title: "Start an incident response",
    criterion: "A core task is widely blocked, or money or data integrity is at risk.",
    steps: [
      ["Confirm the impact", "Check when it started, who is affected, the failure rate, and recent releases."],
      ["Assign an incident owner", "Have the on-call lead assess throttling, disabling the affected entry point, or rolling back. Record the decision."],
      ["Verify recovery", "Use monitoring and failed examples to retest the core workflow. Record when it recovered and what still needs attention."],
    ],
  },
  B: {
    title: "Reproduce and log the defect",
    criterion: "An existing feature fails under known conditions. The impact is limited or users have a workaround.",
    steps: [
      ["Reproduce the issue", "Record the account type, version, device, steps, and expected and actual behavior."],
      ["File a defect", "Include evidence, impact, and any workaround. Assign it to the owner of the affected feature."],
      ["Verify the fix", "Retest the original scenario and nearby workflows. Record the version and result."],
    ],
  },
  C: {
    title: "Validate the feature request",
    criterion: "The feature works as designed. The user wants a change and has described their task or goal.",
    steps: [
      ["Understand the task", "Identify who has the problem, when it occurs, and how they handle it today."],
      ["Check how often it happens", "Interview a sample of users or review usage data. Identify the goal behind the requested change."],
      ["Prepare a proposal", "Summarize the impact, success measures, evidence, and the smallest change worth testing."],
    ],
  },
  D: {
    title: "Link to the existing ticket",
    criterion: "A known ticket has the same trigger and symptoms, with no new severe impact.",
    steps: [
      ["Verify the match", "Check the ticket number, trigger, symptoms, and current status."],
      ["Add new evidence", "Record when this occurred, how many users were affected, and any reproduction details."],
      ["Keep the reporter informed", "Update the original ticket and note what progress to share with the reporter."],
    ],
  },
  E: {
    title: "Ask for missing details",
    criterion: "The report lacks enough detail about the symptoms, environment, or user goal to choose a workflow.",
    steps: [
      ["Clarify the problem", "Ask what the user was trying to do, what happened, and what they expected."],
      ["Collect evidence", "Ask for the time, device or version, affected users, and screenshots with sensitive details removed."],
      ["Triage again", "Review the updated report and record what to communicate and when to follow up."],
    ],
  },
};

const samples = {
  incident: {
    title: "Service outage",
    feedback: "After the morning release, paying customers in several regions could no longer place orders. Payment success fell from 97% to 24% in 15 minutes. Signing in again did not help.",
  },
  existing: {
    title: "Known issue",
    feedback: "Two iOS users stopped receiving verification codes after changing their phone numbers. The trigger, device version, and symptoms match open ticket FB-218, with no new impact.",
  },
  request: {
    title: "Feature request",
    feedback: "Our operations team exports reports from five channels every week and combines them by hand. Reporting works as intended, but the team wants a bulk export to cut down on repetitive work.",
  },
  details: {
    title: "Missing details",
    feedback: "A user said the latest release does not feel right and asked for a quick fix. They did not say what they were trying to do, what happened, or what they expected.",
  },
};

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
const latency = byId("latency");
let current;
let modelRoute;
let submittedFeedback;

const menuToggle = byId("menu-toggle");
const siteMenu = byId("site-menu");
const navigation = siteMenu.closest(".navigation");

function setMenu(open) {
  menuToggle.setAttribute("aria-expanded", String(open));
  navigation.toggleAttribute("data-menu-open", open);
}

function closeMenus() {
  setMenu(false);
  siteMenu.querySelectorAll("details[open]").forEach((group) => { group.open = false; });
}

menuToggle.addEventListener("click", () => {
  if (menuToggle.getAttribute("aria-expanded") === "true") closeMenus();
  else setMenu(true);
});
siteMenu.addEventListener("click", (event) => {
  if (event.target.closest("a")) closeMenus();
});
document.addEventListener("click", (event) => {
  if (!navigation.contains(event.target)) closeMenus();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !navigation.contains(document.activeElement)) return;
  const trigger = siteMenu.querySelector("details[open] summary");
  if (menuToggle.getAttribute("aria-expanded") === "true" && menuToggle.getClientRects().length) menuToggle.focus();
  else if (trigger) trigger.focus();
  closeMenus();
});

function updateCount() {
  byId("count").textContent = `${[...feedback.value].length} / ${feedback.maxLength}`;
}

function updateProgress() {
  const steps = byId("steps").querySelectorAll("input");
  const done = [...steps].filter((step) => step.checked).length;
  byId("progress").textContent = `${done} of ${steps.length} complete`;
}

function renderWorkflow(code) {
  current = code;
  const route = workflowCopy[code];
  byId("decision-title").textContent = route.title;
  byId("criterion").textContent = route.criterion;
  const original = byId("original-decision");
  original.hidden = code === modelRoute;
  original.textContent = original.hidden ? "" : `Original recommendation: ${workflowCopy[modelRoute].title}`;
  override.value = code;
  byId("steps").replaceChildren(...route.steps.map(([stepTitle, stepDetail]) => {
    const label = document.createElement("label");
    label.className = "step";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.addEventListener("change", updateProgress);
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = stepTitle;
    const detail = document.createElement("small");
    detail.textContent = stepDetail;
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

function fillSample(content) {
  feedback.value = content;
  updateCount();
  showError("");
  feedback.focus({preventScroll: true});
}

document.querySelectorAll("[data-sample]").forEach((link) => {
  link.addEventListener("click", () => fillSample(samples[link.dataset.sample].feedback));
});

for (const sample of Object.values(samples)) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = sample.title;
  button.addEventListener("click", () => fillSample(sample.feedback));
  byId("samples").append(button);
}

for (const [code, route] of Object.entries(workflowCopy)) {
  override.add(new Option(route.title, code));
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
  status.textContent = "Connecting";
  try {
    const response = await apiFetch("/api/config");
    if (!response.ok) throw new Error();
    const config = await response.json();
    feedback.maxLength = config.max_feedback;
    updateCount();
    status.classList.add("ready");
    status.textContent = `Local model · ${config.device === "cuda" ? "GPU" : "CPU"}`;
    const evaluation = config.evaluation;
    byId("local-evaluation").hidden = !evaluation;
    if (evaluation) {
      const p = document.createElement("p");
      p.textContent = `On ${evaluation.eval_examples} synthetic cases, the base model got ${evaluation.baseline.correct} right and the fine-tuned model got ${evaluation.lora.correct} right.`;
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
  status.textContent = "Not connected";
  connectPanel.hidden = false;
  connectError.textContent = hosted
    ? "Check that WorkPattern is running and this site can access your local network. You can also open the local page."
    : "Start the WorkPattern server on this computer, then try again.";
  connectError.hidden = false;
  run.disabled = true;
  latency.textContent = "";
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
    showError("Enter feedback to analyze.");
    return;
  }
  showError("");
  run.disabled = true;
  run.firstChild.textContent = "Analyzing ";
  latency.textContent = "Analyzing feedback";
  try {
    const response = await apiFetch("/api/triage", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({feedback: input}),
    });
    const payload = await response.json();
    if (!response.ok || !Object.hasOwn(workflowCopy, payload.workflow?.code)) throw new Error("Invalid result");
    submittedFeedback = input;
    modelRoute = payload.workflow.code;
    renderWorkflow(modelRoute);
    latency.textContent = `Completed in ${payload.elapsed_ms} ms`;
  } catch (exception) {
    latency.textContent = "";
    if (exception instanceof TypeError) disconnect();
    else showError("The model could not analyze this feedback. Try a shorter report and submit it again.");
  } finally {
    run.firstChild.textContent = "Analyze feedback ";
    run.disabled = !connectPanel.hidden;
  }
});

override.addEventListener("change", () => {
  renderWorkflow(override.value);
});

byId("download").addEventListener("click", () => {
  const checked = [...byId("steps").querySelectorAll("input")];
  const steps = workflowCopy[current].steps.map(([title, detail], index) => `- [${checked[index].checked ? "x" : " "}] **${title}**\n  ${detail}`);
  const escaped = submittedFeedback.split("\n").map((line) => `> ${line}`).join("\n");
  const content = [
    "# Feedback record", "## Feedback", escaped, "## Triage decision",
    `Model recommendation: ${workflowCopy[modelRoute].title}\n\nReviewed workflow: ${workflowCopy[current].title}`,
    "## Next steps", steps.join("\n"),
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
