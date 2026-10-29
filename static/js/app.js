
const machineSelect = document.getElementById("machine");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const askButtonText = document.getElementById("ask-button-text");
const askSpinner = document.getElementById("ask-spinner");
const statusMessage = document.getElementById("status-message");
const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer-text");
const answerStatus = document.getElementById("answer-status");
const sourcesList = document.getElementById("sources-list");
const historyList = document.getElementById("history-list");
const refreshHistoryButton = document.getElementById("refresh-history");

function getCookie(name) {
  const cookie = document.cookie
    .split("; ")
    .find((item) => item.startsWith(`${name}=`));

  return cookie ? decodeURIComponent(cookie.split("=").slice(1).join("=")) : "";
}

function showStatus(message, type = "info") {
  statusMessage.textContent = message;
  statusMessage.className = `alert alert-${type}`;
  statusMessage.classList.remove("d-none");
}

function clearStatus() {
  statusMessage.textContent = "";
  statusMessage.className = "alert d-none";
}

async function apiRequest(url, options = {}) {
  const headers = {
    Accept: "application/json",
    ...(options.headers || {}),
  };

  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers,
  });

  if (response.status === 401 || response.status === 403) {
    let detail = "You don't have permission to perform this action.";
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (_) {
      // Keep the default message.
    }
    throw new Error(detail);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const data = await response.json();
      detail = data.detail || JSON.stringify(data);
    } catch (_) {
      // Keep the default message.
    }
    throw new Error(detail);
  }

  return response.json();
}

async function loadMachines() {
  machineSelect.innerHTML = '<option value="">Loading machines...</option>';

  try {
    const data = await apiRequest("/api/machines/");
    const machines = Array.isArray(data) ? data : data.results || [];

    machineSelect.replaceChildren();

    if (machines.length === 0) {
      machineSelect.add(new Option("No machines registered", ""));
      machineSelect.disabled = true;
      showStatus("No machines are registered yet. Ask an Administrator to add a machine.", "warning");
      return;
    }

    machineSelect.add(new Option("Choose a machine...", ""));

    for (const machine of machines) {
      const label = `${machine.name} (${machine.model_number})`;
      machineSelect.add(new Option(label, machine.id));
    }

    machineSelect.disabled = false;
  } catch (error) {
    machineSelect.replaceChildren(
      new Option("Could not load machines", "")
    );
    machineSelect.disabled = true;
    showStatus(error.message, "danger");
  }
}

function renderSources(sources) {
  sourcesList.replaceChildren();

  if (!sources || sources.length === 0) {
    const message = document.createElement("p");
    message.className = "text-secondary small mb-0";
    message.textContent = "No supporting source passage was saved for this answer.";
    sourcesList.appendChild(message);
    return;
  }

  for (const source of sources) {
    const card = document.createElement("article");
    card.className = "source-card";

    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = source.manual || "Manual";

    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.textContent = `Revision ${source.revision || "Unknown"} · Page ${source.page ?? "Unknown"}`;

    const passage = document.createElement("div");
    passage.className = "source-passage";
    passage.textContent = source.passage || "";

    card.append(title, meta, passage);
    sourcesList.appendChild(card);
  }
}

function displayAnswer(data) {
  answerText.textContent = data.answer || "No answer was returned.";

  if (data.insufficient_evidence) {
    answerStatus.className = "badge text-bg-warning";
    answerStatus.textContent = "Insufficient evidence";
  } else {
    answerStatus.className = "badge text-bg-success";
    answerStatus.textContent = "Evidence found";
  }

  renderSources(data.sources);
  answerSection.classList.remove("d-none");
}

async function loadHistory() {
  historyList.innerHTML = '<p class="text-secondary small mb-0">Loading history...</p>';

  try {
    const data = await apiRequest("/api/questions/");
    const history = Array.isArray(data) ? data : data.results || [];

    historyList.replaceChildren();

    if (history.length === 0) {
      const empty = document.createElement("p");
      empty.className = "text-secondary small py-3 mb-0";
      empty.textContent = "No questions have been asked yet.";
      historyList.appendChild(empty);
      return;
    }

    for (const item of history) {
      const wrapper = document.createElement("article");
      wrapper.className = "history-item";

      const question = document.createElement("div");
      question.className = "history-question";
      question.textContent = item.question;

      const answer = document.createElement("div");
      answer.className = "history-answer";
      answer.textContent = item.answer || "No answer recorded.";

      const meta = document.createElement("div");
      meta.className = "history-meta";
      meta.textContent = `Question #${item.id} · ${item.created_at ? new Date(item.created_at).toLocaleString() : "Date unavailable"}`;

      wrapper.append(question, answer, meta);
      historyList.appendChild(wrapper);
    }
  } catch (error) {
    historyList.replaceChildren();
    const message = document.createElement("p");
    message.className = "text-danger small py-3 mb-0";
    message.textContent = error.message;
    historyList.appendChild(message);
  }
}

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearStatus();

  const machine = machineSelect.value;
  const question = questionInput.value.trim();

  if (!machine) {
    showStatus("Please select a machine first.", "warning");
    return;
  }

  if (!question) {
    showStatus("Please enter a maintenance question.", "warning");
    questionInput.focus();
    return;
  }

  askButton.disabled = true;
  askButtonText.textContent = "Searching manuals";
  askSpinner.classList.remove("d-none");

  try {
    const data = await apiRequest("/api/questions/ask/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({
        machine: Number(machine),
        question,
      }),
    });

    displayAnswer(data);
    showStatus("Your question has been processed.", "success");
    await loadHistory();
  } catch (error) {
    showStatus(error.message, "danger");
  } finally {
    askButton.disabled = false;
    askButtonText.textContent = "Ask question";
    askSpinner.classList.add("d-none");
  }
});

refreshHistoryButton.addEventListener("click", loadHistory);

async function initializeDashboard() {
  await Promise.all([loadMachines(), loadHistory()]);
}

initializeDashboard();
