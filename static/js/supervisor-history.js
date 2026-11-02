
const historyList = document.getElementById("supervisor-history-list");
const historyMessage = document.getElementById("supervisor-history-message");
const historyCount = document.getElementById("history-count");
const refreshButton = document.getElementById("refresh-supervisor-history");

async function fetchData(url) {
    const response = await fetch(url, {
        credentials: "same-origin",
        headers: { Accept: "application/json" }
    });

    if (!response.ok) {
        let message = `Request failed (${response.status}).`;

        try {
            const data = await response.json();
            message = data.detail || message;
        } catch (_) {}

        throw new Error(message);
    }

    return response.json();
}

function createElement(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = content;
    return element;
}

function renderSources(sources) {
    const container = createElement("div", "mt-3");
    container.appendChild(
        createElement("div", "small fw-bold mb-2", "Supporting sources")
    );

    if (!sources || !sources.length) {
        container.appendChild(
            createElement("p", "small text-secondary", "No supporting sources recorded.")
        );
        return container;
    }

    sources.forEach(source => {
        const card = createElement("div", "source-card mb-2");

        card.appendChild(
            createElement("div", "source-title", source.manual_title || "Manual")
        );

        card.appendChild(
            createElement(
                "div",
                "source-meta",
                `Document: ${source.document_id || "Unknown"} · Revision: ${source.revision || "Unknown"} · Page: ${source.page_number ?? "Unknown"}`
            )
        );

        card.appendChild(
            createElement("p", "source-passage mb-0", source.passage || "")
        );

        container.appendChild(card);
    });

    return container;
}

function renderHistory(history, machines) {
    historyList.replaceChildren();
    historyCount.textContent = `${history.length} questions`;

    if (!history.length) {
        historyList.appendChild(
            createElement("p", "text-secondary py-3", "No questions have been recorded yet.")
        );
        return;
    }

    history.forEach(item => {
        const wrapper = createElement("article", "history-item");

        const machine = machines.get(String(item.machine));
        const machineName = machine
            ? `${machine.name} (${machine.model_number})`
            : `Machine #${item.machine}`;

        wrapper.appendChild(
            createElement("div", "small fw-semibold mb-2", machineName)
        );

        wrapper.appendChild(
            createElement("div", "history-question", item.question)
        );

        wrapper.appendChild(
            createElement("div", "history-answer", item.answer || "No answer recorded.")
        );

        const status = createElement(
            "span",
            item.insufficient_evidence
                ? "badge text-bg-warning mt-2"
                : "badge text-bg-success mt-2",
            item.answer_status || (
                item.insufficient_evidence ? "Insufficient evidence" : "Answered"
            )
        );

        wrapper.appendChild(status);

        wrapper.appendChild(
            createElement(
                "div",
                "history-meta",
                item.created_at
                    ? new Date(item.created_at).toLocaleString()
                    : "Date unavailable"
            )
        );

        wrapper.appendChild(renderSources(item.sources));

        historyList.appendChild(wrapper);
    });
}

async function loadSupervisorHistory() {
    historyList.replaceChildren(
        createElement("p", "text-secondary", "Loading question history...")
    );

    try {
        const [historyData, machineData] = await Promise.all([
            fetchData("/api/supervisor/history/"),
            fetchData("/api/machines/")
        ]);

        const history = Array.isArray(historyData)
            ? historyData
            : historyData.results || [];

        const machineArray = Array.isArray(machineData)
            ? machineData
            : machineData.results || [];

        const machines = new Map(
            machineArray.map(machine => [String(machine.id), machine])
        );

        historyMessage.className = "alert d-none";
        renderHistory(history, machines);

    } catch (error) {
        historyList.replaceChildren();
        historyCount.textContent = "Unavailable";

        historyMessage.textContent = error.message;
        historyMessage.className = "alert alert-danger";
    }
}

refreshButton.addEventListener("click", loadSupervisorHistory);

loadSupervisorHistory();
