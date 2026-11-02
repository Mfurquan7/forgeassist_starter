
const manualList = document.getElementById("manual-list");
const manualMessage = document.getElementById("manual-message");
const uploadForm = document.getElementById("manual-upload-form");
const uploadMessage = document.getElementById("upload-message");
const uploadButton = document.getElementById("upload-button");
const machineSelect = document.getElementById("manual-machine");
const refreshButton = document.getElementById("refresh-manuals");

function getCookie(name) {
    const cookie = document.cookie
        .split("; ")
        .find(item => item.startsWith(`${name}=`));

    return cookie
        ? decodeURIComponent(cookie.split("=").slice(1).join("="))
        : "";
}

async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
        credentials: "same-origin",
        headers: {
            Accept: "application/json",
            ...(options.headers || {})
        },
        ...options
    });

    if (!response.ok) {
        let message = `Request failed (${response.status}).`;

        try {
            const data = await response.json();
            message = data.detail || JSON.stringify(data);
        } catch (_) { }

        throw new Error(message);
    }

    return response.json();
}

function showMessage(element, message, type = "info") {
    element.textContent = message;
    element.className = `alert alert-${type}`;
}

async function loadMachines() {
    if (!machineSelect) return;

    try {
        const data = await apiRequest("/api/machines/");
        const machines = Array.isArray(data) ? data : data.results || [];

        machineSelect.replaceChildren();

        machineSelect.add(new Option("Choose a machine...", ""));

        machines.forEach(machine => {
            machineSelect.add(
                new Option(
                    `${machine.name} (${machine.model_number})`,
                    machine.id
                )
            );
        });

        if (!machines.length) {
            machineSelect.disabled = true;
            machineSelect.replaceChildren(
                new Option("No machines registered", "")
            );
        }
    } catch (error) {
        machineSelect.replaceChildren(
            new Option("Could not load machines", "")
        );
        machineSelect.disabled = true;
        showMessage(uploadMessage, error.message, "danger");
    }
}

function createCell(value) {
    const cell = document.createElement("td");
    cell.textContent = value ?? "—";
    return cell;
}

async function loadManuals() {
    manualList.replaceChildren();

    const loading = document.createElement("tr");
    const loadingCell = createCell("Loading manuals...");
    loadingCell.colSpan = 5;
    loading.appendChild(loadingCell);
    manualList.appendChild(loading);

    try {
        const data = await apiRequest("/api/manuals/");
        const manuals = Array.isArray(data) ? data : data.results || [];

        manualList.replaceChildren();

        if (!manuals.length) {
            const row = document.createElement("tr");
            const cell = createCell("No manuals uploaded yet.");
            cell.colSpan = 5;
            row.appendChild(cell);
            manualList.appendChild(row);
            return;
        }

        manuals.forEach(manual => {
            const row = document.createElement("tr");

            row.appendChild(createCell(manual.title));
            row.appendChild(createCell(manual.machine_name));
            row.appendChild(createCell(manual.revision));
            row.appendChild(createCell(manual.processing_status));

            const uploaded = manual.uploaded_at
                ? new Date(manual.uploaded_at).toLocaleDateString()
                : "—";

            row.appendChild(createCell(uploaded));

            manualList.appendChild(row);
        });

    } catch (error) {
        manualList.replaceChildren();

        const row = document.createElement("tr");
        const cell = createCell(error.message);
        cell.colSpan = 5;
        cell.className = "text-danger";
        row.appendChild(cell);
        manualList.appendChild(row);
    }
}

if (uploadForm) {
    uploadForm.addEventListener("submit", async event => {
        event.preventDefault();

        const pdf = document.getElementById("manual-pdf").files[0];

        if (!pdf || !pdf.name.toLowerCase().endsWith(".pdf")) {
            showMessage(uploadMessage, "Please select a PDF file.", "warning");
            return;
        }

        const formData = new FormData(uploadForm);

        uploadButton.disabled = true;
        uploadButton.textContent = "Uploading and processing...";

        try {
            await apiRequest("/api/manuals/", {
                method: "POST",
                headers: {
                    "X-CSRFToken": getCookie("csrftoken")
                },
                body: formData
            });

            showMessage(
                uploadMessage,
                "Manual uploaded and processed successfully.",
                "success"
            );

            uploadForm.reset();
            await loadManuals();

        } catch (error) {
            showMessage(uploadMessage, error.message, "danger");
        } finally {
            uploadButton.disabled = false;
            uploadButton.textContent = "Upload and process PDF";
        }
    });
}

if (refreshButton) {
    refreshButton.addEventListener("click", loadManuals);
}

async function initializeManuals() {
    await Promise.all([
        loadMachines(),
        loadManuals()
    ]);
}

initializeManuals();
