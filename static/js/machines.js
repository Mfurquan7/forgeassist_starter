
const machineGrid = document.getElementById("machine-grid");
const machineMessage = document.getElementById("machine-message");

async function loadMachines() {
    try {
        const response = await fetch("/api/machines/", {
            credentials: "same-origin",
            headers: {
                Accept: "application/json"
            }
        });

        if (!response.ok) {
            throw new Error("Unable to load machines.");
        }

        const data = await response.json();
        const machines = Array.isArray(data) ? data : data.results || [];

        machineGrid.replaceChildren();

        if (!machines.length) {
            machineGrid.innerHTML =
                '<div class="col-12"><div class="panel p-4 text-secondary">No machines registered yet.</div></div>';
            return;
        }

        machines.forEach(machine => {
            const column = document.createElement("div");
            column.className = "col-md-6 col-xl-4";

            const card = document.createElement("div");
            card.className = "panel p-4 h-100";

            const icon = document.createElement("div");
            icon.className = "assistant-icon mb-3";
            icon.textContent = "M";

            const name = document.createElement("h3");
            name.className = "h5 fw-bold";
            name.textContent = machine.name;

            const model = document.createElement("div");
            model.className = "small fw-semibold mb-3";
            model.style.color = "var(--forge-accent-dark)";
            model.textContent = machine.model_number;

            const description = document.createElement("p");
            description.className = "text-secondary small mb-3";
            description.textContent = machine.description || "No description available.";

            const date = document.createElement("div");
            date.className = "small text-secondary";
            date.textContent = machine.created_at
                ? `Added ${new Date(machine.created_at).toLocaleDateString()}`
                : "Date unavailable";

            card.append(icon, name, model, description, date);
            column.appendChild(card);
            machineGrid.appendChild(column);
        });

    } catch (error) {
        machineGrid.innerHTML = "";
        machineMessage.textContent = error.message;
        machineMessage.className = "alert alert-danger";
    }
}

loadMachines();

const createMachineForm = document.getElementById("machine-create-form");
const createMachineMessage = document.getElementById("create-machine-message");
const createMachineButton = document.getElementById("create-machine-button");

function getCookie(name) {
    const cookie = document.cookie
        .split("; ")
        .find(item => item.startsWith(`${name}=`));

    return cookie
        ? decodeURIComponent(cookie.split("=").slice(1).join("="))
        : "";
}

if (createMachineForm) {
    createMachineForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        createMachineButton.disabled = true;
        createMachineButton.textContent = "Registering...";

        const machineData = {
            name: document.getElementById("machine-name").value.trim(),
            model_number: document.getElementById("machine-model").value.trim(),
            description: document.getElementById("machine-description").value.trim()
        };

        try {
            const response = await fetch("/api/machines/", {
                method: "POST",
                credentials: "same-origin",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-CSRFToken": getCookie("csrftoken")
                },
                body: JSON.stringify(machineData)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail || JSON.stringify(data)
                );
            }

            createMachineMessage.textContent = "Machine registered successfully!";
            createMachineMessage.className = "alert alert-success";

            createMachineForm.reset();

            await loadMachines();

        } catch (error) {
            createMachineMessage.textContent = error.message;
            createMachineMessage.className = "alert alert-danger";
        } finally {
            createMachineButton.disabled = false;
            createMachineButton.textContent = "Register machine";
        }
    });
}
