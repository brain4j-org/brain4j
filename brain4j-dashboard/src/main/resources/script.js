const POLL_MS = 200;
const HTTP_TIMEOUT_MS = 3000;

const lossBtn = document.getElementById("lossBtn");
const accuracyBtn = document.getElementById("accuracyBtn");
const f1Btn = document.getElementById("f1Btn");
const statusBadge = document.querySelector(".status-badge");
const controlButtons = Array.from(document.querySelectorAll(".control-panel button"));

const startBtn = controlButtons.find((btn) => btn.textContent.trim() === "START");
const pauseBtn = controlButtons.find((btn) => btn.textContent.trim() === "PAUSE");
const stopBtn = controlButtons.find((btn) => btn.textContent.trim() === "STOP");
const saveBtn = document.getElementById("saveBtn");
const lossChartCanvas = document.getElementById("lossChart");
let lossChart = null;
let chartMode = "loss";
let refreshInFlight = false;
let latestTables = {
    loss_table: {},
    accuracy_table: {},
    f1_table: {}
};

const trainingOverviewRoot = Array.from(document.querySelectorAll(".section-title")).find(
    (n) => n.textContent.trim() === "Training Overview"
)?.closest(".card-body");

const metricCards = {};
if (trainingOverviewRoot) {
    const metricCells = Array.from(trainingOverviewRoot.querySelectorAll(".col-6"));
    metricCells.forEach((cell) => {
        const label = cell.querySelector(".metric-label");
        const value = cell.querySelector(".metric-value");
        if (!label || !value) return;
        metricCards[label.textContent.trim()] = value;
    });
}

const batchProgressPercent = trainingOverviewRoot
    ? Array.from(trainingOverviewRoot.querySelectorAll(".metric-label.fw-bold")).find((n) => n.textContent.trim().endsWith("%"))
    : null;
const batchProgressBar = trainingOverviewRoot?.querySelector(".progress-bar.progress-bar-striped") || null;

const systemUsageRoot = Array.from(document.querySelectorAll(".section-title")).find(
    (n) => n.textContent.trim() === "System Usage"
)?.closest(".card-body");

const systemRows = {};
if (systemUsageRoot) {
    const labels = Array.from(systemUsageRoot.querySelectorAll(".metric-label"));
    labels.forEach((label) => {
        const row = label.closest(".mb-4") || label.parentElement?.parentElement;
        if (!row) return;
        systemRows[label.textContent.trim()] = {
            value: row.querySelector(".small.fw-bold"),
            bar: row.querySelector(".progress-bar")
        };
    });
}

function isFiniteNumber(value) {
    return Number.isFinite(Number(value));
}

function formatNumber(value, digits = 4, fallback = "---") {
    return isFiniteNumber(value) ? Number(value).toFixed(digits) : fallback;
}

function formatPercent(value, digits = 2) {
    return isFiniteNumber(value) ? `${Number(value).toFixed(digits)}%` : "0%";
}

function formatDuration(ms) {
    if (!isFiniteNumber(ms) || ms < 0) return "N/A";
    const totalSec = Math.round(ms / 1000);
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function setProgress(bar, percent) {
    if (!bar) return;
    const safe = Math.max(0, Math.min(100, Number(percent) || 0));
    bar.style.width = `${safe}%`;
}

async function api(path, method = "GET") {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HTTP_TIMEOUT_MS);

    let response;
    try {
        response = await fetch(path, { method, signal: controller.signal });
    } finally {
        clearTimeout(timeoutId);
    }

    if (!response.ok) throw new Error(`${path} -> ${response.status}`);
    return response.json();
}

function initLossChart() {
    if (!lossChartCanvas || typeof Chart === "undefined") return;
    const cfg = getChartMetricConfig(chartMode);

    lossChart = new Chart(lossChartCanvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: cfg.label,
                data: [],
                borderColor: cfg.borderColor,
                backgroundColor: cfg.backgroundColor,
                borderWidth: 2,
                pointRadius: 2,
                pointHoverRadius: 4,
                tension: 0.25,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: { display: true, text: "Epoch" }
                },
                y: {
                    title: { display: true, text: cfg.yTitle },
                    beginAtZero: false
                }
            }
        }
    });
}

function getChartMetricConfig(mode) {
    if (mode === "accuracy") {
        return {
            key: "accuracy_table",
            label: "Accuracy",
            yTitle: "Accuracy",
            borderColor: "#0d6efd",
            backgroundColor: "rgba(13, 110, 253, 0.15)"
        };
    }
    if (mode === "f1") {
        return {
            key: "f1_table",
            label: "F1",
            yTitle: "F1",
            borderColor: "#20c997",
            backgroundColor: "rgba(32, 201, 151, 0.15)"
        };
    }
    return {
        key: "loss_table",
        label: "Loss",
        yTitle: "Loss",
        borderColor: "#dc3545",
        backgroundColor: "rgba(220, 53, 69, 0.15)"
    };
}

function updateMetricButtons() {
    if (lossBtn) lossBtn.classList.toggle("active", chartMode === "loss");
    if (accuracyBtn) accuracyBtn.classList.toggle("active", chartMode === "accuracy");
    if (f1Btn) f1Btn.classList.toggle("active", chartMode === "f1");
}

function updateLossChart() {
    if (!lossChart) return;
    const cfg = getChartMetricConfig(chartMode);
    const table = latestTables[cfg.key] || {};

    const entries = Object.entries(table)
        .map(([epoch, loss]) => [Number(epoch), Number(loss)])
        .filter(([epoch, loss]) => Number.isFinite(epoch) && Number.isFinite(loss))
        .sort((a, b) => a[0] - b[0]);

    const labels = entries.map(([epoch]) => String(epoch + 1));
    const values = entries.map(([, loss]) => loss);

    lossChart.data.labels = labels;
    lossChart.data.datasets[0].label = cfg.label;
    lossChart.data.datasets[0].borderColor = cfg.borderColor;
    lossChart.data.datasets[0].backgroundColor = cfg.backgroundColor;
    lossChart.data.datasets[0].data = values;
    lossChart.options.scales.y.title.text = cfg.yTitle;
    lossChart.update();
}

function updateControls(training) {
    if (!startBtn || !pauseBtn || !stopBtn) return;
    startBtn.disabled = training;
    pauseBtn.disabled = !training;
    stopBtn.disabled = !training;
}

function updateStatusBadge(training) {
    if (!statusBadge) return;
    statusBadge.classList.toggle("bg-success", training);
    statusBadge.classList.toggle("bg-secondary", !training);
    statusBadge.textContent = training ? "● TRAINING" : "● IDLE";
}

function updateTrainingInfo(info) {
    const epoch = Number(info?.epoch ?? info?.currentEpoch ?? -1);
    const totalEpochs = Number(info?.total_epochs ?? info?.totalEpochs ?? -1);
    const batch = Number(info?.batch ?? info?.currentBatch ?? -1);
    const totalBatches = Number(info?.total_batches ?? info?.totalBatches ?? -1);
    const training = Boolean(info?.training);
    const paused = Boolean(info?.paused);

    const loss = info?.loss;
    const accuracy = info?.accuracy;
    const avgTimePerBatchMs = info?.average_time_per_batch ?? info?.avgTimePerBatch;
    latestTables = {
        loss_table: info?.loss_table ?? {},
        accuracy_table: info?.accuracy_table ?? {},
        f1_table: info?.f1_table ?? {}
    };

    if (metricCards["Epoch"]) {
        metricCards["Epoch"].innerHTML = `${epoch >= 0 ? epoch + 1 : "--"}<small class="text-muted fs-6">/${totalEpochs >= 0 ? totalEpochs : "--"}</small>`;
    }

    if (metricCards["Batch"]) {
        metricCards["Batch"].innerHTML = `${batch >= 0 ? batch + 1 : "--"}<small class="text-muted fs-6">/${totalBatches >= 0 ? totalBatches : "--"}</small>`;
    }

    if (metricCards["Current Loss"]) {
        metricCards["Current Loss"].textContent = formatNumber(loss, 4);
    }

    if (metricCards["Accuracy"]) {
        const accuracyPct = isFiniteNumber(accuracy) ? Number(accuracy) * 100 : null;
        metricCards["Accuracy"].textContent = isFiniteNumber(accuracyPct) ? formatPercent(accuracyPct, 2) : "---";
    }

    if (metricCards["Avg. Time/Batch"]) {
        const avgText = isFiniteNumber(avgTimePerBatchMs) ? `${Number(avgTimePerBatchMs).toFixed(2)} ms` : "N/A";
        metricCards["Avg. Time/Batch"].innerHTML = `${avgText}<small class="text-muted fs-6"></small>`;
    }

    if (metricCards["ETA"]) {
        let eta = "0";
        if (
            totalEpochs > 0 &&
            totalBatches > 0 &&
            epoch >= 0 &&
            batch >= 0 &&
            isFiniteNumber(avgTimePerBatchMs)
        ) {
            const remainingEpochs = Math.max(totalEpochs - (epoch + 1), 0);
            const remainingCurrentEpochBatches = Math.max(totalBatches - (batch + 1), 0);
            const remainingBatches = remainingCurrentEpochBatches + remainingEpochs * totalBatches;
            eta = formatDuration(remainingBatches * Number(avgTimePerBatchMs));
        }

        metricCards["ETA"].innerHTML = `${eta}<small class="text-muted fs-6"></small>`;
    }

    const progress = totalBatches > 0 && batch >= 0 ? ((batch + 1) / totalBatches) * 100 : 0;
    if (batchProgressPercent) batchProgressPercent.textContent = `${Math.round(progress)}%`;
    setProgress(batchProgressBar, progress);

    updateStatusBadge(training);
    updateControls(training);
    updateLossChart();
    if (pauseBtn) {
        pauseBtn.textContent = paused ? "RESUME" : "PAUSE";
    }
}

function updateSystemResources(resources) {
    const cpu = Math.max(0, Math.min(100, Number(resources?.cpu_usage || 0)));
    const gpu = Math.max(0, Math.min(100, Number(resources?.gpu_usage || 0)));
    const total = Number(resources?.total || 0);
    const free = Number(resources?.free || 0);
    const used = Math.max(0, total - free);
    const ram = total > 0 ? (used / total) * 100 : 0;

    if (systemRows["CPU Usage"]?.value) systemRows["CPU Usage"].value.textContent = formatPercent(cpu, 0);
    if (systemRows["GPU Load / VRAM"]?.value) systemRows["GPU Load / VRAM"].value.textContent = `${formatPercent(gpu, 0)} `;
    if (systemRows["System RAM"]?.value) systemRows["System RAM"].value.textContent = `${formatPercent(ram, 0)} `;

    setProgress(systemRows["CPU Usage"]?.bar, cpu);
    setProgress(systemRows["GPU Load / VRAM"]?.bar, gpu);
    setProgress(systemRows["System RAM"]?.bar, ram);
}

async function refresh() {
    if (refreshInFlight) return;
    refreshInFlight = true;
    try {
        const info = await api("/api/training/info");
        const resources = await api("/api/system/resources");
        updateTrainingInfo(info);
        updateSystemResources(resources);
    } catch (err) {
        console.error("Dashboard refresh failed:", err);
    } finally {
        refreshInFlight = false;
    }
}

async function sendTrainingAction(path) {
    try {
        await api(path, "POST");
        await refresh();
    } catch (err) {
        console.error("Training action failed:", err);
    }
}

async function saveModel() {
    if (!saveBtn) return;

    const directory = window.prompt(
        "Destination folder (local path of the Java process, ex. C:\\\\models):",
        ""
    );
    if (directory === null) return;

    const filenameInput = window.prompt("Model file name:", "model.zip");
    if (filenameInput === null) return;

    const cleanDirectory = directory.trim().replace(/[\\/]+$/, "");
    let cleanFilename = filenameInput.trim().replace(/^[/\\]+/, "");

    if (!cleanFilename) {
        alert("Invalid file name.");
        return;
    }

    if (!cleanFilename.toLowerCase().endsWith(".zip")) {
        cleanFilename += ".zip";
    }

    const fullPath = cleanDirectory ? `${cleanDirectory}/${cleanFilename}` : cleanFilename;

    try {
        saveBtn.disabled = true;
        const response = await fetch(`/api/save-model?path=${encodeURIComponent(fullPath)}`, {
            method: "POST"
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.message || `Error HTTP ${response.status}`);
        }

        alert(`Model saved in:\n${payload.path || fullPath}`);
    } catch (err) {
        console.error("Model save failed:", err);
        alert(`Save failed: ${err.message || err}`);
    } finally {
        saveBtn.disabled = false;
    }
}

if (startBtn) startBtn.addEventListener("click", () => sendTrainingAction("/api/training/start"));
if (pauseBtn) {
    pauseBtn.addEventListener("click", async () => {
        const action = pauseBtn.textContent.trim() === "RESUME"
            ? "/api/training/resume"
            : "/api/training/pause";
        await sendTrainingAction(action);
    });
}
if (stopBtn) stopBtn.addEventListener("click", () => sendTrainingAction("/api/training/stop"));
if (saveBtn) saveBtn.addEventListener("click", saveModel);

if (lossBtn) {
    lossBtn.addEventListener("click", () => {
        chartMode = "loss";
        updateMetricButtons();
        updateLossChart();
    });
}
if (accuracyBtn) {
    accuracyBtn.addEventListener("click", () => {
        chartMode = "accuracy";
        updateMetricButtons();
        updateLossChart();
    });
}
if (f1Btn) {
    f1Btn.addEventListener("click", () => {
        chartMode = "f1";
        updateMetricButtons();
        updateLossChart();
    });
}

initLossChart();
updateMetricButtons();
refresh();
setInterval(refresh, POLL_MS);
