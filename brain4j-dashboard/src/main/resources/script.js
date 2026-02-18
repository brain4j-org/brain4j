const POLL_MS = 200;

const lossBtn = document.getElementById("lossBtn");
const accuracyBtn = document.getElementById("accuracyBtn");
const statusBadge = document.querySelector(".status-badge");
const controlButtons = Array.from(document.querySelectorAll(".control-panel button"));

const startBtn = controlButtons.find((btn) => btn.textContent.trim() === "START");
const pauseBtn = controlButtons.find((btn) => btn.textContent.trim() === "PAUSE");
const stopBtn = controlButtons.find((btn) => btn.textContent.trim() === "STOP");

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
    const response = await fetch(path, { method });
    if (!response.ok) throw new Error(`${path} -> ${response.status}`);
    return response.json();
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
    try {
        const [info, resources] = await Promise.all([
            api("/api/training/info"),
            api("/api/system/resources")
        ]);
        updateTrainingInfo(info);
        updateSystemResources(resources);
    } catch (err) {
        console.error("Dashboard refresh failed:", err);
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

if (startBtn) startBtn.addEventListener("click", () => sendTrainingAction("/api/training/start"));
if (pauseBtn) {
    pauseBtn.addEventListener("click", async () => {
        const action = pauseBtn.textContent.trim() === "RESUME"
            ? "/api/training/resume"
            : "/api/training/pause";
        pauseBtn.textContent = "RESUME";
        await sendTrainingAction(action);
    });
}
if (stopBtn) stopBtn.addEventListener("click", () => sendTrainingAction("/api/training/stop"));

if (lossBtn && accuracyBtn) {
    lossBtn.addEventListener("click", () => {
        lossBtn.classList.add("active");
        accuracyBtn.classList.remove("active");
    });

    accuracyBtn.addEventListener("click", () => {
        accuracyBtn.classList.add("active");
        lossBtn.classList.remove("active");
    });
}

refresh();
setInterval(refresh, POLL_MS);
