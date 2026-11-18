/**
 * MaskGuard - Modern Claymorphic Dashboard Client Logic
 * Powers live browser webcam streaming, canvas bounding box rendering,
 * real-time inference, detection history feeds, snapshot capture, and multi-tab views.
 */

// Global State
let currentTab = "live";
let webcamStream = null;
let isDetecting = false;
let isRequestInProgress = false;
let renderAnimationId = null;
let latestDetections = [];
let smoothFps = 0;
let renderFrames = 0;
let lastFpsUpdate = performance.now();
let displayFps = 30;
let lastLoggedTime = 0;
let detectionHistory = [];
let isCameraFlipped = true; // Mirror camera by default for natural selfie view

const offscreenCanvas = document.createElement("canvas");
const offscreenCtx = offscreenCanvas.getContext("2d");

// DOM Elements
let videoEl, displayCanvas, displayCtx, cameraPlaceholder, soundwaveBars;

document.addEventListener("DOMContentLoaded", () => {
  videoEl = document.getElementById("webcam-video");
  displayCanvas = document.getElementById("display-canvas");
  displayCtx = displayCanvas.getContext("2d");
  cameraPlaceholder = document.getElementById("camera-placeholder");
  soundwaveBars = document.getElementById("soundwave-bars");

  initNavigation();
  initControls();
  initSettings();
  loadServerSettings();
  initMediaAnalysis();
  enumerateCameras();
  loadSystemStatus();
  loadAnalytics();

  if (window.lucide) {
    lucide.createIcons();
  }
});

async function loadServerSettings() {
  try {
    const res = await fetch("/api/settings");
    const s = await res.json();
    const slider = document.getElementById("confidence-slider");
    const sliderPill = document.getElementById("threshold-val-pill");
    if (slider && s.confidence_threshold !== undefined) {
      const pct = Math.round(s.confidence_threshold * 100);
      slider.value = pct;
      if (sliderPill) sliderPill.textContent = `${s.confidence_threshold.toFixed(2)} (${pct}%)`;
      updateSliderFill(slider, pct);
    }
  } catch (e) {
    console.log("Could not load initial settings:", e);
  }
}

// --- Tab Navigation ---
function initNavigation() {
  const tabs = {
    dashboard: { title: "Dashboard", subtitle: "Operational overview and model status", icon: "layout-dashboard" },
    live: { title: "Live Detection", subtitle: "Detect face masks from a live camera feed using AI", icon: "video" },
    history: { title: "Detection History", subtitle: "Chronological log of recent face mask detections", icon: "history" },
    dataset: { title: "Dataset", subtitle: "Kaggle ~12K images dataset breakdown and hygiene audit", icon: "database" },
    model: { title: "Model Architecture", subtitle: "Custom 3-block CNN parameters and evaluation metrics", icon: "box" },
    settings: { title: "Settings", subtitle: "Detector thresholds and vision configuration", icon: "settings" },
    about: { title: "About MaskGuard", subtitle: "Technical specifications and architecture details", icon: "info" },
  };

  document.querySelectorAll("#sidebar-nav .nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      if (!tabId || !tabs[tabId]) return;

      // Update Nav active styling
      document.querySelectorAll("#sidebar-nav .nav-item").forEach((nav) => {
        nav.classList.remove("clay-primary", "font-semibold", "active");
        nav.classList.add("text-slate-500");
      });
      item.classList.remove("text-slate-500");
      item.classList.add("clay-primary", "font-semibold", "active");

      // Switch panels
      document.querySelectorAll(".tab-content").forEach((p) => p.classList.remove("active"));
      const targetPane = document.getElementById(`tab-${tabId}`);
      if (targetPane) targetPane.classList.add("active");

      // Update Header
      document.getElementById("header-title").textContent = tabs[tabId].title;
      document.getElementById("header-subtitle").textContent = tabs[tabId].subtitle;
      const hIcon = document.getElementById("header-icon");
      if (hIcon) {
        hIcon.setAttribute("data-lucide", tabs[tabId].icon);
      }

      currentTab = tabId;
      if (window.lucide) {
        lucide.createIcons();
      }
    });
  });
}

function updateSliderFill(slider, val) {
  if (!slider) return;
  const min = parseFloat(slider.min) || 0;
  const max = parseFloat(slider.max) || 100;
  const rawVal = val !== undefined ? parseFloat(val) : (parseFloat(slider.value) || 0);
  const pct = Math.max(0, Math.min(100, ((rawVal - min) / (max - min)) * 100));
  // Exact geometric alignment to circular clay thumb center (thumb width 24px, radius 12px)
  const offsetCalc = `calc(12px + (100% - 24px) * ${(pct / 100).toFixed(4)})`;
  slider.style.background = `linear-gradient(to right, #8054f7 0%, #8054f7 ${offsetCalc}, #e2e8f0 ${offsetCalc}, #e2e8f0 100%)`;
}

// --- Live Camera Controls ---
function initControls() {
  const startBtn = document.getElementById("btn-start-camera");
  const stopBtn = document.getElementById("btn-stop-camera");
  const capBtn = document.getElementById("btn-capture");
  const slider = document.getElementById("confidence-slider");
  const sliderPill = document.getElementById("threshold-val-pill");
  const fullBtn = document.getElementById("btn-fullscreen");
  const clearBtn = document.getElementById("btn-clear-history");
  const clearFullBtn = document.getElementById("btn-clear-full-history");
  const flipBtn = document.getElementById("btn-flip-camera");

  startBtn.addEventListener("click", startLiveDetection);
  stopBtn.addEventListener("click", stopLiveDetection);
  capBtn.addEventListener("click", () => captureSnapshot(true));

  if (flipBtn) {
    flipBtn.addEventListener("click", () => {
      isCameraFlipped = !isCameraFlipped;
      const flipText = document.getElementById("flip-status-text");
      if (isCameraFlipped) {
        flipBtn.classList.add("text-indigo-600", "border-indigo-200");
        flipBtn.classList.remove("text-slate-600");
        if (flipText) flipText.textContent = "Mirrored";
      } else {
        flipBtn.classList.remove("text-indigo-600", "border-indigo-200");
        flipBtn.classList.add("text-slate-600");
        if (flipText) flipText.textContent = "Normal";
      }
    });
  }

  if (cameraPlaceholder) {
    cameraPlaceholder.addEventListener("click", () => {
      if (!isDetecting) startLiveDetection();
    });
  }

  if (slider) {
    updateSliderFill(slider, parseInt(slider.value, 10));
    slider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      const floatVal = (val / 100).toFixed(2);
      if (sliderPill) sliderPill.textContent = `${floatVal} (${val}%)`;
      updateSliderFill(slider, val);
      debounceUpdateThreshold(floatVal);
    });
  }

  if (clearBtn) clearBtn.addEventListener("click", clearDetectionHistory);
  if (clearFullBtn) clearFullBtn.addEventListener("click", clearDetectionHistory);

  if (fullBtn) {
    fullBtn.addEventListener("click", () => {
      const container = document.getElementById("viewport-container");
      if (!document.fullscreenElement) {
        container.requestFullscreen().catch((err) => console.log(err));
      } else {
        document.exitFullscreen().catch((err) => console.log(err));
      }
    });
  }
}

let thresholdTimer = null;
function debounceUpdateThreshold(val) {
  clearTimeout(thresholdTimer);
  thresholdTimer = setTimeout(async () => {
    try {
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confidence_threshold: parseFloat(val) }),
      });
    } catch (err) {
      console.warn("Threshold update note:", err);
    }
  }, 350);
}

// --- Enumerate Media Devices ---
async function enumerateCameras() {
  const select = document.getElementById("camera-source-select");
  if (!select || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === "videoinput");
    if (videoDevices.length > 0) {
      select.innerHTML = "";
      videoDevices.forEach((device, idx) => {
        const opt = document.createElement("option");
        opt.value = device.deviceId;
        opt.textContent = device.label || `Camera ${idx + 1}`;
        select.appendChild(opt);
      });
    }
  } catch (e) {
    console.log("Camera enumeration note:", e);
  }
}

// --- Live Webcam Streaming Loop ---
async function startLiveDetection() {
  const resSelect = document.getElementById("res-selector");
  const [resW, resH] = (resSelect ? resSelect.value : "1280x720").split("x").map(Number);
  const deviceSelect = document.getElementById("camera-source-select");
  const deviceId = deviceSelect && deviceSelect.value ? { exact: deviceSelect.value } : undefined;

  const constraints = {
    video: {
      width: { ideal: resW },
      height: { ideal: resH },
      deviceId: deviceId,
    },
    audio: false,
  };

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = webcamStream;
    await videoEl.play();

    // Display canvas matches camera video dimensions
    displayCanvas.width = videoEl.videoWidth || resW;
    displayCanvas.height = videoEl.videoHeight || resH;

    // Offscreen canvas: high resolution for crisp facial feature inference (max dimension 640)
    const maxInferDim = 640;
    const inferScale = Math.min(1.0, maxInferDim / Math.max(displayCanvas.width, displayCanvas.height));
    offscreenCanvas.width = Math.round(displayCanvas.width * inferScale);
    offscreenCanvas.height = Math.round(displayCanvas.height * inferScale);

    // Toggle Viewport
    cameraPlaceholder.classList.add("hidden");
    displayCanvas.classList.remove("hidden");

    // Toggle HUD buttons & indicators
    document.getElementById("btn-start-camera").classList.add("opacity-50", "pointer-events-none");
    document.getElementById("btn-stop-camera").classList.remove("opacity-50", "pointer-events-none");
    document.getElementById("stream-status-dot").className = "w-3 h-3 rounded-full bg-emerald-500 inline-block shadow-sm animate-pulse";
    document.getElementById("stream-status-text").textContent = "Live Camera Streaming";
    document.getElementById("overlay-ai-badge").textContent = "AI Detecting...";
    document.getElementById("overlay-status-desc").textContent = "Real-time inference running";
    if (soundwaveBars) soundwaveBars.style.opacity = "1";

    isDetecting = true;
    isRequestInProgress = false;
    latestDetections = [];
    smoothFps = 0;

    // Start 60 FPS Canvas Render Loop
    if (renderAnimationId) cancelAnimationFrame(renderAnimationId);
    renderAnimationId = requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error("Webcam startup failed:", err);
    alert(`Could not start camera: ${err.message}. Please allow camera permissions in your browser.`);
  }
}

function stopLiveDetection() {
  isDetecting = false;
  isRequestInProgress = false;
  latestDetections = [];
  activeFaceTracks = [];

  if (renderAnimationId) {
    cancelAnimationFrame(renderAnimationId);
    renderAnimationId = null;
  }

  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }

  // Restore Idle state
  cameraPlaceholder.classList.remove("hidden");
  displayCanvas.classList.add("hidden");
  displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);

  document.getElementById("btn-start-camera").classList.remove("opacity-50", "pointer-events-none");
  document.getElementById("btn-stop-camera").classList.add("opacity-50", "pointer-events-none");
  document.getElementById("stream-status-dot").className = "w-3 h-3 rounded-full bg-slate-400 inline-block shadow-sm";
  document.getElementById("stream-status-text").textContent = "Camera Idle";
  document.getElementById("overlay-ai-badge").textContent = "AI Idle";
  document.getElementById("overlay-status-desc").textContent = "Camera feed paused";
  if (soundwaveBars) soundwaveBars.style.opacity = "0.3";

  // Reset live stats
  document.getElementById("stat-total-faces").textContent = "0";
  document.getElementById("stat-mask-count").textContent = "0";
  document.getElementById("stat-nomask-count").textContent = "0";
  document.getElementById("stat-fps").textContent = "0";
}

// 60 FPS Native Camera Render Loop
function renderLoop() {
  if (!isDetecting) return;

  const now = performance.now();
  renderFrames++;
  if (now - lastFpsUpdate >= 500) {
    displayFps = Math.round((renderFrames * 1000) / (now - lastFpsUpdate));
    renderFrames = 0;
    lastFpsUpdate = now;
    const fpsEl = document.getElementById("stat-fps");
    if (fpsEl) fpsEl.textContent = displayFps;
  }

  if (videoEl && videoEl.readyState >= videoEl.HAVE_CURRENT_DATA) {
    // 1. Draw live webcam frame smoothly at 60 FPS (mirrored horizontally by default for natural selfie view)
    displayCtx.save();
    if (isCameraFlipped) {
      displayCtx.translate(displayCanvas.width, 0);
      displayCtx.scale(-1, 1);
    }
    displayCtx.drawImage(videoEl, 0, 0, displayCanvas.width, displayCanvas.height);
    displayCtx.restore();

    // 2. Draw active bounding boxes & badges
    drawDetections(displayCtx, latestDetections);
  }

  // 3. Dispatch non-blocking inference if idle
  if (!isRequestInProgress) {
    runInference();
  }

  renderAnimationId = requestAnimationFrame(renderLoop);
}

// Sleek client-side detection renderer matching Visage AI / MaskGuard Claymorphic theme
function drawDetections(ctx, detections) {
  if (!detections || detections.length === 0) return;

  ctx.save();
  for (const d of detections) {
    const [x, y, w, h] = d.box;
    const isUncertain = d.is_uncertain || d.label === "UNCERTAIN";
    const isMask = d.label === "MASK" && !isUncertain;
    const strokeColor = isUncertain ? "#f59e0b" : isMask ? "#10b981" : "#ef4444";

    // Sleek rounded bounding box
    ctx.lineWidth = 3;
    ctx.strokeStyle = strokeColor;
    ctx.shadowColor = isUncertain
      ? "rgba(245, 158, 11, 0.45)"
      : isMask
      ? "rgba(16, 185, 129, 0.45)"
      : "rgba(239, 68, 68, 0.45)";
    ctx.shadowBlur = 8;

    const radius = 10;
    if (typeof ctx.roundRect === "function") {
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, radius);
      ctx.stroke();
    } else {
      ctx.strokeRect(x, y, w, h);
    }

    // Badge Pill
    ctx.shadowBlur = 0; // Clear shadow for crisp text rendering
    const badgeText = `${d.label} ${d.confidence_pct || (d.confidence * 100).toFixed(1)}%`;
    ctx.font = "bold 13px 'Plus Jakarta Sans', system-ui, sans-serif";
    const textMetrics = ctx.measureText(badgeText);
    const badgeW = textMetrics.width + 18;
    const badgeH = 26;
    const badgeRadius = 6;

    // Position badge directly above bounding box (or inside if near top canvas boundary)
    let badgeX = Math.max(4, Math.min(x, ctx.canvas.width - badgeW - 4));
    let badgeY = y - badgeH - 3;
    if (badgeY < 4) {
      badgeY = y + 4;
    }

    // Badge pill background
    ctx.fillStyle = strokeColor;
    if (typeof ctx.roundRect === "function") {
      ctx.beginPath();
      ctx.roundRect(badgeX, badgeY, badgeW, badgeH, badgeRadius);
      ctx.fill();
    } else {
      ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
    }

    // Badge text
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeText, badgeX + 9, badgeY + badgeH / 2);
  }
  ctx.restore();
}

// Decoupled Background Inference Loop
async function runInference() {
  if (!isDetecting || isRequestInProgress) return;
  if (!videoEl || videoEl.readyState < videoEl.HAVE_CURRENT_DATA) return;

  isRequestInProgress = true;
  const tStart = performance.now();

  // Capture frame with high quality (0.85) to eliminate DCT block artifacts on mouth and lips
  offscreenCtx.save();
  if (isCameraFlipped) {
    offscreenCtx.translate(offscreenCanvas.width, 0);
    offscreenCtx.scale(-1, 1);
  }
  offscreenCtx.drawImage(videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
  offscreenCtx.restore();
  const frameData = offscreenCanvas.toDataURL("image/jpeg", 0.85);

  try {
    const res = await fetch("/api/detect/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: frameData, return_annotated: false }),
    });
    const data = await res.json();

    const tEnd = performance.now();
    const cycleTime = Math.max(1, tEnd - tStart);
    const measuredFps = 1000.0 / cycleTime;
    smoothFps = smoothFps === 0 ? measuredFps : (0.85 * smoothFps + 0.15 * measuredFps);

    // Compute coordinate mapping from offscreen canvas to display canvas
    const scaleX = displayCanvas.width / offscreenCanvas.width;
    const scaleY = displayCanvas.height / offscreenCanvas.height;

    const scaledDetections = (data.detections || []).map((d) => ({
      ...d,
      box: [
        d.box[0] * scaleX,
        d.box[1] * scaleY,
        d.box[2] * scaleX,
        d.box[3] * scaleY,
      ],
    }));

    latestDetections = scaledDetections;

    // Update Live Stats HUD
    document.getElementById("stat-total-faces").textContent = data.faces_count || 0;
    document.getElementById("stat-mask-count").textContent = data.mask_count || 0;
    document.getElementById("stat-nomask-count").textContent = data.no_mask_count || 0;

    // Log detections to history feed
    if (scaledDetections.length > 0) {
      processDetectionsForHistory(scaledDetections, videoEl);
    }
  } catch (err) {
    console.warn("Inference frame error:", err);
  } finally {
    isRequestInProgress = false;
  }
}

// --- Smart Event-Driven History Feed Processing ---
let activeFaceTracks = []; // [{ cx, cy, label, lastSeen, lastLogged }]

function processDetectionsForHistory(detections, sourceVideo) {
  const now = Date.now();
  if (!detections || detections.length === 0) {
    // Drop tracks not seen in 5s
    activeFaceTracks = activeFaceTracks.filter((t) => now - t.lastSeen < 5000);
    return;
  }

  const vW = sourceVideo.videoWidth || displayCanvas.width;
  const vH = sourceVideo.videoHeight || displayCanvas.height;

  detections.forEach((d) => {
    // Ignore uncertain classifications in history feed
    if (d.is_uncertain || d.label === "UNCERTAIN") return;

    const [x, y, w, h] = d.box;
    const cx = x + w / 2;
    const cy = y + h / 2;

    // Match with existing track within generous pixel distance (150px) to handle normal head movements
    let matchedTrack = null;
    let minDist = 150;
    for (const t of activeFaceTracks) {
      const dist = Math.hypot(t.cx - cx, t.cy - cy);
      if (dist < minDist) {
        minDist = dist;
        matchedTrack = t;
      }
    }

    const currentLabel = d.label;

    if (matchedTrack) {
      matchedTrack.cx = cx;
      matchedTrack.cy = cy;
      matchedTrack.lastSeen = now;

      const labelChanged = matchedTrack.label !== currentLabel;
      const timeElapsed = now - matchedTrack.lastLogged;
      // Re-log only if state changed (e.g. Put on mask or took it off) OR at most once every 30s
      if (labelChanged || timeElapsed > 30000) {
        matchedTrack.label = currentLabel;
        matchedTrack.lastLogged = now;
        logDetectionEntry(d, sourceVideo, vW, vH);
      }
    } else {
      // New face entered camera view
      activeFaceTracks.push({
        cx: cx,
        cy: cy,
        label: currentLabel,
        lastSeen: now,
        lastLogged: now,
      });
      logDetectionEntry(d, sourceVideo, vW, vH);
    }
  });

  // Clean up stale tracks not seen in 5s
  activeFaceTracks = activeFaceTracks.filter((t) => now - t.lastSeen < 5000);
}

function logDetectionEntry(d, sourceVideo, vW, vH) {
  const [x, y, w, h] = d.box;
  let thumbUrl = "";
  try {
    const cropC = document.createElement("canvas");
    cropC.width = 44;
    cropC.height = 44;
    const cropCtx = cropC.getContext("2d");

    // Center-chin safe square thumbnail crop
    const cx = ((x + w / 2) / displayCanvas.width) * vW;
    const cy = ((y + h / 2) / displayCanvas.height) * vH + ((h * 0.1 * vH) / displayCanvas.height);
    const side = Math.max(w, h) * 1.35 * (vW / displayCanvas.width);
    const half = side / 2;
    const sx = Math.max(0, cx - half);
    const sy = Math.max(0, cy - half);
    const sw = Math.min(vW - sx, side);
    const sh = Math.min(vH - sy, side);

    cropCtx.drawImage(sourceVideo, sx, sy, sw, sh, 0, 0, 44, 44);
    thumbUrl = cropC.toDataURL("image/jpeg", 0.8);
  } catch (e) {
    thumbUrl = "";
  }

  const timeStr = `Today, ${new Date().toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })}`;

  addHistoryRecord({
    thumbUrl: thumbUrl,
    label: d.label,
    confidence: d.confidence_pct || (d.confidence * 100).toFixed(1),
    timestamp: timeStr,
  });
}

function addHistoryRecord(item) {
  detectionHistory.unshift(item);
  if (detectionHistory.length > 50) detectionHistory.pop();

  renderHistoryFeeds();
}

function renderHistoryFeeds() {
  const feedEl = document.getElementById("history-feed-list");
  const fullFeedEl = document.getElementById("full-history-feed");

  if (!feedEl) return;
  if (detectionHistory.length === 0) {
    feedEl.innerHTML = `<div id="history-empty" class="text-center py-6 text-xs text-slate-400 font-medium">No detections recorded yet. Start camera to log detections.</div>`;
    if (fullFeedEl) {
      fullFeedEl.innerHTML = `<div class="text-center py-12 text-slate-400 text-sm">No detection history captured yet.</div>`;
    }
    return;
  }

  const buildHtml = (items, maxCount) => {
    return items
      .slice(0, maxCount)
      .map((entry) => {
        const isMask = entry.label === "MASK";
        const pillBg = isMask ? "bg-[#059669]" : "bg-[#f87171]";
        const thumbHtml = entry.thumbUrl
          ? `<img src="${entry.thumbUrl}" alt="Face" class="w-full h-full object-cover">`
          : `<i data-lucide="${isMask ? "smile" : "frown"}" class="w-5 h-5 ${isMask ? "text-emerald-600" : "text-rose-500"}"></i>`;

        return `
        <div class="flex items-center justify-between py-1.5 border-b border-slate-100 last:border-0">
          <div class="flex items-center space-x-2.5">
            <div class="w-9 h-9 rounded-full overflow-hidden bg-slate-200 border-2 border-white shadow-sm shrink-0 flex items-center justify-center">
              ${thumbHtml}
            </div>
            <span class="px-2 py-0.5 rounded-md text-[10px] font-extrabold ${pillBg} text-white tracking-wider">${entry.label}</span>
          </div>
          <div class="flex items-center space-x-3 text-right">
            <span class="text-xs font-bold text-slate-700">${entry.confidence}%</span>
            <span class="text-[11px] text-slate-400 font-medium">${entry.timestamp}</span>
          </div>
        </div>`;
      })
      .join("");
  };

  feedEl.innerHTML = buildHtml(detectionHistory, 10);
  if (fullFeedEl) {
    fullFeedEl.innerHTML = buildHtml(detectionHistory, 50);
  }

  if (window.lucide) {
    lucide.createIcons();
  }
}

function clearDetectionHistory() {
  detectionHistory = [];
  renderHistoryFeeds();
}

// --- Snapshot Capture ---
async function captureSnapshot(notify = true) {
  if (!videoEl || videoEl.readyState < videoEl.HAVE_CURRENT_DATA) {
    alert("Camera is not active. Please start the camera first.");
    return;
  }

  const frameData = displayCanvas.toDataURL("image/jpeg", 0.92);

  try {
    const res = await fetch("/api/detect/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: frameData, save_snapshot: true }),
    });
    const data = await res.json();
    if (notify) {
      alert(`Snapshot captured successfully! Detected ${data.faces_count} faces (${data.mask_count} Masked, ${data.no_mask_count} Unmasked).`);
    }
  } catch (err) {
    console.error("Snapshot error:", err);
  }
}

// --- Static Image and Video Processing ---
function initMediaAnalysis() {
  const imgInput = document.getElementById("image-upload-input");
  const imgPreview = document.getElementById("image-result-preview");
  const imgDetails = document.getElementById("image-result-details");
  const imgResultBox = document.getElementById("image-upload-result");

  if (imgInput) {
    imgInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      imgDetails.textContent = "Analyzing image with custom CNN...";
      imgResultBox.classList.remove("hidden");

      try {
        const res = await fetch("/api/detect/image", { method: "POST", body: formData });
        const data = await res.json();
        imgPreview.src = data.annotated_image;
        if (data.faces_count === 0) {
          imgDetails.innerHTML = `<span class="text-amber-600 font-bold">No face detected</span> in this image. Try an image with a visible, front-facing face or adjust sensitivity in Settings.`;
        } else {
          const maskPart = `<span class="text-emerald-600 font-bold">${data.mask_count} Masked</span>`;
          const noMaskPart = `<span class="text-rose-500 font-bold">${data.no_mask_count} Without Mask</span>`;
          const uncPart = data.uncertain_count > 0 ? `, <span class="text-amber-500 font-bold">${data.uncertain_count} Uncertain</span>` : '';
          imgDetails.innerHTML = `<strong>Result:</strong> ${data.faces_count} face(s) detected — ${maskPart}, ${noMaskPart}${uncPart}.`;
        }
      } catch (err) {
        imgDetails.textContent = `Analysis error: ${err.message}`;
      }
    });
  }

  const vidBtn = document.getElementById("btn-process-video");
  const vidInput = document.getElementById("video-upload-input");
  if (vidBtn && vidInput) {
    vidBtn.addEventListener("click", async () => {
      const file = vidInput.files[0];
      if (!file) {
        alert("Please select a video file first.");
        return;
      }
      const formData = new FormData();
      formData.append("file", file);

      const progBox = document.getElementById("video-progress-box");
      const progStatus = document.getElementById("video-status-text");
      const progPct = document.getElementById("video-percent-text");
      const progBar = document.getElementById("video-bar");
      progBox.classList.remove("hidden");
      progStatus.textContent = "Uploading video to server...";
      progPct.textContent = "0%";
      progBar.style.width = "0%";

      vidBtn.disabled = true;
      vidBtn.classList.add("opacity-60");

      try {
        const res = await fetch("/api/detect/video", { method: "POST", body: formData });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(errData.detail || "Upload failed");
        }
        const data = await res.json();
        const pollUrl = data.poll_url;
        progStatus.textContent = "Analyzing video frames...";

        const interval = setInterval(async () => {
          try {
            const pollRes = await fetch(pollUrl);
            const job = await pollRes.json();
            progStatus.textContent = `Frame ${job.current_frame} / ${job.total_frames || "?"} (${job.fps || 0} FPS)`;
            progPct.textContent = `${job.progress}%`;
            progBar.style.width = `${job.progress}%`;

            if (job.status === "completed") {
              clearInterval(interval);
              vidBtn.disabled = false;
              vidBtn.classList.remove("opacity-60");
              progStatus.innerHTML = `Complete! <a href="${job.download_url}" class="text-purple-600 underline font-bold ml-1" download>Download Processed Video</a>`;
            } else if (job.status === "failed") {
              clearInterval(interval);
              vidBtn.disabled = false;
              vidBtn.classList.remove("opacity-60");
              progStatus.textContent = `Processing failed: ${job.error || "Unknown error"}`;
            }
          } catch (pollErr) {
            console.warn("Poll error:", pollErr);
          }
        }, 800);
      } catch (err) {
        vidBtn.disabled = false;
        vidBtn.classList.remove("opacity-60");
        progStatus.textContent = `Video error: ${err.message}`;
      }
    });
  }
}

// --- Settings Persistence ---
function initSettings() {
  const saveBtn = document.getElementById("btn-save-settings");
  if (!saveBtn) return;

  saveBtn.addEventListener("click", async () => {
    const payload = {
      face_scale_factor: parseFloat(document.getElementById("setting-scale-factor").value),
      face_min_neighbors: parseInt(document.getElementById("setting-min-neighbors").value, 10),
      face_padding: parseFloat(document.getElementById("setting-padding").value),
    };

    try {
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      alert("Vision settings saved and active!");
    } catch (err) {
      alert(`Failed to save settings: ${err.message}`);
    }
  });
}

// --- Status & Analytics Loader ---
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.model_file) {
      const fEl = document.getElementById("dash-model-file");
      if (fEl) fEl.textContent = data.model_file;
    }
  } catch (e) {
    console.log("Status note:", e);
  }
}

async function loadAnalytics() {
  try {
    const res = await fetch("/api/analytics");
    const data = await res.json();
    if (data.metrics) {
      const acc = (data.metrics.test_accuracy * 100).toFixed(1);
      const prec = (data.metrics.precision * 100).toFixed(1);
      const rec = (data.metrics.recall * 100).toFixed(1);
      const f1 = (data.metrics.f1_score * 100).toFixed(1);

      const dAcc = document.getElementById("dash-accuracy");
      if (dAcc) dAcc.textContent = `${acc}%`;

      const mAcc = document.getElementById("model-metric-acc");
      if (mAcc) mAcc.textContent = `${acc}%`;
      const mPrec = document.getElementById("model-metric-prec");
      if (mPrec) mPrec.textContent = `${prec}%`;
      const mRec = document.getElementById("model-metric-rec");
      if (mRec) mRec.textContent = `${rec}%`;
      const mF1 = document.getElementById("model-metric-f1");
      if (mF1) mF1.textContent = `${f1}%`;
    }
  } catch (e) {
    console.log("Analytics note:", e);
  }
}
