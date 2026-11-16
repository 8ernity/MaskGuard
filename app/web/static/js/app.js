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
let detectionHistory = [];

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
  initMediaAnalysis();
  enumerateCameras();
  loadSystemStatus();
  loadAnalytics();

  if (window.lucide) {
    lucide.createIcons();
  }
});

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

  startBtn.addEventListener("click", startLiveDetection);
  stopBtn.addEventListener("click", stopLiveDetection);
  capBtn.addEventListener("click", () => captureSnapshot(true));

  slider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    const floatVal = (val / 100).toFixed(2);
    sliderPill.textContent = `${floatVal} (${val}%)`;
    debounceUpdateThreshold(floatVal);
  });

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

    displayCanvas.width = videoEl.videoWidth || resW;
    displayCanvas.height = videoEl.videoHeight || resH;
    offscreenCanvas.width = displayCanvas.width;
    offscreenCanvas.height = displayCanvas.height;

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
    requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error("Webcam startup failed:", err);
    alert(`Could not start camera: ${err.message}. Please allow camera permissions in your browser.`);
  }
}

function stopLiveDetection() {
  isDetecting = false;
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

async function detectionLoop() {
  if (!isDetecting) return;

  if (!isRequestInProgress && videoEl.readyState === videoEl.HAVE_ENOUGH_DATA) {
    isRequestInProgress = true;

    // Capture frame to offscreen buffer
    offscreenCtx.drawImage(videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const frameData = offscreenCanvas.toDataURL("image/jpeg", 0.72);

    try {
      const res = await fetch("/api/detect/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frameData }),
      });
      const data = await res.json();

      // Render annotated frame to canvas
      if (data.annotated_image) {
        const img = new Image();
        img.onload = () => {
          displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
        };
        img.src = data.annotated_image;
      }

      // Update Live Stats
      document.getElementById("stat-total-faces").textContent = data.faces_count || 0;
      document.getElementById("stat-mask-count").textContent = data.mask_count || 0;
      document.getElementById("stat-nomask-count").textContent = data.no_mask_count || 0;
      document.getElementById("stat-fps").textContent = data.fps || 0;

      // Log detections to history feed
      if (data.detections && data.detections.length > 0) {
        processDetectionsForHistory(data.detections, offscreenCanvas);
      }
    } catch (err) {
      console.warn("Inference frame error:", err);
    } finally {
      isRequestInProgress = false;
    }
  }

  if (isDetecting) {
    requestAnimationFrame(detectionLoop);
  }
}

// --- History Feed Processing ---
function processDetectionsForHistory(detections, sourceCanvas) {
  const now = new Date();
  const timeStr = `Today, ${now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;

  detections.forEach((d) => {
    // Generate crop thumbnail
    let thumbUrl = "";
    try {
      const [x, y, w, h] = d.box;
      const cropC = document.createElement("canvas");
      cropC.width = 44;
      cropC.height = 44;
      const cropCtx = cropC.getContext("2d");
      // Clip to bounds
      const sx = Math.max(0, x);
      const sy = Math.max(0, y);
      const sw = Math.min(w, sourceCanvas.width - sx);
      const sh = Math.min(h, sourceCanvas.height - sy);
      cropCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, 44, 44);
      thumbUrl = cropC.toDataURL("image/jpeg", 0.8);
    } catch (e) {
      thumbUrl = "";
    }

    addHistoryRecord({
      thumbUrl: thumbUrl,
      label: d.label,
      confidence: d.confidence_pct || (d.confidence * 100).toFixed(1),
      timestamp: timeStr,
    });
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
  if (!videoEl || videoEl.readyState !== videoEl.HAVE_ENOUGH_DATA) {
    alert("Camera is not active. Please start the camera first.");
    return;
  }

  offscreenCtx.drawImage(videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
  const frameData = offscreenCanvas.toDataURL("image/jpeg", 0.9);

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
        imgDetails.innerHTML = `<strong>Result:</strong> ${data.faces_count} face(s) detected — <span class="text-emerald-600 font-bold">${data.mask_count} Masked</span>, <span class="text-rose-500 font-bold">${data.no_mask_count} Without Mask</span>.`;
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

      try {
        const res = await fetch("/api/detect/video", { method: "POST", body: formData });
        const data = await res.json();
        const pollUrl = data.poll_url;

        const interval = setInterval(async () => {
          const pollRes = await fetch(pollUrl);
          const job = await pollRes.json();
          progStatus.textContent = `Frame ${job.current_frame} / ${job.total_frames || "?"} (${job.fps} FPS)`;
          progPct.textContent = `${job.progress}%`;
          progBar.style.width = `${job.progress}%`;

          if (job.status === "completed") {
            clearInterval(interval);
            progStatus.innerHTML = `Complete! <a href="${job.download_url}" class="text-indigo-600 underline font-bold" download>Download Processed Video</a>`;
          } else if (job.status === "failed") {
            clearInterval(interval);
            progStatus.textContent = `Processing failed: ${job.error}`;
          }
        }, 1000);
      } catch (err) {
        progStatus.textContent = `Video upload error: ${err.message}`;
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
