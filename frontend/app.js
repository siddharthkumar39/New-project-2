const screenButton = document.querySelector("#screenButton");
const cancelScreenButton = document.querySelector("#cancelScreenButton");
const screenDelay = document.querySelector("#screenDelay");
const voiceButton = document.querySelector("#voiceButton");
const cameraButton = document.querySelector("#cameraButton");
const response = document.querySelector("#response");
const status = document.querySelector("#status");
const details = document.querySelector("#details");
const contextElem = document.querySelector("#context");
const screenTextElem = document.querySelector("#screenText");
const noteElem = document.querySelector("#note");

let isBusy = false;
let screenCountdownTimer = null;
let screenCountdownInterval = null;

function setBusy(busy, activeBtn, busyText) {
  isBusy = busy;
  if (screenButton) screenButton.disabled = busy;
  if (voiceButton) voiceButton.disabled = busy;
  if (cameraButton) cameraButton.disabled = busy;
  if (screenDelay) screenDelay.disabled = busy;
  if (activeBtn && busy) {
    activeBtn.innerHTML = busyText;
  }
}

function resetScreenUI() {
  if (screenCountdownTimer) {
    clearTimeout(screenCountdownTimer);
    screenCountdownTimer = null;
  }
  if (screenCountdownInterval) {
    clearInterval(screenCountdownInterval);
    screenCountdownInterval = null;
  }
  setBusy(false);
  if (screenButton) {
    screenButton.disabled = false;
    screenButton.innerHTML = "Analyze screen <span>→</span>";
  }
  if (cancelScreenButton) cancelScreenButton.hidden = true;
  if (voiceButton) {
    voiceButton.disabled = false;
    voiceButton.innerHTML = "Record Voice <span>→</span>";
  }
  if (cameraButton) {
    cameraButton.disabled = false;
    cameraButton.innerHTML = "Capture Camera <span>→</span>";
  }
  if (screenDelay) screenDelay.disabled = false;
}

async function executeScreenCapture() {
  // State: CAPTURING -> ANALYZING
  status.textContent = "Capturing";
  response.textContent = "Capturing desktop screen in-memory...";
  if (screenButton) screenButton.innerHTML = "Capturing…";
  if (cancelScreenButton) cancelScreenButton.hidden = true;
  setBusy(true, screenButton, "Analyzing…");

  try {
    status.textContent = "Analyzing";
    response.textContent = "Reading captured screen with local OCR...";

    const request = await fetch("/api/screen", { method: "POST" });
    const data = await request.json();
    if (!request.ok) throw new Error(data.detail || "Screen analysis failed.");

    // State: RESULT
    const hasText = Boolean(data.screen_text && data.screen_text.trim());
    if (data.status === "success" && hasText) {
      status.textContent = "Complete";
      response.textContent = data.response;
    } else {
      status.textContent = "No text detected";
      response.textContent = "No readable text was detected on the captured screen.";
    }

    contextElem.textContent = data.visual_context || "";
    screenTextElem.textContent = hasText
      ? data.screen_text
      : "No readable text was detected on the captured screen.";
    noteElem.textContent = data.note || "";
    details.hidden = false;
  } catch (error) {
    status.textContent = "Error";
    response.textContent = error.message || "Screen analysis failed.";
  } finally {
    resetScreenUI();
  }
}

if (screenButton) {
  screenButton.addEventListener("click", () => {
    if (isBusy) return;

    const delaySec = screenDelay ? parseInt(screenDelay.value, 10) || 0 : 3;
    details.hidden = true;

    if (delaySec <= 0) {
      executeScreenCapture();
      return;
    }

    // State: PREPARING
    setBusy(true, screenButton, `Capturing in ${delaySec}s…`);
    if (cancelScreenButton) cancelScreenButton.hidden = false;
    status.textContent = "Preparing";
    response.textContent = "Ready to analyze. Switch to the window you want SnapSight to understand.";

    let remaining = delaySec;

    screenCountdownTimer = setTimeout(() => {
      // State: COUNTDOWN
      status.textContent = `Countdown (${remaining}s)`;
      response.textContent = `Switch to the window you want SnapSight to understand. Capturing in ${remaining}...`;
      if (screenButton) screenButton.innerHTML = `Capturing in ${remaining}s…`;

      screenCountdownInterval = setInterval(() => {
        remaining -= 1;
        if (remaining > 0) {
          status.textContent = `Countdown (${remaining}s)`;
          response.textContent = `Switch to the window you want SnapSight to understand. Capturing in ${remaining}...`;
          if (screenButton) screenButton.innerHTML = `Capturing in ${remaining}s…`;
        } else {
          clearInterval(screenCountdownInterval);
          screenCountdownInterval = null;
          if (cancelScreenButton) cancelScreenButton.hidden = true;
          executeScreenCapture();
        }
      }, 1000);
    }, 400);
  });
}

if (cancelScreenButton) {
  cancelScreenButton.addEventListener("click", () => {
    resetScreenUI();
    status.textContent = "Cancelled";
    response.textContent = "Screen analysis was cancelled. No screen was captured.";
  });
}

if (voiceButton) {
  voiceButton.addEventListener("click", async () => {
    setBusy(true, voiceButton, "Listening (5s)…");
    status.textContent = "Recording audio";
    response.textContent = "Listening to microphone for 5 seconds and transcribing locally…";
    details.hidden = true;

    try {
      const request = await fetch("/api/voice", { method: "POST" });
      const data = await request.json();
      if (!request.ok) throw new Error(data.detail || "Voice processing failed.");

      response.textContent = data.response;
      status.textContent = data.status === "success" ? "Complete" : (data.status === "partial" ? "Needs audio" : "Error");
      contextElem.textContent = data.audio_context || "";
      screenTextElem.textContent = data.transcript ? `"${data.transcript}"` : "No spoken words transcribed.";
      noteElem.textContent = data.note || "";
      details.hidden = false;
    } catch (error) {
      status.textContent = "Error";
      response.textContent = error.message || "Voice processing failed.";
    } finally {
      setBusy(false);
      voiceButton.innerHTML = "Record Voice <span>→</span>";
      if (screenButton) screenButton.innerHTML = "Analyze screen <span>→</span>";
      if (cameraButton) cameraButton.innerHTML = "Capture Camera <span>→</span>";
    }
  });
}

if (cameraButton) {
  cameraButton.addEventListener("click", async () => {
    setBusy(true, cameraButton, "Capturing…");
    status.textContent = "Working locally";
    response.textContent = "Capturing one frame from your camera locally in memory…";
    details.hidden = true;

    try {
      const request = await fetch("/api/camera", { method: "POST" });
      const data = await request.json();
      if (!request.ok) throw new Error(data.detail || "Camera capture failed.");

      response.textContent = data.response;
      status.textContent = data.status === "success" ? "Complete" : (data.status === "partial" ? "Needs camera" : "Error");
      contextElem.textContent = data.visual_context || "";
      if (data.frame_info) {
        const info = data.frame_info;
        let detailsText = `Frame: ${info.width}x${info.height} (${info.channels} channels, ${info.format})`;
        if (info.lighting && info.sharpness) {
          detailsText += ` | Lighting: ${info.lighting} | Focus: ${info.sharpness}`;
        }
        if (data.objects && data.objects.length > 0) {
          const objSummary = data.objects
            .map((o) => `${o.label} (${Math.round(o.confidence * 100)}%)`)
            .join(", ");
          detailsText += `\nDetected Objects: ${objSummary}`;
        } else {
          detailsText += "\nDetected Objects: None recognized";
        }
        if (info.detected_text) {
          detailsText += `\nVisible Text: "${info.detected_text}"`;
        }
        screenTextElem.textContent = detailsText;
      } else {

        screenTextElem.textContent = "No frame captured.";
      }
      noteElem.textContent = data.note || "";
      details.hidden = false;
    } catch (error) {
      status.textContent = "Error";
      response.textContent = error.message || "Camera capture failed.";
    } finally {
      setBusy(false);
      cameraButton.innerHTML = "Capture Camera <span>→</span>";
      if (screenButton) screenButton.innerHTML = "Analyze screen <span>→</span>";
      if (voiceButton) voiceButton.innerHTML = "Record Voice <span>→</span>";
    }
  });
}

