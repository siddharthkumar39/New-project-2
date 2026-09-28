const screenButton = document.querySelector("#screenButton");
const voiceButton = document.querySelector("#voiceButton");
const cameraButton = document.querySelector("#cameraButton");
const response = document.querySelector("#response");
const status = document.querySelector("#status");
const details = document.querySelector("#details");
const contextElem = document.querySelector("#context");
const screenTextElem = document.querySelector("#screenText");
const noteElem = document.querySelector("#note");

function setBusy(isBusy, activeBtn, busyText) {
  if (screenButton) screenButton.disabled = isBusy;
  if (voiceButton) voiceButton.disabled = isBusy;
  if (cameraButton) cameraButton.disabled = isBusy;
  if (activeBtn && isBusy) {
    activeBtn.innerHTML = busyText;
  }
}

if (screenButton) {
  screenButton.addEventListener("click", async () => {
    setBusy(true, screenButton, "Capturing…");
    status.textContent = "Working locally";
    response.textContent = "Capturing one screen image and checking it for readable text…";
    details.hidden = true;

    try {
      const request = await fetch("/api/screen", { method: "POST" });
      const data = await request.json();
      if (!request.ok) throw new Error(data.detail || "Screen analysis failed.");

      response.textContent = data.response;
      status.textContent = data.status === "success" ? "Complete" : "Needs setup";
      contextElem.textContent = data.visual_context;
      screenTextElem.textContent = data.screen_text || "No readable text returned.";
      noteElem.textContent = data.note || "";
      details.hidden = false;
    } catch (error) {
      status.textContent = "Error";
      response.textContent = error.message || "Screen analysis failed.";
    } finally {
      setBusy(false);
      screenButton.innerHTML = "Analyze screen <span>→</span>";
      if (voiceButton) voiceButton.innerHTML = "Record Voice <span>→</span>";
      if (cameraButton) cameraButton.innerHTML = "Capture Camera <span>→</span>";
    }
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

