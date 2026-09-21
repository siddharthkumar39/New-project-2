# SnapSight

SnapSight is a beginner-friendly, privacy-first multimodal assistant prototype for Snapdragon-powered HP PCs. This repository deliberately starts with one small feature that works end-to-end: **Understand Screen**.

## What works now

1. The dashboard asks Windows to capture one screen image.
2. The backend uses Windows.Graphics.Capture and keeps the captured frame in memory instead of saving it permanently.
3. Optional local Tesseract OCR extracts visible text.
4. The page shows a conservative response based only on what was actually read.

Camera, voice, multimodal context, Qualcomm AI Hub deployment, NPU execution, and performance claims are **not implemented or claimed yet**. They will be added only after their compatibility is verified on the target device.

## Why this is the first module

Screen understanding demonstrates the main product idea with the smallest number of moving parts. It lets us learn the app structure (browser → API → local module → response) before adding a microphone, camera, or a model runtime.

## Windows setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install Python packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Install **Tesseract OCR for Windows**. SnapSight dynamically detects `tesseract` from your system `PATH` first; if it is not found in `PATH`, it automatically falls back to the default Windows installation path (`C:\Program Files\Tesseract-OCR\tesseract.exe`). Tesseract is a local text-reading tool; without it, screen capture still works but no text will be read.

4. Start the app:

   ```powershell
   .\start-snapsight.ps1
   ```

5. Open `http://127.0.0.1:8000`, place readable text on screen, and select **Analyze screen**.

### Important: open PowerShell in the project folder first

The commands only work when PowerShell is pointed at this project folder. In PowerShell, copy and run:

```powershell
cd "C:\Users\User\Documents\ChatGPT\New project 2"
.\start-snapsight.ps1
```

The launcher uses the project's own Python environment automatically, so you do **not** need to activate `.venv` yourself. Keep that terminal window open while you use the dashboard. Press `Ctrl+C` there when you want to stop the app.

## Test

Run the small automated test before changing features:

```powershell
python -m pytest
```

## Project structure

```text
backend/screen.py    Screen capture and optional local OCR
backend/main.py      Small FastAPI web layer
frontend/            Dashboard shown in the browser
tests/               Automated checks for predictable behaviour
```

## Next build steps

1. Confirm this screen module works on the target laptop.
2. Research and verify a Snapdragon-compatible model/runtime before replacing or augmenting OCR.
3. Add the voice module, then camera, then deterministic context combination.
