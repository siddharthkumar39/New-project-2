# SnapSight

SnapSight is a privacy-first, on-device multimodal assistant prototype designed for Snapdragon-powered HP PCs. Built for the Snapdragon AI Lab Build & Present Challenge, SnapSight combines screen understanding, voice transcription, and local camera vision into a single unified local dashboard. All perceptual understanding—screen OCR, speech transcription, object detection, scene metrics, and camera text extraction—executes 100% locally with zero cloud API dependencies and zero disk persistence for captured frames.

---

## 1. Overview
SnapSight provides contextual, multi-sensory awareness directly on the user's PC:
- **Screen Understanding**: Captures user-consented desktop screen regions using native Windows APIs and extracts on-screen text without saving images to disk.
- **Voice Understanding**: Transcribes microphone speech locally using in-memory audio buffers and on-device Whisper models.
- **Camera Understanding**: Captures single video frames into memory via Windows DirectShow, immediately releasing hardware handles.
- **Local Object Detection**: Detects common everyday objects across 80 COCO categories using a local YOLOv8n ONNX neural network.
- **Optical Character Recognition (OCR)**: Extracts visible text from documents, labels, and screens using a local Tesseract OCR engine.
- **Visual Context Analysis**: Evaluates environmental lighting and camera focus metrics from pixel data to synthesize truthful scene descriptions.

---

## 2. Problem
Modern digital workspaces require assistants that understand what users see and say. However, traditional multimodal assistants rely heavily on cloud APIs, presenting significant privacy, bandwidth, and latency challenges:
1. **Privacy & Data Security**: Streaming personal screen contents, microphone audio, and webcam video to external cloud servers exposes confidential user data.
2. **Offline Resilience**: Cloud-dependent tools become unusable during poor or unavailable internet connectivity.
3. **Bandwidth & Latency**: Uploading high-resolution images and audio streams consumes upstream bandwidth and introduces network jitter.

SnapSight solves this by keeping all capture, inference, and analysis strictly on-device. Frames and audio buffers remain entirely in memory and are never persisted to disk or sent across the network.

---

## 3. Core Capabilities

### Screen Understanding
- Uses native `Windows.Graphics.Capture` to acquire desktop visual context.
- Extracts on-screen text via local OCR.
- Operates entirely in memory with no screenshots written to storage.

### Voice Understanding
- Captures microphone input into memory via `sounddevice`.
- Transcribes speech locally using an on-device Whisper model via `faster-whisper`.
- Processes audio in-memory without creating temporary WAV/MP3 files.

### Camera Capture
- Interfaces with camera devices using Windows DirectShow (`cv2.CAP_DSHOW`) for low-latency acquisition.
- Captures exactly one frame ($640 \times 480$ BGR) directly into a memory buffer.
- Hardware handles are deterministically released in a `finally` block to prevent camera lockouts.

### Object Detection
- Executes edge-AI inference using a local YOLOv8n ONNX model (`models/yolov8n.onnx`, 12.7 MB FP32).
- Classifies objects across 80 COCO categories with real-time confidence scores and bounding boxes.
- Zero hardcoded labels: all detections originate from neural network evaluations.

### Optical Character Recognition (OCR)
- Uses local Tesseract OCR to read text from signs, phones, documents, or packaging visible to the webcam.
- Evaluates text in memory directly from NumPy image arrays.

### Visual Context Analysis
- Evaluates ambient lighting condition (e.g., low-light, well-lit, glare) from mean grayscale luminance.
- Measures camera focus and image sharpness using variance of the Laplacian.
- Synthesizes truthful, non-hallucinatory textual summaries of the scene.

---

## 4. Architecture

### System Data Flow
```text
+-------------------------------------------------------------------------+
|                                User                                     |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                      SnapSight Local Web Dashboard                       |
|                       (HTML5 / CSS3 / JavaScript)                       |
+-------------------------------------------------------------------------+
                                    |
                           REST API (FastAPI)
                                    v
+-------------------------------------------------------------------------+
|                        FastAPI Backend Layer                            |
|             (GET /api/screen, POST /api/speech, POST /api/camera)       |
+-------------------------------------------------------------------------+
         |                          |                          |
         v                          v                          v
+------------------+       +------------------+       +-------------------+
|  Screen Module   |       |   Voice Module   |       |   Camera Module   |
| (Windows.Graphics|       |  (sounddevice +  |       | (DirectShow BGR   |
|     Capture)     |       |  faster-whisper) |       |  in-memory frame) |
+------------------+       +------------------+       +-------------------+
         |                          |                          |
         |                          |                          v
         |                          |                 +-------------------+
         |                          |                 |   Vision Layer    |
         |                          |                 | (backend/vision)  |
         |                          |                 +-------------------+
         |                          |                          |
         |                          |          +---------------+---------------+
         |                          |          |               |               |
         v                          v          v               v               v
+------------------+       +------------------+  +-----------+  +-----------+  +-----------+
| Local Tesseract  |       | In-Memory Audio  |  | Lighting  |  |  ONNX     |  | In-Memory |
|   Screen OCR     |       |  Transcription   |  | & Focus   |  |  YOLOv8n  |  |  Camera   |
|                  |       |                  |  |  Metrics  |  | Inference |  |    OCR    |
+------------------+       +------------------+  +-----------+  +-----------+  +-----------+
         \                          |                          /
          \                         |                         /
           v                        v                        v
+-------------------------------------------------------------------------+
|                 Structured JSON Response (Pydantic)                     |
|                 (status, response, visual_context, objects)             |
+-------------------------------------------------------------------------+
```

### Camera Vision Pipeline
```text
                    Camera Hardware (e.g. EasyCamera)
                                   |
                                   v
                    OpenCV DirectShow Capture
                                   |
                                   v
               In-Memory BGR Frame (480x640x3 uint8)
                                   |
           +-----------------------+-----------------------+
           |                       |                       |
           v                       v                       v
    Pixel Metrics            Preprocessing            In-Memory OCR
(Luminance + Focus)    (Letterbox 640x640 NCHW)    (Tesseract Engine)
           |                       |                       |
           |                       v                       |
           |              ONNX Runtime Session             |
           |          (Dynamic Provider: QNN/DML/CPU)      |
           |                       |                       |
           |                       v                       |
           |              Raw Output (1x84x8400)           |
           |                       |                       |
           |                       v                       |
           |            Vectorized NumPy Masking           |
           |             (Threshold + Slicing)             |
           |                       |                       |
           |                       v                       |
           |             cv2.dnn.NMSBoxes (NMS)            |
           |                       |                       |
           +-----------------------+-----------------------+
                                   |
                                   v
             Truthful Natural Language & Structured Objects
```

---

## 5. Snapdragon Optimization

### Execution-Provider Negotiation
SnapSight is designed to leverage Qualcomm Hexagon NPU and Adreno GPU accelerators on Windows on ARM64 Copilot+ PCs. The vision runtime dynamically inspects available ONNX Runtime execution providers at session initialization and prioritizes hardware acceleration:

```text
              +-------------------------------------+
              |    Inspect Available ORT Providers  |
              +-------------------------------------+
                                 |
                                 v
                     [QNNExecutionProvider?]
                     /                     \
                   Yes                      No
                   /                          \
+-------------------------+        [DmlExecutionProvider?]
| Qualcomm Hexagon NPU    |        /                     \
| Hardware Acceleration   |      Yes                      No
+-------------------------+      /                          \
                          +------------------------+    +-----------------------+
                          | DirectML Acceleration  |    |  CPUExecutionProvider |
                          | (Adreno GPU / NPU)     |    | (Universal Fallback)  |
                          +------------------------+    +-----------------------+
```

1. **`QNNExecutionProvider` (Priority 1)**: Targets the Qualcomm Hexagon Tensor Processor (HTP) on Snapdragon X Elite and Snapdragon X Plus platforms via the Qualcomm Neural Network (QNN) SDK.
2. **`DmlExecutionProvider` (Priority 2)**: Targets DirectML for hardware acceleration across Windows-supported GPUs and NPUs.
3. **`CPUExecutionProvider` (Priority 3 / Fallback)**: Universal baseline ensuring the application remains fully functional if accelerator runtimes or drivers are absent.
4. **Fault-Tolerant Initialization**: If an accelerator provider fails to initialize (e.g., missing NPU backend DLLs or incompatible runtime packages), SnapSight catches the error, logs a warning, and safely falls back to CPU execution without crashing.

### Truthful Hardware Disclosure
> **Hardware Status**: SnapSight is architected and optimized for Snapdragon-powered HP PCs through dynamic ONNX Runtime execution-provider negotiation, prioritizing QNN, then DirectML, with CPU fallback. Development and functional validation were performed on a non-Snapdragon x86_64 Windows machine, so actual QNN/NPU hardware execution was not independently validated during development.

---

## 6. Privacy-First Local Processing
SnapSight enforces privacy by design:
- **No Cloud Vision APIs**: Object detection runs entirely on-device via ONNX Runtime.
- **No Cloud OCR APIs**: Text extraction is handled by a local Tesseract installation.
- **In-Memory Frame Lifecycle**: Camera frames and screen captures exist only in RAM as volatile NumPy arrays and are released immediately after inference.
- **No Disk Persistence**: Zero image or video files are written to disk during capture, preprocessing, inference, or reporting.
- **Local Audio**: Microphone buffers are processed directly in-memory by local Whisper models.

---

## 7. Performance

The following benchmarks were measured on the local development machine (Intel Core x86_64 Windows 11 PC, CPUExecutionProvider):

| Pipeline Stage / Metric | Development-Machine Result | Notes |
| :--- | :--- | :--- |
| **YOLO Postprocessing (Before)** | ~25.0 – 35.0 ms | Iterative 8,400-row Python loop |
| **YOLO Postprocessing (After Vectorization)** | **1.02 – 2.46 ms** | NumPy array broadcasting & masking (**~10x–14x speedup**) |
| **Full Object Detection (`detect_objects`)** | **~92.8 – 102.1 ms** | Letterbox preprocessing + ONNX CPU inference + NMS |
| **Preprocessing ($640 \times 640$ NCHW)** | ~11.1 – 25.8 ms | In-memory resize, RGB conversion, normalization |
| **Warm Model Inference** | ~89.9 – 106.9 ms | YOLOv8n FP32 CPU execution |
| **Camera Hardware Capture** | ~679 – 1,380 ms | Windows DirectShow USB driver & sensor AE/AGC sync |
| **Automated Test Suite** | **73 passed, 0 failures** | Unit and integration coverage |
| **Dependency Verification (`pip check`)** | **Clean** | No broken requirements or version conflicts |

*Note: All performance figures above represent development-machine measurements on CPUExecutionProvider, NOT Snapdragon NPU hardware.*

---

## 8. Technology Stack
SnapSight uses exclusively the packages and runtimes present in the repository:
- **Runtime Environment**: Python 3.13 (Windows)
- **Web API Framework**: FastAPI & Uvicorn (Asynchronous REST API)
- **Computer Vision**: OpenCV (`opencv-python-headless` 5.0) via Windows DirectShow
- **Machine Learning Runtime**: ONNX Runtime (CPUExecutionProvider with dynamic QNN/DML negotiation)
- **Vision Model**: YOLOv8n ONNX (12.7 MB FP32 nano 80-class COCO object detector)
- **Numerical Computing**: NumPy (Vectorized tensor preparation and NMS filtering)
- **Optical Character Recognition**: Tesseract OCR & `pytesseract`
- **Speech Recognition**: `faster-whisper` & `ctranslate2` with `sounddevice`
- **Screen Capture**: Windows Runtime (`winrt-Windows.Graphics.Capture`)
- **Frontend Dashboard**: Native HTML5, CSS3, and JavaScript (Zero external CDN dependencies)
- **Testing**: `pytest`

---

## 9. Hardware & Platform Support

### Development Environment (Verified)
- **OS**: Windows 11 (Version 10.0.26200)
- **Architecture**: x86_64 (`AMD64`)
- **Processor**: Intel Core (64-bit)
- **Verified Execution Provider**: `CPUExecutionProvider`

### Target Environment (Architected For)
- **Target Platform**: Snapdragon-powered HP PCs (Qualcomm Snapdragon X Elite / Snapdragon X Plus)
- **Target OS**: Windows on ARM64 (Copilot+ PC)
- **Target Acceleration**: Qualcomm Hexagon NPU via `QNNExecutionProvider` (or Adreno GPU via `DmlExecutionProvider`)
- **Fallback**: Seamless `CPUExecutionProvider` fallback

---

## 10. Installation & Setup

### Prerequisites
1. **Windows 10 / 11**
2. **Python 3.11 – 3.13**
3. **Tesseract OCR**: Install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki). SnapSight checks system `PATH` first, falling back to `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### Setup Steps
Open PowerShell in the project directory:

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Verify dependencies
.\.venv\Scripts\python.exe -m pip check
```

### Running SnapSight
Run the included launcher script:

```powershell
.\start-snapsight.ps1
```

Or run directly via Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open your browser to:
```
http://127.0.0.1:8000
```

---

## 11. Testing & Validation

Run the full automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
**Result**: `73 passed, 3 warnings in 2.16s`

Run dependency integrity checks:

```powershell
.\.venv\Scripts\python.exe -m pip check
```
**Result**: `No broken requirements found.`

---

## 12. Demonstration & Evaluation Flow

For evaluators and judges reviewing SnapSight:
1. **Launch Dashboard**: Start the server via `.\start-snapsight.ps1` and open `http://127.0.0.1:8000`.
2. **Screen Understanding**: Click **Analyze screen**. Observe on-screen text extraction from current windows without screenshot files created on disk.
3. **Voice Understanding**: Click **Record Voice**, speak a command, and observe on-device speech transcription.
4. **Camera Capture**: Click **Capture Camera**. Observe single-frame capture via Windows DirectShow with immediate device release.
5. **Object Detection**: Present objects (e.g., person, bottle, laptop, cell phone) to the camera. Observe real-time label detection and confidence scores.
6. **Optical Character Recognition**: Hold printed text or a phone screen in view. Observe extracted text populated in the visual context panel.
7. **Verify Local Privacy**: Note that network activity remains zero during inference and no image or video files are written to disk.
8. **Verify Snapdragon Provider Architecture**: Inspect `backend/vision.py` to observe dynamic QNN $\rightarrow$ DirectML $\rightarrow$ CPU provider negotiation and resilient fallback.

---

## 13. Limitations
- **Hardware Validation Scope**: Development and functional validation were performed on an x86_64 Windows PC. Physical Qualcomm Hexagon NPU hardware was not accessible during development; therefore, `QNNExecutionProvider` execution has not been independently hardware-validated on Snapdragon silicon.
- **Model Quantization**: The included YOLOv8n model is in FP32 format. Peak NPU efficiency on Snapdragon Hexagon processors typically benefits from INT8 or FP16 quantization.
- **Camera Startup Latency**: Opening and closing Windows DirectShow capture devices introduces ~700–1,300 ms of hardware sensor exposure and USB synchronization latency per capture.

---

## 14. Future Work
- **Snapdragon NPU Validation**: Deploy and validate `QNNExecutionProvider` directly on Snapdragon X Elite hardware.
- **Model Quantization**: Quantize vision models to INT8 using Qualcomm AI Hub or ONNX Runtime quantization tools for peak NPU TOPS utilization.
- **Multimodal Context Fusion**: Implement an integrated context engine combining contemporaneous screen, voice, and camera signals into a unified timeline.
- **Continuous Camera Stream Option**: Provide an optional low-latency video worker thread for real-time video feeds alongside snapshot mode.

---

## 15. Submission Summary
SnapSight demonstrates an end-to-end, on-device multimodal perception engine tailored for Snapdragon-powered Windows PCs. By unifying screen capture, speech transcription, and local edge-AI vision understanding within a zero-cloud, privacy-preserving architecture, SnapSight highlights the potential of Copilot+ PC edge computing. Its execution-provider architecture is prepared for Qualcomm Hexagon NPU acceleration while ensuring resilient, verified fallback on standard CPU environments. All 73 tests pass cleanly and dependencies are fully verified.
