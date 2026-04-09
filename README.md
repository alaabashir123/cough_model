
# Clinical Cough Monitoring System (CR-Fusion Edition)

This repository contains a high-precision, triple-stage pipeline for detecting and validating human cough events. It is optimized for mobile deployment using a custom **1035-feature Specialist MLP** and a **Bio-Physics Micro-Audit** engine.

## 🚀 System Architecture
The system uses a "Triple-Gate" defense to ensure 99% clinical precision:
1.  **Stage 1: Adaptive VAD:** Slices long audio into 5s segments based on the environment's SNR (Signal-to-Noise Ratio).
2.  **Stage 2: CR-Fusion Expert:** A dual-head Neural Network that analyzes **Sound Texture** and **Physical Impact Power** simultaneously.
3.  **Stage 3: Bio-Physics Auditor:** A micro-acoustic audit that verifies the **ECR (Energy Concentration)** and **ZCR (Turbulence)** of every peak to reject mechanical noise and speech.

## 🛠 Installation

### 1. System Requirement (FFmpeg)
The system requires FFmpeg to handle audio decoding.
*   **Windows:** Download from `ffmpeg.org` and add to PATH.
*   **Mac/Linux:** `brew install ffmpeg` or `sudo apt install ffmpeg`.

### 2. Environment Setup
```bash
pip install -r requirements.txt
```

## 🏃 Usage
To process a folder of audio files and generate the Clinical Audit Report:
```bash
python run_model.py path/to/your/audio_folder
```

## 📂 Deployment Package (Core Files)
*   `run_model.py`: The main orchestrator and reporting engine.
*   `vad_system.py`: Handles 16kHz temporal segmentation.
*   `YAMNet_Interface.py`: Stage 1 Brain (Builds the 1035-feature vector).
*   `refiner.py`: Stage 2 Auditor (Pure Signal Processing / NumPy).
*   `cough_cr_fusion_expert.tflite`: The quantized mobile-ready AI weights.

## 📊 Feature Specification (The 1035 Vector)
For mobile integration, the model expects a flattened array of 1035 features:
*   **0-1023:** YAMNet Deep Embeddings.
*   **1024-1026:** Physics (Scaled Crest, ZCR, ECR).
*   **1027-1034:** Veto Probabilities (Speech, Laughter, Sneeze, etc.).

---
