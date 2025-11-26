<h1>🧠 Dual-Stream Landmark-Based Face Presentation Attack Detection (PAD)</h1>

**Sun-hong Min, Moonseung Choi, and Yonggang Kim**  
**The 4th International Conference on Mobile • Military • Maritime IT Convergence (ICMIC 2025)**  
Kongju National University, Republic of Korea

### Acknowledgement
This research was supported by the MSIT (Ministry of Science and ICT), Korea,  
under the National Program for Excellence in Software, supervised by the IITP  
(Institute of Information & Communications Technology Planning & Evaluation) in 2025  
(2024-0-00073).

---

This repository presents a **Dual-Stream neural architecture** designed for Face Presentation Attack Detection (PAD).  
The system distinguishes real faces from spoofing attacks (printed photos, digital screen replays, etc.) by analyzing **frame-to-frame photometric changes** at **468 facial landmark points**.

MediaPipe FaceMesh is used to extract landmark coordinates, followed by computation of **color and brightness variation** between consecutive frames.  
These signals are stabilized using a **Kalman filter** and modeled through a **Dual-Stream BiLSTM network**.

---

<h3> ✨ Key Contributions</h3>

- **Photometric variation–based PAD** using only RGB-derived temporal changes, without relying on 3D geometry or texture learning
- **Dual-Stream architecture** combining frame-level statistics and sequential temporal dynamics
- **Kalman smoothing** to mitigate landmark detection noise and improve time-series consistency
- **Experimentally proven** superiority over single-branch baselines



<h3> 🗂 Dataset</h3>

This research uses the **MSU Mobile Face Spoofing Database (MSU-MFSD)**,  
which contains replay and printed attack videos recorded with multiple devices.

- **Total samples used:** 128 videos  
  - **Real (Bona fide)**: 64  
  - **Attack (Print / Display)**: 64  
- **Split**
  - **Train set**: 80 videos (51 real + 51 attack)
  - **Test set**: 26 videos (13 real + 13 attack)
  - **Validation**: 20% of training dataset
- Average duration ~12 seconds at 30 FPS (~360 frames per video)
- Converted to CSV sequences through landmark-based photometric extraction

---

<h3>📈 Experimental ResultsM</h3>

| Method | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------|-----------|------------|------------|------------|-----------|
| Branch1 | 0.8846 | 0.8326 | 0.9692 | 0.8950 | 0.9361 |
| Branch2 | 0.8385 | 0.8076 | 0.8923 | 0.8464 | 0.9325 |
| **Dual-Stream** | **0.9154** | **0.8648** | **0.9846** | **0.9207** | **0.9574** |

### Normalized Confusion Matrix
| Ground Truth | Face | Attack |
|--------------|-------|---------|
| Face | **0.92** | 0.08 |
| Attack | 0.14 | **0.86** |

---

## 🎯 Motivation

Conventional PAD approaches relying on texture features or single-frame analysis suffer severe performance degradation under variations in **illumination**, **camera quality**, and **screen artifacts**.  

This work demonstrates that **temporal photometric variation alone** can effectively separate real from spoofed faces, enabling a **lightweight and robust** PAD framework suitable for real-time applications.

---

## 🚀 Future Work

- Replay & 3D mask attack extension
- Transformer or Graph Neural Network integration
- Cross-dataset evaluation
- Real-time optimization

---

## 📚 Academic Information

> **Submitted to ICMIC 2025**  
> *Dual-Stream Landmark-Based Sequence Network for Face Presentation Attack Detection*

📩 **If you would like to read the full research paper, please email me and I will share the PDF directly.**  
📧 devminsh02@smail.kongju.ac.kr

---

## 🏷 Keywords
