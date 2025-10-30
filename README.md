# 🛡️ FaceShield — Real-Time Face Detection & Blurring (OpenCV DNN)

A Python project for **real-time facial detection and anonymization** using **OpenCV’s Deep Neural Network (DNN)** module.  

---

## 🎯 Project Overview

FaceShield detects human faces from a webcam or video input and automatically **blurs** them to protect identity.  
It demonstrates how computer vision can be applied for **privacy-preserving AI** applications such as:
- Public dataset anonymization  
- CCTV footage masking  
- Social media or journalism privacy tools

---

## ⚙️ Features

✅ Real-time face detection using OpenCV DNN (ResNet-based SSD)  
✅ Automatic Gaussian blur over detected faces  
✅ Runs on any modern Python version (3.8–3.13+)  

---

## 🧠 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.8+ |
| **Libraries** | `opencv-python`, `numpy` |
| **Model** | `res10_300x300_ssd_iter_140000.caffemodel` |
| **Framework** | OpenCV DNN module |
| **IDE Tested** | VS Code |

---

## 📦 Installation

1. **Clone or download this project folder.**
2. Open a terminal or PowerShell inside it.
3. Install dependencies:

```bash
pip install opencv-python numpy
