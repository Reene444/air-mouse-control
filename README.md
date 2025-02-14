# 🖐️ Enhanced Gesture Mouse Control

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Supported-orange.svg)

🚀 **Enhanced Gesture Mouse Control** is a hand gesture-based mouse control tool powered by **MediaPipe** and **OpenCV**. It supports gesture locking, pinching for clicks, dragging, and scrolling operations.

---

## 🎯 Features

✅ **Gesture-based Mouse Control** - Move the cursor using your index finger with smooth motion and edge acceleration.

✅ **Position Locking** - Slightly bending your index or middle finger (>10°) locks the mouse position.

✅ **Click & Drag** - Pinching your thumb and index finger triggers a mouse click, while holding the pinch allows dragging.

✅ **Scrolling Control** - Quickly flicking your middle finger up and down enables page scrolling.

✅ **Optimized Smoothing Algorithm** - Uses a moving average filter for stable and responsive movement.

✅ **Edge Acceleration** - Enhances cursor movement speed near screen edges for efficient navigation.

✅ **Cross-Platform Compatibility** - Works on Windows, Linux, and macOS.

---


## 🛠️ Installation

Ensure you have Python 3.7 or later installed.

```bash
pip install opencv-python mediapipe pyautogui numpy
```

---

## 🚀 Run the Program

Run the following command in your terminal:

```bash
python gesture_mouse.py
```

---

## Gesture Controls

| Gesture | Action | Description |
|---|---|---|
| ☝️ Index Finger | Move Cursor | The index finger controls mouse movement |
| ✊ Finger Bend | Lock Position | Slightly bending the index or middle finger (>10°) |
| 🤏 Pinch | Click | Bringing the thumb and index finger close together (<0.03) triggers a mouse click |
| ✋ Hold Pinch | Drag | Holding the pinch allows dragging of windows or files |
| 🖖 Flick Middle Finger | Scroll | Quickly moving the middle finger up and down scrolls the page |

---


## 🤝 Contributing

We welcome contributions! Follow these steps to contribute:

1. **Fork** this repository.
2. **Create a branch** (`git checkout -b feature-xyz`).
3. **Commit your changes** (`git commit -m 'Added XYZ feature'`).
4. **Push to the branch** (`git push origin feature-xyz`).
5. **Open a Pull Request**!

---

## 📜 License

This project is licensed under the **MIT License**, allowing free use and modifications.


