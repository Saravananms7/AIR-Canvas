# AirCanvas - Draw in the Air!

AirCanvas is a futuristic project that allows users to draw on a digital whiteboard by moving their hands in the air. No physical contact needed! It also features an AI-powered "Sketch-to-Image" mode where your rough sketches are recognized and turned into refined images.

## Features

- ✈️ Air Writing: Track hand movements to draw on a digital canvas.
- 🧘 Real-time Hand Tracking.
- 🎨 AI Draw Mode: Convert hand-drawn sketches into realistic images.
- 🔊 Gesture-Based Controls.
- 🔒 Secure & Smooth User Experience.

## Tech Stack

- **OpenCV** (Hand Tracking)
- **TensorFlow / ONNX Models** (Sketch Recognition)
- **Supabase** (Backend Storage for AI models and optional saving)


### Prerequisites

- Python 3.8+ (for AI backend if needed)
- OpenCV (`pip install opencv-python`)
- TensorFlow (`pip install tensorflow`) or ONNX Runtime (`pip install onnxruntime`)
- A webcam

### How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/aircanvas.git
   cd aircanvas
   ```

2. **Install Dependencies**
   ```bash
   flutter pub get
   ```

3. **Run the App (Flutter Windows App)**
   ```bash
   flutter run -d windows
   ```

4. **(Optional) Start the AI Backend**
   ```bash
   cd ai_backend
   python app.py
   ```

## Folder Structure

```
/aircanvas
 |-- /lib
 |-- /assets
 |-- /models
 |-- /ai_backend (optional for AI draw)
 |-- README.md
```

## Demo

![AirCanvas Demo GIF](link_to_your_demo_gif)

## Future Enhancements
- ✨ Add support for mobile (Android/iOS)
- 🌐 Web version with WebAssembly
- 🏋️ Multi-user Air Drawing
- 🔗 Integration with external AI Art APIs

## License

This project is licensed under the [MIT License](LICENSE).

## Credits

- OpenCV Community
- Flutter Devs
- TensorFlow / ONNX Contributors

---

Made with ❤️ by Saravanan

