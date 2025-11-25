# Edge-AI-for-Real-Time-Sign-Language-Translation

🌐 Edge AI for Real-Time ASL Sign Language Translation

A lightweight, high-speed American Sign Language recognition system built using MediaPipe and TensorFlow Lite, designed for real-time translation on laptops and edge devices like Raspberry Pi.

This project uses hand-landmark detection + an INT8-quantized deep learning model to recognize ASL gestures directly from a webcam feed.
It runs smoothly without GPU, supports offline execution, and includes optional Text-to-Speech output to speak predictions.

🎯 How It Works (Short Explanation)

MediaPipe detects 21 hand landmarks.
Landmarks are normalized and fed into the TFLite classifier.
The model predicts the ASL gesture.
The label is mapped using class_indices.json.
Text or TTS output is displayed.


🔥 Key Highlights

Real-time ASL detection using MediaPipe Hands
Optimized TFLite INT8 model (fast + edge-friendly)
Works on Raspberry Pi 4 and standard laptops
Webcam-based prediction with live overlays
Optional offline text-to-speech
Ideal for accessibility, education, and assistive AI applications


✨ Features

🔍 Real-time hand tracking using MediaPipe
🤖 Int8-quantized TFLite model → low-latency prediction
🎥 Webcam-based ASL recognition
🔊 Optional Text-to-Speech output
⚡ Designed for Raspberry Pi, laptops & edge devices
🪶 Lightweight & fast (no heavy GPU required)


🚀 Future Improvements

Add LSTM for sequence-based detection
Build Android/iOS app
Add full sentence formation
Add multilingual TTS
Improve dataset and accuracy
