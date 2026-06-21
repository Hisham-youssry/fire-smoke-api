# 🔥 Fire & Smoke Detection System using AI (YOLOv8 + FastAPI)

## 📌 Project Overview

This project is an AI-powered Fire and Smoke Detection System designed to analyze images and detect fire or smoke in real-time using a trained YOLOv8 deep learning model.

The system is exposed as a RESTful API using **FastAPI**, making it easy to integrate with web or mobile applications.

It can:

- Detect fire and smoke in images
- Return bounding boxes for detected objects
- Save annotated images with detections
- Provide processing time and metadata
- Serve processed images via API

---

## 🚀 Features

- 🔥 Fire detection using YOLOv8
- 💨 Smoke detection using YOLOv8
- 📦 REST API built with FastAPI
- 🖼️ Returns bounding boxes for detections
- 💾 Saves annotated images automatically
- ⚡ Fast inference with processing time tracking
- 🌐 Image retrieval endpoint via URL
- 🩺 Health check endpoint for system status

---

## 🧠 AI Model

- Model Type: YOLOv8 (Ultralytics)
- Task: Object Detection
- Classes:
  - Fire
  - Smoke
  - (Optional filtered classes: light, nonfire)
- Input: Image (JPG/PNG)
- Output: Bounding boxes + class labels + confidence scores

---

## 📡 API Endpoints

### 🟢 1. Health Check
