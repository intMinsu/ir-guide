[ [English](#english) | [한국어](#한국어) ]
# Remote Camera Streaming Examples
**Up one level:** [`../README.md`](../README.md) · **Root:** [`../../README.md`](../../README.md)

---

## English

Short pointers to each example:

- **01_mediamtx/** — RTSP via **MediaMTX** (separate server). Use when you want a central RTSP broker serving multiple clients and flexible routing; scripts show OpenCV/GStreamer → NVENC → MediaMTX.
- **02_jetson_inference/** — **jetson-utils** with a **built‑in WebRTC/RTSP server**. Fastest way to get a zero‑copy CUDA pipeline streaming.
- **03_efficientViT/** — Lightweight model demo integrated with the streaming loop; shows how to run on‑device inference and overlay results during streaming.

---

## 한국어

각 예제의 간단 안내:

- **01_mediamtx/** — **MediaMTX** 기반 RTSP(별도 서버). 다중 클라이언트 송출/라우팅이 필요한 경우 사용; OpenCV/GStreamer → NVENC → MediaMTX 파이프라인 예시 포함.
- **02_jetson_inference/** — **jetson-utils**의 **내장 WebRTC/RTSP 서버** 사용. 제로‑카피 CUDA 파이프라인을 가장 빠르게 스트리밍합니다.
- **03_efficientViT/** — 경량 모델을 스트리밍 루프에 결합한 데모; 디바이스 내 추론과 오버레이 예시를 제공합니다.