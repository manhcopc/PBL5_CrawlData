# TrendEngine AI Core - Project Overview

## 1. Project Purpose
TrendEngine AI Core is the Artificial Intelligence microservice of a larger e-commerce fashion project (PBL5). The system is designed to automate the fashion design process by bridging data-driven market insights with Generative AI. 

Instead of relying on human intuition, the system identifies current fashion trends from customer reviews and applies those trending styles to existing base garments, producing ready-to-manufacture designs.

## 2. End-to-End Flow
The system operates in a two-stage pipeline:

*   **Phase 1: Trend Analysis (The "Left Brain")**
    *   **Input:** E-commerce data (JSON/CSV) containing product details and user reviews.
    *   **Process:** Uses a pre-trained NLP model (PhoBERT) to perform sentiment analysis on Vietnamese reviews. It scores products based on positive feedback.
    *   **Output:** Identifies the "Top Trending Products" and extracts their style keywords (e.g., "royal tailcoat", "red velvet").
*   **Phase 2: Generative Design (The "Right Brain")**
    *   **Input:** A base image of an existing garment (to preserve its shape) and the style keywords generated from Phase 1.
    *   **Process:** Uses a Computer Vision model (Stable Diffusion + ControlNet) to map the trending textures, colors, and styles onto the structure of the base garment.
    *   **Output:** High-resolution, photorealistic images of the new fashion designs.

## 3. Architecture & Tech Stack
*   **Framework:** FastAPI (Python).
*   **NLP:** `transformers`, PhoBERT (by VinAI), `pyvi` (Vietnamese Tokenizer).
*   **Computer Vision:** `diffusers`, Stable Diffusion v1.5, ControlNet (Canny Edge Detection), OpenCV.
*   **Infrastructure:** Deployed on Google Colab (Tesla T4 GPU - 15GB VRAM) and exposed via `ngrok` for the Backend to consume.
*   **Design Pattern:** Microservices architecture, Singleton pattern for AI model loading to prevent VRAM overflow.

## 4. System Directory Structure
```text
app/
├── api/          # FastAPI route handlers (endpoints)
├── core/         # Global configurations and paths
├── schemas/      # Pydantic models for request/response payloads
├── services/     # AI logic (PhobertService, GenerativeService)
├── main.py       # FastAPI application entry point
scripts/          # Offline batch processing scripts
data/             # Local storage for weights, datasets, and outputs