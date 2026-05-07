# End-to-End System Workflow: AI-Driven Trend Analysis and Generative Design

## Overview
This document outlines the end-to-end (E2E) execution flow of the TrendEngine AI Core. The architecture consists of three primary layers:
1.  **Frontend (FE):** The user interface for fashion designers or shop owners.
2.  **Backend (BE):** The core server that manages business logic, database interactions, and acts as a bridge to the AI Server.
3.  **AI Server (AI):** The independent microservice hosting NLP (PhoBERT) and Generative (Stable Diffusion + ControlNet) models.

This workflow describes the exact sequence of events when a user attempts to discover market trends and automatically generate a new fashion design based on those trends.

---

## Phase 1: Trend Discovery (Market Analysis)

**Step 1.1: User Initiation (FE)**
The user navigates to the dashboard, enters a target product category (e.g., "Vest"), and clicks the "Analyze Market Trend" button. The Frontend sends a request to the Backend.

**Step 1.2: Data Aggregation (BE)**
Instead of performing real-time web scraping, the Backend retrieves pre-collected or simulated product data (including product details and user reviews) from the database. The Backend formats this data strictly according to the `TrendRequest` JSON schema.

**Step 1.3: AI Processing Request (BE to AI)**
The Backend sends a synchronous `POST` request to the AI Server's `/api/v1/analyze-trend` endpoint with the constructed payload.

**Step 1.4: NLP Analysis and Scoring (AI)**
The AI Server receives the payload and processes the reviews using the PhoBERT model. It calculates the positive sentiment rate, computes the final `trend_score`, and extracts relevant `style_keywords` from the text. The AI Server sorts the products by trend score and returns the top results to the Backend.

**Step 1.5: Visualization (BE to FE)**
The Backend receives the AI analysis, persists the history in the database, and forwards the structured data to the Frontend. The Frontend renders a trend chart, highlighting the top-trending product and its associated keywords (e.g., "leather", "oversized").

---

## Phase 2: Design Ideation and Generation Request

**Step 2.1: Design Trigger (FE)**
The user selects the Top 1 trending product from the analysis results and clicks "Generate Design from Inspiration". The Frontend notifies the Backend to initiate the generative pipeline.

**Step 2.2: Payload Construction (BE)**
The Backend retrieves the base image URL (`source_image_url`) and the extracted style keywords associated with the selected product. 

**Step 2.3: Image Pre-processing (BE) - CRITICAL**
To prevent Out-Of-Memory (OOM) exceptions on the AI Server's GPU, the Backend MUST append resizing parameters to the image URL or programmatically resize the image to a standard resolution (e.g., 512x512 pixels) before sending it to the AI Server.

**Step 2.4: Prompt Engineering (BE)**
The Backend injects the extracted keywords into a predefined system prompt template. 
*Example: "high fashion blazer vest, [KEYWORDS] texture, luxury cinematic lighting, 8k resolution, photorealistic".*

**Step 2.5: Job Submission (BE to AI)**
The Backend sends a `POST` request to the `/api/v1/generate-design` endpoint. Because image generation is computationally heavy, the AI Server immediately responds with a `202 Accepted` status and a unique `job_id`, placing the actual rendering task in a background queue.

---

## Phase 3: Asynchronous Polling and Result Delivery

**Step 3.1: Loading State (FE)**
Upon successful job submission, the Frontend displays a loading indicator (e.g., a spinner or progress bar) informing the user that the AI is rendering the new design.

**Step 3.2: Status Polling (BE to AI)**
The Backend initiates a polling mechanism. It sends a `GET` request to the `/api/v1/generation-jobs/{job_id}` endpoint at fixed intervals (e.g., every 3 to 5 seconds) to check the status of the generation task.

**Step 3.3: Fault Tolerance Implementation (BE)**
The Backend's polling logic must include a retry mechanism. If the network connection is temporarily dropped or the AI Server fails to respond due to heavy load, the Backend must catch the timeout exception, wait, and retry the request rather than throwing a critical error to the Frontend.

**Step 3.4: Completion and Retrieval (AI to BE)**
Once the background worker finishes the rendering process, the status of the job updates to "success", and the payload includes the URL of the newly generated image. The Backend retrieves this URL during its next polling cycle.

**Step 3.5: Final Delivery (BE to FE)**
The Backend saves the generated image URL and the associated metadata to the database. It then pushes the final URL to the Frontend. The Frontend removes the loading state and displays the newly generated, trend-aligned fashion design to the user.