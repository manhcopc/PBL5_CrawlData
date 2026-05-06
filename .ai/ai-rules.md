# AI Engineering Rules & Guidelines

When modifying, debugging, or generating code for this repository, you MUST strictly adhere to the following technical rules and constraints.

## 1. AI Modeling Constraints (CRITICAL)
*   **NO Fine-Tuning:** Do NOT suggest or implement model fine-tuning (e.g., LoRA, Dreambooth, or Textual Inversion) for Stable Diffusion or PhoBERT. The system is explicitly designed as a **Zero-shot, Training-free Pipeline**.
*   **Image Conditioning Over Prompting:** To retain the original shape of garments, we rely on **ControlNet (Canny Edge Detection)**. Do not attempt to use standard `StableDiffusionImg2ImgPipeline`. You MUST use `StableDiffusionControlNetPipeline`.
*   **NLP Usage:** PhoBERT is used purely in inference mode (`eval()`) for sequence classification (sentiment analysis) and feature extraction. Do not introduce training loops.

## 2. Memory & Hardware Optimization
The target deployment environment is a Google Colab instance with a single Tesla T4 GPU (~15GB VRAM). Memory management is the highest priority.
*   **CPU Offloading:** When initializing `diffusers` pipelines, ALWAYS implement `pipe.enable_model_cpu_offload()` and `pipe.enable_xformers_memory_efficient_attention()` to prevent CUDA Out-Of-Memory (OOM) errors.
*   **Precision:** ALWAYS load Stable Diffusion and ControlNet models in `torch.float16` precision.
*   **Inference Mode:** Wrap all AI inference calls (both PyTorch and Hugging Face) inside `with torch.no_grad():` or `with torch.autocast("cuda"):` contexts.

## 3. Software Architecture Standards
*   **Singleton Pattern for AI Services:** AI models are massive and take time to load. Service classes (e.g., `PhobertService`, `GenerativeService`) MUST be implemented as Singletons. Models should be loaded into RAM/VRAM exactly once during application startup.
*   **Separation of Concerns:** Keep API routing logic (`app/api/`) completely decoupled from AI inference logic (`app/services/`). API routes should only handle HTTP requests, Pydantic validation, and response formatting.
*   **Error Handling:** If an AI service fails to load, the API must not crash the entire application. Use `is_ready` flags in the service constructors and return graceful HTTP 500 errors if a model is unavailable.

## 4. Coding Standards
*   **Language:** Python 3.10+.
*   **Typing:** Use Python type hints (`List`, `Dict`, `Optional`, etc.) extensively, especially for service method parameters and return types.
*   **API Validation:** Strictly use `pydantic.BaseModel` for defining payload structures in `app/schemas/`.
*   **Logging:** Use clear, descriptive print statements (or the `logging` module) with prefixes like `[PhoBERT Service]` or `[Gen Service]` to trace the execution flow easily.