from typing import Dict


class PromptEngineer:
    """Builds generation prompts from user trend/style input."""

    def build_prompt(self, target_prompt: str) -> Dict[str, str]:
        """Create positive and negative prompts for fashion design generation."""
        sanitized_prompt = " ".join(target_prompt.split())[:600]
        royal_aesthetic = (
            "British royal fashion aesthetic, Savile Row tailoring, "
            "premium velvet and heavy tweed materials, intricate gold button details, "
            "regal aristocratic vibe, elegant luxury menswear"
        )
        base_prompt = (
            "high-end fashion product photography, sharp tailoring, realistic fabric texture, "
            "studio lighting, detailed garment construction, clean background, 8k resolution"
        )
        negative_prompt = (
            "low quality, blurry, distorted garment, messy seams, bad anatomy, extra limbs, "
            "text, watermark, logo artifacts, cropped garment, duplicate clothing, cheap fabric"
        )
        return {
            "prompt": f"{sanitized_prompt}, {royal_aesthetic}, {base_prompt}",
            "negative_prompt": negative_prompt,
        }
