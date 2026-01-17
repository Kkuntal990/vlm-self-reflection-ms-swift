"""
VLM Judge Implementations.

This package contains concrete implementations of VLM judges for
evaluating self-refinement in vision-language models.

Available judges:
    - LLaVACriticR1Judge: LLaVA-Critic-R1 (SOTA, based on Qwen2.5-VL)
    - LLaVACriticJudge: Original LLaVA-Critic (fallback)

Usage:
    from vlm_judge import create_judge

    # Use factory (recommended)
    judge = create_judge("llava_critic_r1")

    # Or import directly
    from judges.llava_critic_r1 import LLaVACriticR1Judge
    judge = LLaVACriticR1Judge()
"""

from .llava_critic_r1 import LLaVACriticJudge, LLaVACriticR1Judge


__all__ = ["LLaVACriticR1Judge", "LLaVACriticJudge"]
