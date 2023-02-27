"""
evaluate_grasp.py

Example script for loading a pretrained V-Cond model (from the `voltron` library), configuring a MAP-based extractor
factory function, and then defining/invoking the GraspAffordanceHarness.
"""
import torch
from voltron import instantiate_extractor, load

import voltron_evaluation as vet


def evaluate_grasp() -> None:
    # Load Backbone (V-Cond)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, preprocess = load("v-cond", device=device)

    # Create MAP Extractor Factory (for ARC Grasping) --> 4 PuP stages with an 80 x 80 output resolution.
    #   => Because we're segmenting, the `n_latents` is intermingled with the details of the adapter; not really a clean
    #      way to deal with this, so we'll hardcode based on the above.
    output_resolution, upsample_stages = 80, 4
    map_extractor_fn = instantiate_extractor(backbone, n_latents=int((output_resolution**2) / (4**upsample_stages)))

    # Create ARC Grasping Harness
    grasp_evaluator = vet.GraspAffordanceHarness("v-cond", backbone, preprocess, map_extractor_fn)
    grasp_evaluator.fit()
    grasp_evaluator.test()


if __name__ == "__main__":
    evaluate_grasp()
