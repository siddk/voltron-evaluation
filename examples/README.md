# Examples

Useful example scripts for running the various evaluation applications with Voltron models (in these examples, the
Voltron variants).

**Note:** We currently do not have the permission to host or otherwise distribute the raw data for the various tasks
ourselves (for now!); we ask that you follow the instructions below to acquire the various datasets prior to running
each example script!

---

## Grasp Affordance Prediction

Download the Suction-Based Grasping Dataset from the
[ARC Grasping Project Website](https://vision.princeton.edu/projects/2017/arc/).

Unzip the archive to a location of your choice (default: `data/grasping/suction-based`).

Run the evaluation with `python examples/evaluate_grasp.py`.

## Referring Expression Grounding

Download the original OCID dataset and OCID-Ref annotations following the instructions on the
[OCID-Ref Repository](https://github.com/lluma/OCID-Ref#dataset).

Unzip the `OCID-dataset.tar.gz` to a location of your choice (default: `tasks/langref/OCID-dataset`). Place the
downloaded `<split>_expressions.json` files in the same top-level directory
(default: `data/langref/<split>_expressions.json`).

Run the evaluation with `python examples/evaluate_refer.py`.

## Single-Task Visuomotor Control

TBD; figuring out how to adapt the original code (built on top of the Franka Kitchen / Adroit Mujoco Simulation
Environments) is taking a little bit. In the meantime, running the existing code pipeline from the
[R3M Evaluation Branch](https://github.com/facebookresearch/r3m/tree/eval) is the easiest way to get started!

Will try to refactor, test, and streamline this by 3/4/23.

## Intent Inference

TBD; waiting on permissions to share/use videos from the [WHiRL Website](https://human2robot.github.io/).
