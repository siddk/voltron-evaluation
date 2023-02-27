# Project Roadmap

We document the future of this project (new features to be added, issues to address) here. For the most part, any
new features/bugfixes are documented as [Github Issues](https://github.com/siddk/voltron-evaluation/issues).

## Timeline

[X] - **February 26th, 2023**: Initial V-Evaluation release with support for Grasp Affordance Prediction & Referring
                               Expression Grounding pipelines, with links to run Visuomotor Control externally
                               (through R3M repository).

[ ] - **March 4, 2023**: [#1](https://github.com/siddk/voltron-evaluation/issues/1) – Refactor and consolidate the
                         Visuomotor Control Evaluation (Franka Kitchen & Adroit) into the V-Evaluation Harness API;
                         requires figuring out how to add Mujoco environments ergonomically.

[ ] - **March 7, 2023**: [#2](https://github.com/siddk/voltron-evaluation/issues/2) – Get permission to use WHiRL videos,
                         add zero-shot intent scoring example.

[ ] - **Future**:        [#4](https://github.com/siddk/voltron-evaluation/issues/4) – Create a unified evaluation runner and
                         add better API documentation for extending V-Evaluation with other tasks. Should support single
                         file to run *all downstream evaluations*.

[ ] - **Future**:        [#5](https://github.com/siddk/voltron-evaluation/issues/5) – Upload to PyPI + host data on HF Hub.
