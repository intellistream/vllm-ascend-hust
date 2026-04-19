# Welcome to vLLM Hust Ascend

:::{figure} ./logos/vllm-ascend-logo-text-light.png
:align: center
:alt: vLLM
:class: no-scaled-link
:width: 70%
:::

:::{raw} html
<p style="text-align:center">
<strong>vLLM Hust Ascend
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/intellistream/vllm-ascend-hust" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/intellistream/vllm-ascend-hust/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/intellistream/vllm-ascend-hust/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

vLLM Hust Ascend (`vllm-ascend-hust`) is the localized Ascend plugin
distribution for running `vllm-hust` on the Ascend NPU.

The Python import/module namespace remains `vllm_ascend` for compatibility with
the upstream hardware plugin interface.

This plugin is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Experts, Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

## Documentation

% How to start using vLLM on Ascend NPU?
:::{toctree}
:caption: Getting Started
:maxdepth: 1
quick_start
installation
fork_release_install_guide
tutorials/models/index
tutorials/features/index
tutorials/hardwares/index
faqs
:::

% What does vLLM Ascend Plugin support?
:::{toctree}
:caption: User Guide
:maxdepth: 1
user_guide/support_matrix/index
user_guide/configuration/index
user_guide/feature_guide/index
user_guide/deployment_guide/index
user_guide/release_notes
:::

% How to contribute to the vLLM Ascend project
:::{toctree}
:caption: Developer Guide
:maxdepth: 1
developer_guide/contribution/index
developer_guide/feature_guide/index
developer_guide/evaluation/index
developer_guide/performance_and_debug/index
:::

% How to involve vLLM Ascend
:::{toctree}
:caption: Community
:maxdepth: 1
community/governance
community/contributors
community/versioning_policy
community/user_stories/index
:::
