.. {#llms_intro}

Large Language Models in OpenVINO
========================================

Large Language Models (LLMs) like GPT are transformative deep learning networks
capable of a broad range of natural language tasks, from text generation to language translation.
OpenVINO optimizes the deployment of these models, enhancing their performance and integration into various applications.
This guide shows how to use LLMs with OpenVINO, from model loading and conversion to advanced use cases.

The advantages of using OpenVINO for LLM deployment:

●	OpenVINO requires **fewer dependencies** than frameworks like
Hugging Face and PyTorch, resulting in a **smaller binary size and reduced
memory footprint**, making deployments easier and updates more manageable.

●	Offers **optimized LLM inference**; provides a full **C/C++ API**, leading to faster operation
than Python-based runtimes; includes a **Python API** for rapid development,
with the option for further optimization in C++.

●	Supports a **wide range of deep learning models and architectures** beyond LLMs,
enabling the development of multimodal applications, allowing for **write-once, deploy-anywhere** capabilities.

●	**Compatible with diverse hardware**, including CPUs, GPUs, and neural accelerators
across ARM and x86/x64 architectures; features automated optimization to maximize
performance on target hardware.

Generative AI is an innovative technique that creates new data, such as text, images, video, or audio, using neural networks. OpenVINO accelerates Generative AI use cases as they mostly rely on model inference, allowing for faster development and better performance. When it comes to generative models, OpenVINO supports:

* Conversion, optimization and inference for text, image and audio generative models, for example, Llama 2, MPT, OPT, Stable Diffusion, Stable Diffusion XL, etc.
* Int8 weight compression for text generation models.
* Storage format reduction (fp16 precision for non-compressed models and int8 for compressed models).
* Inference on CPU and GPU platforms, including integrated Intel® Processor Graphics, discrete Intel® Arc™ A-Series Graphics, and discrete Intel® Data Center GPU Flex Series.


1. Choose framework


Hugging Face vs Native OpenVINO

This comparison outlines the key differences between integrating LLMs
with OpenVINO through Hugging Face frameworks and directly using native
OpenVINO APIs. The choice between these approaches depends on the desired
balance between ease of use and customizability for LLM deployment.
