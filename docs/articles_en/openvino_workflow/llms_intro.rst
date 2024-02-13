.. {#llms_intro}


Large Language Models in OpenVINO
========================================

.. toctree::
   :maxdepth: 1
   :hidden:

   Hugging Face and Native OpenVINO API <native_vs_hugging_face_api>
   Generative AI Optimization and Deployment <gen_ai_guide>
   Weight Compression <weight_compression>
   LLM Inference with Hugging Face and Optimum Intel <llm_inference>
   LLM Inference with Native OpenVINO <llm_inference_native_API>

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

The following articles provide more code snippets and information on how to use LLMs in OpenVINO.

:doc:`Hugging Face vs Native OpenVINO <native_vs_hugging_face_api>`

This comparison shows the key differences between integrating LLMs with OpenVINO through
Hugging Face frameworks and directly using native OpenVINO APIs. The choice depends on the
desired balance between ease of use and customizability for LLM deployment.

:doc:`LLM Weight Compression <weight_compression>`

Weight compression in LLMs, essential for performance enhancement, is facilitated by
Neural Network Compression Framework (NNCF). This section covers the techniques and benefits of
weight compression.

:doc:`LLM Inference in OpenVINO  <gen_ai_guide>`

The approach to LLM inference in OpenVINO varies significantly between using Hugging Face API
and native OpenVINO methods. With Hugging Face API, the process is streamlined through a simple
pipeline command, allowing for straightforward model interaction and generation of outputs.
In contrast, native OpenVINO requires a more hands-on approach, where developers must create
their own inference loop. This involves detailed steps like token generation and selection,
offering greater control and customization at the cost of increased complexity.

Both methods cater to different needs, with Hugging Face providing ease of use and native
OpenVINO offering extensive customization options for LLM inference.

LLM Model Serving in OpenVINO

Deploying LLMs on a Model Server involves making them accessible over network protocols.
This includes Unary Model Serving for simple applications and Stream Model Serving for real-time
data streams. The section outlines best practices for setting up a model server to match specific
application needs and optimize model accessibility.


**Quick Start Example**

Here is a quick example of how to run a Llama2 model using OpenVINO optimizations for CPU.
First, set up a Python virtual environment for OpenVINO by following
the :doc:`Install OpenVINO PIP <openvino_docs_install_guides_overview>` Instructions.

Once the environment is created and activated, install Optimum Intel, OpenVINO,
NNCF and their dependencies in a Python environment by issuing:

.. code-block:: python

    pip install optimum[openvino,nncf]

Run the code shown below to download an LLM from Hugging Face, convert it to OpenVINO IR format,
and run text generation on an input prompt.

.. code-block:: python

    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer, pipeline

    # load the model
    model_id = "HuggingFaceH4/zephyr-7b-beta"
    model = OVModelForCausalLM.from_pretrained(model_id, export=True)

    # inference
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=50)
    prompt = "The weather is"
    results = pipe(prompt)

Here is  the output from running the example. Try changing the prompt to see what other text outputs can be generated.

.. code-block:: diff

    The weather is finally starting to warm up, and that means it’s time to start thinking about summer activities.

