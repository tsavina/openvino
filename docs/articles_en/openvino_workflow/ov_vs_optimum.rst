.. {#native_vs_hugging_face_api}

Hugging Face vs Native OpenVINO API
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

OpenVINO offers two main paths for Generative AI use cases:

* **Hugging Face**: use OpenVINO as a backend for Hugging Face frameworks (transformers, diffusers) through the `Optimum Intel <https://huggingface.co/docs/optimum/intel/inference>`__ extension.

* **Native OpenVINO**: use OpenVINO native APIs (Python and C++) with custom pipeline code.

In both cases, the OpenVINO runtime is used for inference,
and OpenVINO tools are used for optimization. The main differences are
in footprint size, ease of use and customizability.

The Hugging Face API is easy to learn, provides a simple interface and hides
the complexity of model initialization and text generation for a better developer experience.
However, it has more dependencies, less customization, and cannot be ported to C/C++.

The Native OpenVINO API requires fewer dependencies,
mininmizing the application footprint, and enables the use
of generative models in C++ applications. However, it requires explicit
implementation of the text generation loop, tokenization functions,
and scheduler functions used in a typical LLM pipeline.

It is recommended to start with Hugging Face frameworks to experiment with
different models and scenarios. Then the model can be used with OpenVINO native APIs
if it needs to be optimized further. Optimum Intel provides interfaces that enable model optimization (weight compression)
using `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__,
and export models to the OpenVINO model format for use in native API applications.


The table below summarizes the differences between Hugging Face and the native OpenVINO API approaches.

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * -
     - Hugging Face through OpenVINO
     - OpenVINO Native API
   * - Model support
     - Supports transformer-based models such as LLMs
     - Supports all model architectures from most frameworks
   * - APIs
     - Python (Hugging Face API)
     - Python, C++ (OpenVINO API)
   * - Model Format
     - Source Framework / OpenVINO
     - Source Framework / OpenVINO
   * - Inference code
     - Hugging Face based
     - Custom inference pipelines
   * - Additional dependencies
     - Many Hugging Face dependencies
     - Lightweight (e.g. numpy, etc.)
   * - Application footprint
     - Large
     - Small
   * - Pre/post-processing and glue code
     - Provided through high-level Hugging Face APIs
     - Must be custom implemented (see OpenVINO samples and notebooks)
   * - Performance
     - Good, but less efficient compared to native APIs
     - Inherent speed advantage with C++, but requires hands-on optimization
   * - Flexibility
     - Constrained to Hugging Face API
     - High flexibility with Python and C++; allows custom coding
   * - Learning Curve and Effort
     - Lower learning curve; quick to integrate
     - Higher learning curve; requires more effort in integration
   * - Ideal Use Case
     - Ideal for quick prototyping and Python-centric projects
     - Best suited for high-performance, resource-optimized production environments
   * - Model Serving
     - Paid service, based on CPU/GPU usage with Hugging Face
     - Free code solution, run script for own server; costs may incur for cloud services like AWS but generally cheaper than Hugging Face rates

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
