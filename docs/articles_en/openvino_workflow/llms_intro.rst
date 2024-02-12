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

**Quick Start Example**

Here is a quick example of how to run a Llama2 model using OpenVINO optimizations for CPU. First, set up a Python virtual environment for OpenVINO by following the OpenVINO Installation Instructions.

Once the environment is created and activated, install Optimum Intel, OpenVINO, NNCF and their dependencies in a Python environment by issuing:

.. code-block:: python

    pip install optimum[openvino,nncf]

Run the code shown below to download an LLM from Hugging Face, convert it to OpenVINO IR format, and run text generation on an input prompt.

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


