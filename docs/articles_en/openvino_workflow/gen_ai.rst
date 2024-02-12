.. {#gen_ai_guide}

Loading an LLM to OpenVINO
========================================

Running Generative AI Models using Hugging Face Optimum Intel
##############################################################

The steps below show how to load LLMs from Hugging Face using Optimum Intel.
They also show how to convert models into OpenVINO IR format so they can be optimized
by NNCF and used with other OpenVINO tools.

Prerequisites
+++++++++++++++++++++++++++

* Create a Python environment by following the instructions on the :doc:`Install OpenVINO <openvino_docs_install_guides_overview>` page.
* Install the necessary dependencies for Optimum Intel:

.. code-block:: console

    pip install optimum[openvino,nncf]

Loading a Hugging Face Model to Optimum Intel
+++++++++++++++++++++++++++++++++++++++++++++++++++++


To start using OpenVINO as a backend for Hugging Face, change the original
Hugging Face code with an Optimum Intel equivalent in two places:

.. code-block:: diff

    -from transformers import AutoModelForCausalLM
    +from optimum.intel import OVModelForCausalLM

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    -model = AutoModelForCausalLM.from_pretrained(model_id)
    +model = OVModelForCausalLM.from_pretrained(model_id, export=True)

Instead of using ``AutoModelForCasualLM`` from the Hugging Face transformers library,
switch to ``OVModelForCasualLM`` from the `optimum.intel` library. This change enables
you to use OpenVINO's optimization features. You may also use other AutoModel types,
such as ``OVModelForSeq2SeqLM``, though this guide will focus on CausalLM.

By setting the parameter ``export=True``, the model is converted to OpenVINO IR format on the fly.

After that, you can call ``save_pretrained()`` method to save the model to use it further.

.. code-block:: python

    model.save_pretrained("ov_model")

This will create a new folder called `ov_model` with the LLM in OpenVINO IR format inside.
You can change the folder and provide another model directory instead of `ov_model`.

Once the model is saved, you can load it with the following command:

.. code-block:: python

    model = OVModelForCausalLM.from_pretrained("ov_model")

Obtaining OpenVINO Model Object
+++++++++++++++++++++++++++++++++++++++++++++++++++++

When you use Intel Optimum for loading, the resulting model is a Hugging
Face model with additional functionalities provided by Optimum.
The model object created in the snippets above is not a native OpenVINO IR model
but rather a Hugging Face model adapted to work with OpenVINO's optimizations.

If you need to access the underlying OpenVINO model object directly, you
can do so through a specific attribute of the Optimum Intel model named ``model``.

To access this native OpenVINO model object, you can assign it to a new variable like this:

.. code-block:: python

    openvino_model = model.model

The first model refers to the Optimum Intel `model` you loaded, and the `.model`
accesses the native OpenVINO model object within it. Now, `openvino_model` holds
the native OpenVINO model, allowing you to interact with it directly,
as you would with a standard OpenVINO model. You can compress the model using `NNCF <https://github.com/openvinotoolkit/nncf>`__
and infer it with a custom OpenVINO pipeline. For more information, see the :doc:`LLM Weight Compression <weight_compression>` page.

If you want to work with Native OpenVINO after loading the model with Optimum Intel,
it is recommended to disable model compilation in the loading function.
Set the compile attribute to False while loading the model:

.. code-block:: python

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, compile=False)

Converting a Hugging Face Model to OpenVINO IR
+++++++++++++++++++++++++++++++++++++++++++++++++++++

The optimum-cli tool allows you to convert models from Hugging Face to
the OpenVINO IR format:

.. code-block:: python

    optimum-cli export openvino --model <MODEL_NAME> <NEW_MODEL_NAME>

*	--model <MODEL_NAME>: specifies the name of the model you want to convert.
Replace <MODEL_NAME> with the actual model name from Hugging Face.

*	<NEW_MODEL_NAME>: specify the name you want to give to the new model
in the OpenVINO IR format. Replace <NEW_MODEL_NAME> with your desired name.

If you want to convert the `Llama 2` model from Hugging Face to an OpenVINO IR
model and name it `ov_llama_2`, the command would look like this:

.. code-block:: python

    optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf ov_llama_2

In this case, you can load the converted model in OpenVINO representation directly from the disk:

.. code-block:: python

    model_id = "llama_openvino"
    model = OVModelForCausalLM.from_pretrained(model_id)


By default, inference will run on CPU. To select a different inference device,
for example, GPU, add ``device="GPU"`` to the ``from_pretrained()`` call.
To switch to a different device after the model has been loaded, use
the ``.to()`` method. The device naming convention is the same as in OpenVINO native API:

.. code-block:: python

    model.to("GPU")


Optimum-Intel API also provides out-of-the-box model optimization through
weight compression using NNCF which substantially reduces the model footprint and inference latency:

.. code-block:: python

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)


Weight compression is applied by default to models larger
than one billion parameters and is also available for CLI interface as the ``--int8`` option.

.. note::

   8-bit weight compression is enabled by default for models larger than 1 billion parameters.

`NNCF <https://github.com/openvinotoolkit/nncf>`__ also provides 4-bit weight compression,
which is supported by OpenVINO. It can be applied to Optimum objects as follows:

.. code-block:: python

    from nncf import compress_weights, CompressWeightsMode

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False)
    model.model = compress_weights(model.model, mode=CompressWeightsMode.INT4_SYM, group_size=128, ratio=0.8)


The optimized model can be saved as usual with a call to ``save_pretrained()``.
For more details on compression options, refer to the :doc:`weight compression guide <weight_compression>`.

.. note::

   OpenVINO also supports 4-bit models from Hugging Face `Transformers <https://github.com/huggingface/transformers>`__ library optimized
   with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case, there is no need for an additional model optimization step because model conversion will automatically preserve the INT4 optimization results, allowing model inference to benefit from it.


Below are some examples of using Optimum-Intel for model conversion and inference:

* `Stable Diffusion v2.1 using Optimum-Intel OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/236-stable-diffusion-v2/236-stable-diffusion-v2-optimum-demo.ipynb>`__
* `Image generation with Stable Diffusion XL and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/248-stable-diffusion-xl/248-stable-diffusion-xl.ipynb>`__
* `Instruction following using Databricks Dolly 2.0 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/240-dolly-2-instruction-following/240-dolly-2-instruction-following.ipynb>`__
* `Create an LLM-powered Chatbot using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb>`__

Working with Models Tuned with LoRA
++++++++++++++++++++++++++++++++++++

Low-rank Adaptation (LoRA) is a popular method to tune Generative AI models to a downstream task or custom data.
However, it requires some extra steps to be done for efficient deployment using the Hugging Face API.
Namely, the trained adapters should be fused into the baseline model to avoid extra computation.
This is how it can be done for LLMs:

.. code-block:: python

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    lora_adaptor = "./lora_adaptor"

    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
    model = PeftModelForCausalLM.from_pretrained(model, lora_adaptor)
    model.merge_and_unload()
    model.get_base_model().save_pretrained("fused_lora_model")


Now the model can be converted to OpenVINO using Optimum Intel Python API or CLI interfaces mentioned above.

Running Generative AI Models using Native OpenVINO APIs
########################################################

To run Generative AI models using native OpenVINO APIs you need to follow regular **Сonvert -> Optimize -> Deploy** path with a few simplifications.

To convert model from Hugging Face you can use Optimum-Intel export feature that allows to export model in OpenVINO format without invoking conversion API and tools directly, as it is shown above. In this case, the conversion process is a bit more simplified. You can still use a regular conversion path if model comes from outside of Hugging Face ecosystem, i.e., in source framework format (PyTorch, etc.)

Model optimization can be performed within Hugging Face or directly using NNCF as described in the :doc:`weight compression guide <weight_compression>`.

Inference code that uses native API cannot benefit from Hugging Face pipelines. You need to write your custom code or take it from the available examples. Below are some examples of popular Generative AI scenarios:

* In case of LLMs for text generation, you need to handle tokenization, inference and token selection loop, and de-tokenization. If token selection involves beam search, it also needs to be written.
* For image generation models, you need to make a pipeline that includes several model inferences: inference for source (e.g., text) encoder models, inference loop for diffusion process and inference for decoding part. Scheduler code is also required.

To write such pipelines, you can follow the examples provided as part of OpenVINO:

* `llama2.openvino <https://github.com/OpenVINO-dev-contest/llama2.openvino>`__
* `LLM optimization by custom operation embedding for OpenVINO <https://github.com/luo-cheng2021/ov.cpu.llm.experimental>`__
* `C++ Implementation of Stable Diffusion <https://github.com/yangsu2022/OV_SD_CPP>`__


Additional Resources
############################

* `Optimum Intel documentation <https://huggingface.co/docs/optimum/intel/inference>`_
* :doc:`LLM Weight Compression <weight_compression>`
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`_

