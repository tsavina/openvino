.. {#llm_inference}

LLM Inference in OpenVINO
========================================

OpenVINO provides optimized inference for Large Language Models (LLMs). This page explains how
to perform LLM inference using either the Hugging Face Optimum Intel API or the native OpenVINO API.

Before performing inference, a model must be converted into OpenVINO IR format. This conversion
occurs automatically when loading an LLM from Hugging Face with the Optimum Intel library.
For more information on how to load LLMs in OpenVINO, see :doc:`Loading an LLM to OpenVINO <gen_ai_guide>`.

Inference with Hugging Face
############################


**Installation**

1. Create a virtual environment

.. code-block:: python

  python -m venv openvino_llm

``openvino_llm`` is an example name; you can choose any name for your environment.

2. Activate the virtual environment

.. code-block:: python

  source openvino_llm/bin/activate

3. Install the libraries

.. code-block:: python

  pip install transformers optimum[openvino,nncf]

Inference Example
+++++++++++++++++++++++++++

For Hugging Face models, the ``AutoTokenizer`` and the ``pipeline`` function are used to create
an inference pipeline. This setup allows for easy text processing and model interaction:

.. code-block:: python

  from optimum.intel import OVModelForCausalLM
  # new imports for inference
  from transformers import AutoTokenizer

  # load the model
  model_id = "meta-llama/Llama-2-7b-chat-hf"
  model = OVModelForCausalLM.from_pretrained(model_id, export=True)

  # inference
  prompt = "The weather is:"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  inputs = tokenizer(prompt, return_tensors="pt")

  outputs = model.generate(**inputs, max_new_tokens=50)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))

.. note::

  Converting LLMs on the fly every time to OpenVINO IR is a resource intensive task.
  It is a good practice to convert the model once, save it in a folder and load it for inference.

By default, inference will run on CPU. To switch to a different device, the ``device`` attribute
from the ``from_pretrained`` function can be used. The device naming convention is the
same as in OpenVINO native API:

.. code-block:: python

  model = OVModelForCausalLM.from_pretrained(model_id, export=True, device="GPU")

For more information on how to run text generation with Huggin Face APIs, see their documentation:

* `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`__
* `Generation with LLMs <https://huggingface.co/docs/transformers/llm_tutorial>`__
*	`Pipeline class <https://huggingface.co/docs/transformers/main_classes/pipelines>`__

Inference with OpenVINO Optimum-Intel API
##########################################

Inference can also be run on LLMs using the native OpenVINO API. To perform inference, models
must be first converted to OpenVINO IR format using Hugging Face Optimum-Intel API.

An inference pipeline for a text generation LLM is set up in the following stages:

1.	Read and compile the model in OpenVINO IR.
2.	Pre-process text prompt with a tokenizer and set the result as model inputs.
3.	Run token generation loop.
4.	De-tokenize outputs.

**Prerequisites**

Linux operating system (as of the current version).

**Installation**

1. Create a virtual environment

.. code-block:: python

  python -m venv openvino_llm

``openvino_llm`` is an example name; you can choose any name for your environment.

2. Activate the virtual environment

.. code-block:: python

  source openvino_llm/bin/activate

3. Install OpenVINO tokenizers and dependencies

.. code-block:: python

  pip install transformers optimum[transformers]


Convert Hugging Face tokenizer and model to OpenVINO IR format
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Convert Tokenizer**

`OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations/user_ie_extensions/tokenizer/python#openvino-tokenizers>`__
come equipped with a CLI tool that facilitates the conversion of tokenizers
from either the Hugging Face Hub or those saved locally to the OpenVINO IR format:

.. code-block:: python

  convert_tokenizer microsoft/Llama2-7b-WhoIsHarryPotter --with-detokenizer -o openvino_tokenizer

In this example, the ``microsoft/Llama2-7b-WhoIsHarryPotter tokenizer`` is transformed from the Hugging
Face hub. You can substitute this tokenizer with one of your preference. You can also rename
the output directory (``openvino_tokenizer``).

**Convert Model**

The optimum-cli command can be used for converting a Hugging Face model to the OpenVINO IR model format.
Learn more in Loading an LLM with OpenVINO.

.. code-block:: python

  optimum-cli export openvino --model nickypro/tinyllama-15M openvino_model

Full OpenVINO Text Generation Pipeline
+++++++++++++++++++++++++++++++++++++++

1.	Import and Compile Models.

Use the model and tokenizer converted from the previous step:

.. code-block:: python

  import numpy as np
  from openvino import compile_model
  from openvino_tokenizers import unpack_strings

  # Compile the tokenizer, model, and detokenizer using OpenVINO. These files are XML representations of the models optimized for OpenVINO
  compiled_tokenizer = compile_model("openvino_tokenizer.xml")
  compiled_model = compile_model("openvino_model.xml")
  compiled_detokenizer = compile_model("openvino_detokenizer.xml")

2.	Tokenize and transform Input.

Tokenization is a mandatory step in the process of generating text using LLMs. Tokenization
converts the input text into a sequence of tokens, which are essentially the format that the
model can understand and process. The input text string must be tokenized and set up in the
structure expected by the model before running inference.

.. code-block:: python

  text_input = ["Quick brown fox was"]
  ov_input = compiled_tokenizer(text_input)

3.	Generate Tokens.

The core of text generation lies in the inference and token selection loop. In each iteration
of this loop, the model runs inference on the input sequence, generates and selects a new token,
and appends it to the existing sequence.

.. code-block:: python

  # Define the number of new tokens to generate
  new_tokens_size = 10

  # Determine the size of the existing prompt
  prompt_size = ov_input["input_ids"].shape[-1]

  # Prepare the input dictionary for the model
  # It combines existing tokens with additional space for new tokens
  input_dict = {
    output.any_name: np.hstack([tensor, np.zeros(shape=(1, new_tokens_size), dtype=np.int_)])
    for output, tensor in ov_input.items()
  }

  # Generate new tokens iteratively
  for idx in range(prompt_size, prompt_size + new_tokens_size):
      # Get output from the model
      output = compiled_model(input_dict)["token_ids"]
      # Update the input_ids with newly generated token
      input_dict["input_ids"][:, idx] = output[:, idx - 1]
      # Update the attention mask to include the new token
      input_dict["attention_mask"][:, idx] = 1

4.	Decode and Display Output

The final step in the process is de-tokenization, where the sequence of token IDs generated by
the model is converted back into human-readable text.
This step is essential for interpreting the model's output.

.. code-block:: python

  # Extract token IDs for the final output
  ov_token_ids = input_dict["input_ids"]
  # Decode the model output back to string
  ov_output = unpack_strings(compiled_detokenizer(ov_token_ids)["string_output"])
  print(f"OpenVINO output string: `{ov_output}`")

.. code-block:: python
  # Example output:
  ['<s> Quick brown fox was walking through the forest. He was looking for something']


