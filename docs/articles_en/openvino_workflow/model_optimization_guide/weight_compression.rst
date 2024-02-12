.. {#weight_compression}

Weight Compression
==================


Enhancing Model Efficiency with Weight Compression
##################################################################

Weight compression is a technique for enhancing the efficiency of models,
especially those with large memory requirements. This method reduces the model's
memory footprint, a crucial factor for Large Language Models (LLMs).

Unlike full model quantization, where weights and activations are quantized,
weight compression in `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__
only targets the model's weights. This approach
allows the activations to remain as floating-point numbers, preserving most
of the model's accuracy while improving its speed and reducing
its size.

The reduction in size is especially noticeable with larger models,
for instance the 7 billion parameter Llama 2 model can be reduced
from about 25GB to 4GB using 4-bit weight compression. With smaller models (i.e. less than 1B parameters),
weight compression may result in more accuracy reduction than with larger models.

LLMs and other models that require
extensive memory to store the weights during inference can benefit
from weight compression as it:

* enables inference of exceptionally large models
that cannot be accommodated in the device memory;

* reduces storage and memory overhead, making models
more lightweight and less resource intensive for deployment;

* improves inference speed by reducing
the latency of memory access when computing the
operations with weights, for example, Linear layers.
The weights are smaller and thus faster to load from memory;

* unlike quantization, does not require
sample data to calibrate the range of activation values.

Currently, `NNCF <https://github.com/openvinotoolkit/nncf>`__
provides weight quantization to 8 and 4-bit integer data types as a compression
method primarily designed to optimize LLMs.

Compress Model Weights
######################

- **8-bit weight quantization** - this method offers a balance between model size reduction
and maintaining accuracy, which usually leads to significant performance improvements for Transformer-based models.
Models with 8-bit compressed weights are performant on the vast majority of supported CPU and GPU platforms.

The code snippet below shows how to do INT8 weight compression of the model
weights represented on an OpenVINO IR using NNCF:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_8bit]

Now, the model is ready for compilation and inference.
It can be also saved into a compressed format, resulting in a smaller binary file.

- **4-bit weight quantization** - this method stands for an INT4-INT8
mixed-precision weight quantization, where INT4 is considered as the
primary precision and INT8 is the backup one. It usually results in a
smaller model size and lower inference latency, although the accuracy
degradation could be higher, depending on the model.

The table below summarizes the benefits and trade-offs for each compression type in terms of memory reduction, speed gain, and accuracy loss.

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * -
     - Memory Reduction
     - Latency Improvement
     - Accuracy Loss
   * - INT8
     - Low
     - Medium
     - Low
   * - INT4 Symmetric
     - High
     - High
     - High
   * - INT4 Asymmetric
     - High
     - Medium
     - Medium

The INT4 method has several parameters that can provide different performance-accuracy trade-offs after optimization:

* ``mode`` - there are two modes to choose from: ``INT4_SYM`` - stands
for INT4 symmetric weight quantization and results in faster inference
and smaller model size, and ``INT4_ASYM`` - INT4 asymmetric weight quantization
with variable zero-point for more accurate results.

**Symmetric Compression**

INT4 Symmetric mode involves quantizing weights to an unsigned
4-bit integer symmetrically with a fixed zero point of 8. This
mode is faster than the INT8, making it ideal for situations
where speed is prioritized over accuracy. Although it may lead
to some degradation in accuracy, it is well-suited for models
where this trade-off is acceptable for a noticeable gain in speed and size reduction.

.. code-block:: python

  from nncf import compress_weights
  from nncf import CompressWeightsMode

  compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_SYM)

**Asymmetric Compression**

INT4 Asymmetric mode also uses an unsigned 4-bit integer but quantizes weights
asymmetrically with a non-fixed zero point. This mode slightly compromises speed
in favor of better accuracy compared to the symmetric mode. This mode is useful
when minimal accuracy loss is crucial, but a faster performance than INT8 is still desired.

.. code-block:: python

  from nncf import compress_weights
  from nncf import CompressWeightsMode

  compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_ASYM)

* ``group_size`` - controls the size of the group of weights that share
the same quantization parameters. Shared quantization parameters help to
speed up the calculation of activation values as they are dequantized and
quantized between layers. However, they can reduce accuracy.
The following group sizes are recommended: ``128``, ``64``, ``32`` (``128`` is default value).

**Smaller Group Size**: Leads to a more accurate model but increases
the model's footprint and reduces inference speed.
**Larger Group Size**: Results in faster inference and a smaller model,
but might compromise accuracy.

* ``ratio`` - controls the ratio between INT4 and INT8 compressed layers
in the model. Ratio is a decimal between 0 and 1.
For example, 0.8 means that 80% of layers will be compressed
to INT4, while the rest will be compressed to INT8 precision.
The default value for ratio is 1.

**Higher Ratio (more INT4)**: Tends to reduce the model size and increase
inference speed but might lead to higher accuracy degradation.
**Lower Ratio (more INT8)**: Maintains better accuracy but results in
a larger model size and potentially slower inference.

In this example, 90% of the model's layers are quantized to INT4 asymmetrically with a group size of 64:

.. code-block:: python

  from nncf import compress_weights, CompressWeightsMode

  # Example: Compressing weights with INT4_ASYM mode, group size of 64, and 90% INT4 ratio
  compressed_model = compress_weights(
    model,
    mode=CompressWeightsMode.INT4_ASYM,
    group_size=64,
    ratio=0.9,
  )

* ``dataset`` - calibration dataset for data-aware weight compression.
It is required for some compression options, for example, some types
``sensitivity_metric`` can use data for precision selection.

* ``sensitivity_metric`` - controls the metric to estimate the sensitivity
of compressing layers in the bit-width selection algorithm. Some of the metrics
require dataset to be provided. The following types are supported:

  * ``nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR`` - data-free metric computed as the inverted 8-bit quantization noise. Weights with highest value of this metric can be accurately quantized channel-wise to 8-bit. The idea is to leave these weights in 8 bit, and quantize the rest of layers to 4-bit group-wise. Since group-wise is more accurate than per-channel, accuracy should not degrade.

  * ``nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION`` - requires dataset. The average Hessian trace of weights with respect to the layer-wise quantization error multiplied by L2 norm of 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MEAN_ACTIVATION_VARIANCE`` - requires dataset. The mean variance of the layers' inputs multiplied by inverted 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE`` - requires dataset. The maximum variance of the layers' inputs multiplied by inverted 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE`` - requires dataset. The mean magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.


The example below shows data-free 4-bit weight quantization
applied on top of OpenVINO IR. Before trying the example, make sure Optimum Intel
is installed in your environment by running the following command:

.. code-block:: python

  pip install optimum[openvino,nncf]

The first example loads a pre-trained Hugging Face model using the Optimum Intel API,
compresses it using NNCF, and then executes inference with a text phrase.

If the model comes from Hugging Face and is supported by Optimum, then it can
be easier to simply use the Optimum Intel API to perform weight compression. The compression
type is specified when the model is loaded using the ``load_in_8bit=True`` or ``load_in_4bit=True`` parameter.
The second example uses the Weight Compression API from Optimum Intel instead of NNCF to compress the model to INT8

.. tab-set::

   .. tab-item:: NNCF
      :sync: openvino

      .. code-block:: python

        from nncf import compress_weights, CompressWeightsMode
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer, pipeline

        # Load model from Hugging Face
        model_id = "HuggingFaceH4/zephyr-7b-beta"
        model = OVModelForCausalLM.from_pretrained(model_id, export=True)

        # Compress to INT4 Symmetric
        model.model = compress_weights(model.model,  mode=CompressWeightsMode.INT4_SYM)

        # Inference
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        phrase = "The weather is"
        results = pipe(phrase)
        print(results)

   .. tab-item:: Optimum-Intel
      :sync:

      .. code-block:: python

        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer, pipeline

        # Load and compress model from Hugging Face
        model_id = "HuggingFaceH4/zephyr-7b-beta"
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)

        # Inference
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        phrase = "The weather is"
        results = pipe(phrase)
        print(results)


For data-aware weight compression refer to the following `example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino>`__.

**Exporting and Loading Compressed Models**

Once a model has been compressed with NNCF or Optimum Intel,
it can be saved and exported to use in a future session or in a
deployment environment. The compression process takes a while,
so it is preferable to compress the model once, save it, and then
load the compressed model later for faster time to first inference.

.. code-block:: python

  # Save compressed model for faster loading later
  model.save_pretrained("zephyr-7b-beta-int4-sym-ov")
  tokenizer.save_pretrained("zephyr-7b-beta-int4-sym-ov")

  # Load a saved model
  model = OVModelForCausalLM.from_pretrained("zephyr-7b-beta-int4-sym-ov")
  tokenizer = AutoTokenizer.from_pretrained("zephyr-7b-beta-int4-sym-ov")


**GPTQ Models**

OpenVINO also supports 4-bit models from Hugging Face
`Transformers <https://github.com/huggingface/transformers>`__ library optimized
with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case, there is no
need for an additional model optimization step because model conversion will
automatically preserve the INT4 optimization results, allowing model inference to benefit from it.

An end-to-end compression example using a GPTQ model is shown below.
To successfully run the example, make sure to installed GPTQ dependencies by running the following command:

.. code-block:: python

  pip install optimum[openvino] auto-gptq

.. code-block:: python

  from optimum.intel.openvino import OVModelForCausalLM
  from transformers import AutoTokenizer, pipeline

  # Load model from Hugging Face already optimized with GPTQ
  model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
  model = OVModelForCausalLM.from_pretrained(model_id, export=True)

  # Inference
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
  phrase = "The weather is"
  results = pipe(phrase)
  print(results)

An `example of a model <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ>`__ that has been optimized using GPTQ.


The table below shows examples of Text Generation models with different optimization settings:

.. list-table::
   :widths: 40 55 25 25
   :header-rows: 1

   * - Model
     - Optimization
     - Perplexity
     - Model Size (Gb)
   * - databricks/dolly-v2-3b
     - FP32
     - 5.01
     - 10.3
   * - databricks/dolly-v2-3b
     - INT8
     - 5.07
     - 2.6
   * - databricks/dolly-v2-3b
     - INT4_ASYM,group_size=32,ratio=0.5
     - 5.28
     - 2.2
   * - facebook/opt-6.7b
     - FP32
     - 4.25
     - 24.8
   * - facebook/opt-6.7b
     - INT8
     - 4.27
     - 6.2
   * - facebook/opt-6.7b
     - INT4_ASYM,group_size=64,ratio=0.8
     - 4.32
     - 4.1
   * - meta-llama/Llama-2-7b-chat-hf
     - FP32
     - 3.28
     - 25.1
   * - meta-llama/Llama-2-7b-chat-hf
     - INT8
     - 3.29
     - 6.3
   * - meta-llama/Llama-2-7b-chat-hf
     - INT4_ASYM,group_size=128,ratio=0.8
     - 3.41
     - 4.0
   * - togethercomputer/RedPajama-INCITE-7B-Instruct
     - FP32
     - 4.15
     - 25.6
   * - togethercomputer/RedPajama-INCITE-7B-Instruct
     - INT8
     - 4.17
     - 6.4
   * - togethercomputer/RedPajama-INCITE-7B-Instruct
     - INT4_ASYM,group_size=128,ratio=1.0
     - 4.17
     - 3.6
   * - meta-llama/Llama-2-13b-chat-hf
     - FP32
     - 2.92
     - 48.5
   * - meta-llama/Llama-2-13b-chat-hf
     - INT8
     - 2.91
     - 12.1
   * - meta-llama/Llama-2-13b-chat-hf
     - INT4_SYM,group_size=64,ratio=0.8
     - 2.98
     - 8.0


Additional Resources
####################

- `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__: Repository containing example pipelines
that implement image and text generation tasks.
It also provides a tool to benchmark LLMs.
- `LLM Compression Jupyter Notebook  <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot>`__
- `Data-aware weight compression <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino>`__
- :doc:`Post-training Quantization <ptq_introduction>`
- :doc:`Training-time Optimization <tmo_introduction>`
- `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__

