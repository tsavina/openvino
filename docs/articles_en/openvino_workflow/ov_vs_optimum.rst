.. {#native_vs_hugging_face_api}

Hugging Face and Native OpenVINO API
========================================

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

