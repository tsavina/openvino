<div align="center">
<img src="docs/dev/assets/openvino-logo-purple-black.svg" width="400px">

<h3 align="center">
Open-source software toolkit for optimizing and deploying deep learning models.
</h3>

<p align="center">
 <a href="https://docs.openvino.ai/2024/index.html"><b>Documentation</b></a> • <a href="https://blog.openvino.ai"><b>Blog</b></a> • <a href="https://docs.openvino.ai/2024/about-openvino/key-features.html"><b>Key Features</b></a> • <a href="https://docs.openvino.ai/2024/learn-openvino.html"><b>Tutorials</b></a> • <a href="https://docs.openvino.ai/2024/documentation/openvino-ecosystem.html"><b>Integrations</b></a> • <a href="https://docs.openvino.ai/2024/about-openvino/performance-benchmarks.html"><b>Benchmarks</b></a>
</p>


| PyPI | Anaconda | Homebrew |
|------|----------|----------|
| [![PyPI Status](https://badge.fury.io/py/openvino.svg)](https://badge.fury.io/py/openvino) | [![Anaconda Status](https://anaconda.org/conda-forge/openvino/badges/version.svg)](https://anaconda.org/conda-forge/openvino) | [![brew Status](https://img.shields.io/homebrew/v/openvino)](https://formulae.brew.sh/formula/openvino) |
| [![PyPI Downloads](https://static.pepy.tech/badge/openvino)](https://pepy.tech/project/openvino) | [![Anaconda Downloads](https://anaconda.org/conda-forge/libopenvino/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino/files) | [![brew Downloads](https://img.shields.io/homebrew/installs/dy/openvino)](https://formulae.brew.sh/formula/openvino) |


 </div>



<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'Tab1')">Tab 1</button>
  <button class="tablinks" onclick="openTab(event, 'Tab2')">Tab 2</button>
  <button class="tablinks" onclick="openTab(event, 'Tab3')">Tab 3</button>
</div>

<div id="Tab1" class="tabcontent">
  <h3>Tab 1</h3>
  <p>1111Content for Tab 1.</p>
</div>

<div id="Tab2" class="tabcontent">
  <h3>Tab 2</h3>
  <p>222Content for Tab 2.</p>
</div>

<div id="Tab3" class="tabcontent">
  <h3>Tab 3</h3>
  <p>333Content for Tab 3.</p>
</div>

<style>
  .tab {
    overflow: hidden;
    border-bottom: 1px solid #ccc;
  }

  .tab button {
    background-color: inherit;
    border: none;
    outline: none;
    cursor: pointer;
    padding: 14px 16px;
    transition: 0.3s;
  }

  .tab button:hover {
    background-color: #ddd;
  }

  .tab button.active {
    background-color: #ccc;
  }

  .tabcontent {
    display: none;
    padding: 6px 12px;
    border-top: none;
  }
</style>

<script>
  function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
  }
</script>





### OpenVINO Project

-   [🤗Optimum Intel](https://github.com/huggingface/optimum-intel) -  a simple interface to optimize Transformers and Diffusers models.
-   [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - advanced model optimization techniques including quantization, filter pruning, binarization, and sparsity.
-   [GenAI Repository](https://github.com/openvinotoolkit/openvino.genai) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) - resources and tools for developing and optimizing Generative AI applications.
-   [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving models optimized for Intel architectures.
-   [Intel® Geti™](https://geti.intel.com/) - an interactive video and image annotation tool for computer vision use cases.

### Integrations

- hugging face
- torch.compile
- onnx rt ov ep
- ultralytics
- llamaindex
- langchain
- vllm
- triton
- torchserve
- qwenagent
- spark-nlp

By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
\* Other names and brands may be claimed as the property of others.

<!--- **TensorFlow Model**

```python
import numpy as np
import openvino as ov
import tensorflow as tf

# load TensorFlow model into memory
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# convert the model into OpenVINO model
ov_model = ov.convert_model(model)

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# infer the model on random data
data = np.random.rand(1, 224, 224, 3)
output = compiled_model({0: data})
``` -->

<table>
<tr>
<td> Status </td> <td> Response </td>
</tr>
<tr>
<td> 200 </td>
<td>

↑ Blank line!
```json
json
{
    "id": 10,
    "username": "alanpartridge",
    "email": "alan@alan.com",
    "password_hash": "$2a$10$uhUIUmVWVnrBWx9rrDWhS.CPCWCZsyqqa8./whhfzBZydX7yvahHS",
    "password_salt": "$2a$10$uhUIUmVWVnrBWx9rrDWhS.",
    "created_at": "2015-02-14T20:45:26.433Z",
    "updated_at": "2015-02-14T20:45:26.540Z"
}
```
↓ Blank line!

</td>
</tr>
<tr>
<td> 400 </td>
<td>

**Markdown** _here_. (↕︎ Blank lines above and below!)

</td>
</tr>
</table>

<table>
  <thead>
    <tr>
      <th>OpenVINO with PyTorch</th>
      <th>OpenVINO with TensorFlow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>

```python
import openvino as ov
import torch
import torchvision

# load PyTorch model into memory
model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", weights="DEFAULT")

# convert the model into OpenVINO model
example = torch.randn(1, 3, 224, 224)
ov_model = ov.convert_model(model, example_input=(example,))

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# infer the model on random data
output = compiled_model({0: example.numpy()})

   </td>
 <td>


import numpy as np
import openvino as ov
import tensorflow as tf

# load TensorFlow model into memory
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# convert the model into OpenVINO model
ov_model = ov.convert_model(model)

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# infer the model on random data
data = np.random.rand(1, 224, 224, 3)
output = compiled_model({0: data})

  </td>
</tr>

``` </tbody> </table>