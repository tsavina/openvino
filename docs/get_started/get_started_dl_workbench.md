# Get Started with OpenVINO™ Toolkit via Deep Learning Workbench {#openvino_docs_get_started_get_started_dl_workbench}

The OpenVINO™ toolkit optimizes and runs Deep Learning Neural Network models on Intel® hardware. This guide helps you get started with the OpenVINO™ toolkit via Deep Learning Workbench (DL Workbench). 
 
In this guide, you will learn:
* What DL Workbench is and how it is connected to OpenVINO™ toolkit components
* How to run DL Workbench 
* How to create your first project and measure model performance 

## DL Workbench Overview

DL Workbench is a web-based graphical environment with a convenient interface and a wide range of options designed to make the production of pretrained deep learning models significantly easier. 

DL Workbench combines OpenVINO™ tools to assist you with the most commonly used tasks: import a model, analyze its performance and accuracy, visualize the outputs, optimize and prepare the model for deployment without writing a single line of code.

The intuitive web-based interface of the DL Workbench enables you to easily use various
OpenVINO™ toolkit components:

Component  |                 Description 
|------------------|------------------|
| [Open Model Zoo](https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html)| Get access to the collection of high-quality pre-trained deep learning [public](https://docs.openvinotoolkit.org/latest/omz_models_group_public.html) and [Intel-trained](https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html) models trained to resolve a variety of different tasks. |
| [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) |Optimize and transform models trained in supported frameworks to the IR format. <br>Supported frameworks include TensorFlow\*, Caffe\*, Kaldi\*, MXNet\*, and ONNX\* format.  
| [Benchmark Tool](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_benchmark_tool_README.html)| Estimate deep learning model inference performance on supported devices.   
| [Accuracy Checker](https://docs.openvinotoolkit.org/latest/omz_tools_accuracy_checker.html)| Evaluate the accuracy of a model by collecting one or several metric values. 
| [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_README.html)| Optimize pretrained models with lowering the precision of a model from floating-point precision(FP32 or FP16) to integer precision (INT8), without the need to retrain or fine-tune models.    

\htmlonly
<div class="ovino-btn">
 <a style="color: white;" href="https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Install.html">Install DL Workbench</a>
</div>
\endhtmlonly

![](./dl_workbench_img/DL_Workbench.jpg)

Learn more about the DL Workbench workflow and capabilities at the [DL Workbench Overview](@ref workbench_docs_Workbench_DG_Introduction) page. 

## Prerequisites

Before running DL Workbench:
1. Make sure you have met the recommended requirements listed below.
2. Configure [Docker](https://docs.docker.com/get-docker/) on your machine.

Prerequisite | Linux* | Windows* | macOS*
:----- | :----- |:----- |:-----
Operating system|Ubuntu\* 18.04|Windows\* 10 | macOS\* 10.15 Catalina
Available RAM space| 8 GB\** | 8 GB\** | 8 GB\**
Available storage space| 10 GB + space for imported artifacts| 10 GB + space for imported artifacts| 10 GB + space for imported artifacts
Docker\*| Docker CE 18.06.1 | Docker Desktop 2.3.0.3|Docker CE 18.06.1
Browser\*| Google Chrome\* 88  | Google Chrome\* 88 |Google Chrome\* 88

Learn more about the recommended prerequisites in the DL Workbench [documentation](@ref workbench_docs_Workbench_DG_Prerequisites.html).

## Run DL Workbench 

You can [run DL Workbench](@ref workbench_docs_Workbench_DG_Install) on your local system or in the Intel® DevCloud for the Edge. Another option: running DL Workbench from the [Intel® Distribution for OpenVINO™ Toolkit Package](@ref workbench_docs_Workbench_DG_Install_from_Package) is available for Linux systems only.

Run DL Workbench on your local system by using simple [installation form](@ref workbench_docs_Workbench_DG_Run_Locally), where you can select your options and run the commands on the local machine.

Watch the video to learn how to run DL Workbench:

\htmlonly
<iframe width="560" height="315" src="https://www.youtube.com/embed/JBDG2g5hsoM"  frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
\endhtmlonly

For more details on installation settings and security options, visit [Advanced DL Workbench Configurations](Advanced_Config.md) page. 

## <a name="workflow-overview"></a> DL Workbench Workflow 

The simplified OpenVINO™ DL Workbench workflow is:
1. **Import a model** for your use case. 
2. **Create a project** to get the most out of DL Workbench functionality: evaluate model performance and accuracy, optimize the model and prepare it for deployment. 
3. **Optimize the model** to accelerate your model performance.

![](./dl_workbench_img/diagram.jpg)

## Get Started

This section illustrates a sample use case of how to create a project with a pre-trained model from the [Intel® Open Model Zoo](@ref omz_models_group_intel) and not annotated dataset on CPU device. You can watch the demo video with detailed workflow to [Get Started](@ref Workbench_DG_Work_with_Models_and_Sample_Datasets) with DL Workbench. 

For detailed instructions to create a new project, visit the links below: 
* [DL Workbench Get Started](@ref Workbench_DG_Work_with_Models_and_Sample_Datasets) 
* [Select a model](@ref workbench_docs_Workbench_DG_Select_Model)
* [Create project](@ref workbench_docs_Workbench_DG_Create_Project)


Once you open DL Workbench, create a project, which is a combination of a model, dataset, target machine, and device. 

Follow the steps below:

### Step 1. Open a New Project 

On the the **Active Projects** page, click **Create** to open the **Create Project** page:
![](./dl_workbench_img/active_projects_page.png)

### Step 2. Import a Model

Click **Import** next to the **Model** table on the **Create Project** page. The **Import Model** page opens. Select the ssd_mobilenet_v2_coco model from the Open Model Zoo and click **Import**.
![](./dl_workbench_img/import_model_mobilenet.png)

### Step 3. Convert the Model into Intermediate Representation

The **Convert Model to IR** tab opens. To work with DL Workbench, you need to obtain a model in Intermediate Representation (IR) format. Select the FP16 precision and click **Convert**.
![](./dl_workbench_img/convert_model_mobilenet.png)

Once the model is successfully imported, you will be redirected **Create Project** page.
![](./dl_workbench_img/model_imported_mobilenet.png)

### Step 4. Select a Dataset

Scroll down to the **Validation Dataset** table and click **Import**.
![](./dl_workbench_img/validation_dataset_import.png)

You will see the **Create Dataset** page where you can add your own images. Click **Import**.
![](./dl_workbench_img/custom_dataset_creation.png)

You are directed back to the **Create Project** page where you can see the status of the dataset.
![](./dl_workbench_img/dataset_imported.png)

### Step 5. Create the Project and Measure Model Performance

On the **Create Project** page, select the imported model, CPU target, and the dataset. Click **Create**.
![](./dl_workbench_img/create_project_selected.png)

Once the inference stage is complete, the **Projects** page opens automatically. 
![](./dl_workbench_img/project_created_mobilenet.png)

Congratulations, you have created your first project in the OpenVINO DL Workbench. From there you can successfully proceed to explore different DL Workbench features:
* [Tune Model for Enhanced Performance](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Int_8_Quantization.html)
* [Select the inference](@ref workbench_docs_Workbench_DG_Run_Single_Inference) 
* [Visualize model](@ref workbench_docs_Workbench_DG_Visualize_Model)
* [Experiment with model optimization](@ref workbench_docs_Workbench_DG_Int_8_Quantization)
and inference options to profile the configuration

## Additional Resources

* [DL Workbench Overview](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Introduction.html)
* [DL Workbench Installation](@ref workbench_docs_Workbench_DG_Install)
* [DL Workbench Get Started](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Work_with_Models_and_Sample_Datasets.html)
* [DL Workbench User Guide](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Run_Single_Inference.html)
* [DL Workbench Educational Resources](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Additional_Resources.html)
* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [OpenVINO™ Toolkit Overview](../index.md)



