## Edge device-based Real-time Implementation of CycleGAN for the Colorization of Infrared Video

This repository provides some files for real-time implementation of CycleGAN on Jetson Xavier NX with TensorRT.

* Datasets
* Pretrained model
* Source code for inference stage
* Results


### 0.Datasets

Download from [Google Drive](https://drive.google.com/file/d/1GN-mLgHciyRknhO1Awllno4dimGJ8K43/view?usp=sharing).

![Datasets-img](images/Datasets.jpg)

The composition and number of image pairs in datasets:

|DATASETS|TRAIN|TEST|TOTAL|
|--------|-----|----|-----|
|LIR2RGB  | 2880 | 1440 | 4320|
|RGBN2RGB | 3419 | 1709 | 5218|
|PIR2RGB  | 2027 | 337  | 2364|



### 1.Pretrained model

Download from [Google Drive](https://drive.google.com/file/d/12aKrFH1kMfHVoLTCFdDxpgDuGhypADwm/view?usp=sharing).

The pretrained model is only used for inference stage. Onnx and TensorRT models are provided. Please note that the TensorRT model can only be used on the Jetson Xavier NX hardware platform.

### 2.Source code for inference stage

The steps to use the code follow the following steps, and it would be helpful if you are familiar with TensorRT and Makefile.

* Download the source code in this repository, and then put it to Jetson Xavier NX.
* Modify the Makefile, which is in Jetson-Xavier-NX/fp16&fp32(int8)/Makefile. In line7 to line 9, you need to modify it to your path.
>Copy files from the system to the following folders, but of course, if you are familiar with TensorRT and Makefile, you are not bound to do so.

>cuda_home in line 7: some bin, lib, and include files of TensorRT, CUDA, and protobuf

>syslib in line 8: mainly about libstdc++.so and libz.so and other system lib files

>cpp_pkg in line 9: mainly some bin, lib, and include files of OpenCV
* Place the dataset and pre-trained model to the following path: Jetson-Xavier-NX/fp16&fp32(int8)/workspace/
* Execute the "make" command on the console in the path where the Makefile is located.
* Execute the executable file "pro" to see the result. The executable file is saved in the following path: Jetson-Xavier-NX/fp16&fp32(int8)/workspace/

Also, you can try Build Systems like CMake, which will be much easier than Makefile.

If you encounter any problems, please feel free to post them to GitHub issues column.

### 3.Results

Inference results are also available on [Google Drive](https://drive.google.com/file/d/1-18u5aw2AD4kjFe6Qt-2I1CtIsM_orTV/view?usp=sharing).

