# Volume-Renderer-From-Scratch-With-GUI
In this project, differentiable volume render has been implemented with a graphical user interface (GUI). The implementation is in Python and uses PyTorch.


https://github.com/user-attachments/assets/67681aaa-cb39-4ee7-a1d0-d12edc62656b


## How to run the code
Download or clone this repo.
After installing Anaconda, you can find all the requirements in the environment.yml file. You can create a new environment with all the requirements using,
```
conda env create -f environment.yml
```
If you want to use the default volume, then
```
cd volumes/
unzip coronacases_org_001.zip
cd ../
```
**Note:** This volume is from publicly available [Kaggle COVID-19 CT scans](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans?select=ct_scans).

After activating the environment, you can run the default script using Python.
```
cd script/
python main.py ../volumes/coronacases_org_001.nii -s 512 -f 3.5
```
**Note:** The resolution (-s 512) and focal coefficient (-f 3.5) were set based on our available GPU memory. You can change these settings depending on the GPU memory you have available or you can change the device to "CPU" in the code to run the code on CPU instead of GPU.

## System Specification
The code was tested on:

* OS: Ubuntu 22.04.4 LTS
* GPU: NVIDIA RTX A6000 with 48 GB Memory
* CPU: AMD Ryzen 9 7950X
* RAM: 128 GB System Memory
