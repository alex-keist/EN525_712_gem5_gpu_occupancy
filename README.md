# EN525_712_gem5_gpu_occupancy

This is for analyzing the GPU occupancy with a simulated AMD ROCm GPU processor in Gem5
This repo does not have the resources for setting up the Gem5 environment. Info about this setup can be found here: https://github.com/gem5bootcamp/2024/blob/main/slides/04-GPU-model/gpu-slides.pdf

Here are the commands from the module 6 discussion instructions for running one of the HIP samples:

docker pull ghcr.io/gem5/gcn-gpu:v24-0
cd gem5-resources/src/gpu/hip-samples
docker run --rm -v ${PWD}:${PWD} -w ${PWD} ghcr.io/gem5/gcn-gpu:v24-0 make
cd /workspaces/intro-to-gem5-alex-keist
docker run --volume $(pwd):$(pwd) -w $(pwd) ghcr.io/gem5/gcn-gpu:v24-0 gem5/build/VEGA_X86/gem5.opt gem5/configs/example/apu_se.py -n 3 --gfx-version=gfx902 -c gem5-resources/src/gpu/hip-samples/bin/MatrixTranspose

Put the MatrixTranspose_configurable.cpp file in the hip-samples folder and then run the make command to have the binary included in the hip-samples/bin directory
Place the gpu occupancy analyzer script in a ~/analyzer_dir directory, or if placed in a different location, adjust the path in the run_matrix_sweep script to reflect the new path to the analyzer

This sweep script can be edited to support running other programs if you edit the other programs to accept in arguments. You can then edit the configs list in the sweep script to run over different parameter values

**** To Do
The shared memory usage and the register usages currently do not seem to reflect real usage. I have not been able to find a good way to calculate these values with the stats in the gem5 stat files. If a good way to get these values is found, this should be corrected in the analyzer script

The pie chart is probably not the best way to display the data