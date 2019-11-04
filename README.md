# A Generative approach to Audio Inpainting

# Installation guide

To run the notebooks and/or libraries in this repository, it is necessary to install some dependencies. The file conda_environment.yml lists these dependencies and can be directly used to create a Conda environment with them installed (see [HERE](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
for a tutorial on how to install Conda). Once Conda is available, run:
```
conda env create -f conda_environment.yml
conda activate generative-audio-inpainting
pip install ./libraries
```

These commands will first create an environment named "generative-audio-inpainting", then activate it and install in it the libraries in the repository ("mlp", "rml" and "models"). Once this is done, the environment should be ready to run the notebooks. In case you want to install the libraries in developer mode substitute the last line by pip install -e ./libraries

To remove the environment and all its installed libraries execute:
```
conda deactivate
conda remove -y --name generative-audio-inpainting --all
```

