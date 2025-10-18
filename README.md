## Install miniforge

Follow instructions in [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file)

## Jupyter environment
```
conda create -n nb39 -c conda-forge python=3.9
pip install notebook
conda install -c conda-forge nb_conda_kernels
```

## Computer vision
```
conda create -n op_cv -c conda-forge python=3.10
pip install ipykernel
pip install opencv-python
pip install matplotlib pandas seaborn
pip install scipy scikit-learn
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install wget
```