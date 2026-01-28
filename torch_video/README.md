## Prerequisite

See [PytorchVideo/Install.md](https://github.com/facebookresearch/pytorchvideo/blob/main/INSTALL.md)

- `conda create -n pytorchvideo -c conda-forge python=3.10` # pytorch does not support python version 3.8
- `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
- `pip install -U fvcore`
- `pip install -U iopath`
- `pip install pytorchvideo`
- `pip install opencv-python` # cv2
- `conda install "ffmpeg" -c conda-forge` # for torchcodec
- `pip install torchcodec matplotlib`
- `conda install av -c conda-forge` #installing using `pip install av` leading to "ModuleNotFoundError: No module named 'distutils.msvccompiler' " [see](https://github.com/PyAV-Org/PyAV)
- `pip install --upgrade Pillow` # if get "ImportError: DLL load failed while importing _imaging:", run this command [see](https://stackoverflow.com/questions/66385979/dll-load-failed-while-importing-imaging)
- `pip install requests` # for getting videos from URL
- `pip install fsspec aiohttp` # if using file-like object to speed up video read and seek [see](https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/file_like.html)
- `pip install joblib` # for multiprocessing and multithreading capabilities [see](https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/parallel_decoding.html)


## Download Kinetics Dataset

[see](https://mmaction2.readthedocs.io/en/stable/dataset_zoo/kinetics.html)
```
pip install -U openmim
pip install -U openxlab
```

## Dataset

UCF101 (Top recommendation)
HMDB51 (Even smaller, very clean)