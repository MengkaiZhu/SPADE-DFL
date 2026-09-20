# Third-party sources and rights status

No code reuse license is specified for SPADE-DFL. The release package contains no project `LICENSE`, `LICENCE`, or `COPYING` file, and `pyproject.toml` has no license declaration. Dependency licenses do not license this project. No MIT, Apache, or other project license has been inferred.

The datasets, prepared arrays, MRI images, feature caches, pretrained weights, and dependency environments are not included. Their terms are separate from the status of the project code. The paper cites these dataset pages:

- MNIST: [hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- Fashion-MNIST: [zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
- Alzheimer disease: [rabieelkharoua/alzheimers-disease-dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset).
- Brain Tumor MRI: [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

The code also records TensorFlow/Keras MNIST hosting and Zalando Research Fashion-MNIST hosting. These references identify historical acquisition sources; they do not establish the exact dataset versions or redistribution terms. See [data preparation](data/README.md).

The MRI workflow specifies torchvision's ResNet-18 `IMAGENET1K_V1` weights and transforms. NumPy, SciPy, Matplotlib, mpmath, Pillow, pytest, PyTorch, and torchvision are installed separately and remain subject to their own terms. Software attribution and research citations are not grants of reuse rights.

This notice records the evidence currently available. A rights holder should establish the project license and any required third-party notices before public open-source distribution.
