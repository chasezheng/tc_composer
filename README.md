# tc_composer
## Getting started
- In your conda environment, run `conda install pytorch torchvision cuda91 numba -c pytorch`
- Build and install TC from source: https://github.com/facebookresearch/TensorComprehensions/blob/master/docs/source/installation.rst
## Example
- How to write your own TC function: [notebook](examples/mlp3.ipynb).
- How to compose TC functions: [notebook](examples/compose.ipynb).
## Testing
In repo root, run
`python -m unittest discover -s tests -t tests -p \*.py`
