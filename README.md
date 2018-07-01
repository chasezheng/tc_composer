# tc_composer
## Getting started
- In your conda environment, run `conda install pytorch torchvision cuda91 -c pytorch`
- Build and install TC from source: https://github.com/facebookresearch/TensorComprehensions/blob/master/docs/source/installation.rst
## Testing
In repo root, run
`python -m unittest discover -s tests -t tests -p \*.py`