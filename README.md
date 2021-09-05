# Pytorch computer vision project template

### DEPRECATED

Boilerplate for large-scale semantic segmentation project, written before pytorch-lightning and Hydra were a thing, using Python 3.6.

Completely obsolete now, do not use.

### Features
- Dynamic config (options loaded depending on module).
- Pretty print, write, and read config to/from external file.
- Call-by-name modular elements (i.e. liberal use of `importlib`).
- Control over print/save/validate frequency.
- Train/resume/fine-tune functionality with tracking of separate experimental runs.
- Scalability across CPU > multiple GPUs (with external library support).
    - Batch normalization synced across GPUs.
- Full suite of image pre-processing on load.
- Dataset shuffling, cross-validation, etc.
- Bare-minimum implementation of various base classes to facilitate inheritance.

### Points of entry

**train.py:** Main point of entry. Train a neural network.
Use `--help` to inspect possible options. Can use it with partial options, e.g.:
```
$ python train.py --dataset cityscapes --help
```
to see dynamically-loaded options.

**test.py:** Perform validation/inference on saved model. Not implemented due to deprecation.

### Modularity

 The various dataset, model, loss function, optimizer, config and trainer classes all inherit from their own `Base...()` classes, which contain basic functionality that applies to all abstractions of the same type. For example, `BaseOptions.py` contains all config parsing/saving/loading functions as well as general housekeeping config that is used both during training and inference, whereas `TrainOptions.py` contains trainng-specific config.