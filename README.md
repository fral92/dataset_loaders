This repository contains a framework to load the most commonly used datasets
for image and video semantic segmentation. The framework can perform some
on-the-fly preprocessing/data augmentation, as well as run on multiple threads
(if enabled) to speed up the I/O operations.

**NEWS:** You might be interested in checking out 
[Main loop TF](https://github.com/fvisin/main_loop_tf), a python main loop that integrates the *Dataset loaders* with Tensorflow!

### Attribution
If you use this code, please cite:
* \[1\] Francesco Visin, Adriana Romero, (2016). *Dataset loaders: a python
  library to load and preprocess datasets* ([BibTeX](
https://gist.github.com/fvisin/7104500ae8b33c3b65798d5d2707ce6c#file-dataset_loaders-bib))

Check the full documentation on: http://dataset_loaders.readthedocs.io
### Optical flow
The dataset loaders can optionally load from disk, or in some cases compute,
the optical flow associated to the video sequences. To do so it looks for a
file in `<dataset_path>/OF/<OF_type>/prefix/filename.npy>` where prefix is the
name of the subset (or video) as returned by get_names(). If the file is
missing it will try to compute the optical flow for the entire dataset once and
store it on disk.

At the moment the only optical flow algorithm supported to this end is the
Farneback (requires openCV installed, choose 'Farn' as type), but you can
easily pre-compute the optical flow with your preferred algorithm and then load
it via the dataset loaders. An example code for a few algorithms is provided
[here](https://gist.github.com/marcociccone/593638e932a48df7cfd0afe71052ef1d).
NO SUPPORT WILL BE PROVIDED FOR THIS CODE OR ANY OTHER OPTICAL FLOW CODE NOT
DIRECTLY INTEGRATED IN THIS FRAMEWORK.

