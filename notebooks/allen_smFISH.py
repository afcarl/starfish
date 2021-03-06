#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
# # Reproduce Allen smFISH results with Starfish
# 
# This notebook walks through a work flow that reproduces the smFISH result for one field of view using the starfish package. 
# EPY: END markdown

# EPY: START code
from copy import deepcopy
from glob import glob
import json
import os

# EPY: ESCAPE %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util, img_as_float)

from starfish.io import Stack
from starfish.constants import Indices
# EPY: END code

# EPY: START code
# # developer note: for rapid iteration, it may be better to run this cell, download the data once, and load 
# # the data from the local disk. If so, uncomment this cell and run this instead of the above. 
# !aws s3 sync s3://czi.starfish.data.public/20180606/allen_smFISH ./allen_smFISH
# experiment_json = os.path.abspath("./allen_smFISH/fov_001/experiment.json")
# EPY: END code

# EPY: START code
# this is a large (1.1GB) FOV, so the download may take some time
experiment_json = 'https://dmf0bdeheu4zf.cloudfront.net/20180606/allen_smFISH/fov_001/experiment.json'
# EPY: END code

# EPY: START markdown
# Load the Stack object, which while not well-named right now, should be thought of as an access point to an "ImageDataSet". In practice, we expect the Stack object or something similar to it to be an access point for _multiple_ fields of view. In practice, the thing we talk about as a "TileSet" is the `Stack.image` object. The data are currently stored in-memory in a `numpy.ndarray`, and that is where most of our operations are done. 
# 
# The numpy array can be accessed through Stack.image.numpy\_array (public method, read only) or Stack.image.\_data (read and write)
# EPY: END markdown

# EPY: START code
codebook = pd.read_json('https://dmf0bdeheu4zf.cloudfront.net/20180606/allen_smFISH/fov_001/codebook.json')
codebook
# EPY: END code

# EPY: START markdown
# We're ready now to load the experiment into starfish (This experiment is big, it takes a few minutes):
# EPY: END markdown

# EPY: START code
s = Stack()
s.read(experiment_json)
# EPY: END code

# EPY: START markdown
# All of our implemented operations leverage the `Stack.image.apply` method to apply a single function over each of the tiles or volumes in the FOV, depending on whether the method accepts a 2d or 3d array. Below, we're clipping each image independently at the 10th percentile. I've placed the imports next to the methods so that you can easily locate the code, should you want to look under the hood and understand what parameters have been chosen. 
# 
# The verbose flag for our apply loops could use a bit more refinement. We should be able to tell it how many images it needs to process from looking at the image stack, but for now it's dumb so just reports the number of tiles or volumes it's processed. This FOV has 102 images over 3 volumes. 
# EPY: END markdown

# EPY: START code
from starfish.pipeline.filter import Filter
s_clip = Filter.Clip(p_min=10, p_max=100, verbose=True)
s_clip.filter(s)
# EPY: END code

# EPY: START markdown
# We're still working through the backing of the Stack.image object with the on-disk or on-cloud Tile spec. As a result, most of our methods work in-place. For now, we can hack around this by deepcopying the data before administering the operation. I'm doing this on a workstation, so be aware of the memory usage!
# EPY: END markdown

# EPY: START code
# filtered_backup = deepcopy(s)
# EPY: END code

# EPY: START markdown
# If you ever want to visualize the image in the notebook, we've added a widget to do that. The first parameter is an indices dict that specifies which hybridization round, channel, z-slice you want to view. The result is a pageable visualization across that arbitrary set of slices. Below I'm visualizing the first channel, which your codebook tells me is Nmnt. 
# 
# [N.B. once you click on the slider, you can page with the arrow keys on the keyboard.]
# EPY: END markdown

# EPY: START code
s.image.show_stack({Indices.CH: 0});
# EPY: END code

# EPY: START code
s_bandpass = Filter.Bandpass(lshort=0.5, llong=7, threshold=None, truncate=4, verbose=True)
s_bandpass.filter(s)
# EPY: END code

# EPY: START markdown
# For bandpass, there's a point where things get weird, at `c == 0; z <= 14`. In that range the images look mostly like noise. However, _above_ that, they look great + background subtracted! The later stages of the pipeline appear robust to this, though, as no spots are called for the noisy sections. 
# EPY: END markdown

# EPY: START code
# I wasn't sure if this clipping was supposed to be by volume or tile. I've done tile here, but it can be easily
# switched to volume. 
s_clip = Filter.Clip(p_min=10, p_max=100, is_volume=False, verbose=True)
s_clip.filter(s)
# EPY: END code

# EPY: START code
sigma=(1, 0, 0)  # filter only in z, do nothing in x, y
glp = Filter.GaussianLowPass(sigma=sigma, is_volume=True, verbose=True)
glp.filter(s)
# EPY: END code

# EPY: START markdown
# Below, because spot finding is so slow when single-plex, we'll pilot this on a max projection to show that the parameters work. Here's what trackpy.locate, which we wrap, produces for a z-projection of channel 1. To do use our plotting methods on z-projections we have to expose some of the starfish internals, which will be improved upon. 
# EPY: END markdown

# EPY: START code
from showit import image
from trackpy import locate

# grab a section from the tensor. 
ch1 = s.image.max_proj(Indices.Z)[0, 1]

results = locate(ch1, diameter=3, minmass=250, maxsize=3, separation=5, preprocess=False, percentile=10) 
results.columns = ['y', 'x', 'intensity', 'r', 'eccentricity', 'signal', 'raw_mass', 'ep']
# EPY: END code

# EPY: START code
# plot the z-projection
image(ch1, size=20, clim=(15, 52))

# draw called spots on top as red circles
# scale radius plots the red circle at scale_radius * spot radius
s.image._show_spots(results, ax=plt.gca(), scale_radius=7)
# EPY: END code

# EPY: START markdown
# Below spot finding is on the _volumes_ for each channel. This will take about `11m30s`
# EPY: END markdown

# EPY: START code
from starfish.pipeline.features.spots.detector import SpotFinder

# I've guessed at these parameters from the allen_smFISH code, but you might want to tweak these a bit. 
# as you can see, this function takes a while. It will be great to parallelize this. That's also coming, 
# although we haven't figured out where it fits in the priority list. 
kwargs = dict(
    spot_diameter=3, # must be odd integer
    min_mass=300,
    max_size=3,  # this is max _radius_
    separation=5,
    noise_size=0.65,  # this is not used because preprocess is False
    preprocess=False,
    percentile=10,  # this is irrelevant when min_mass, spot_diameter, and max_size are set properly
    verbose=True,
    is_volume=True,
)
lmpf = SpotFinder.LocalMaxPeakFinder(**kwargs)
spot_attributes = lmpf.find(s)
# EPY: END code

# EPY: START code
# save the results to disk as json
for ch, attrs in enumerate(spot_attributes):
    attrs.save(f'spot_attributes_c{ch}.json')
# EPY: END code

# EPY: START code
# # if you want to load them back in the same shape, here's how:
# from starfish.pipeline.features.spot_attributes import SpotAttributes
# spot_attributes = [SpotAttributes.load(attrs) for attrs in glob('spot_attributes_c*.json')]
# EPY: END code

# EPY: START code
# this is not a very performant function because of how matplotlib renders circles as individual artists, 
# but I think it's useful for debugging the spot detection.

# Note that in places where spots are "missed" it is often because they've been localized to nearby z-planes

s.image.show_stack({Indices.CH: 0}, show_spots=spot_attributes[0], figure_size=(20, 20), p_min=60, p_max=99.9);
# EPY: END code
