import random
from itertools import product
from typing import Tuple

import numpy as np
import pytest
from numpy import zeros, array
from numpy.random import rand, normal, poisson
from pandas import DataFrame, concat
from skimage.filters import gaussian
from slicedimage import Tile, TileSet

from starfish.constants import Indices, Coordinates
from starfish.image import ImageStack
from starfish.io import Stack

# TODO ambrosejcarr: all fixtures should emit a stack and a codebook, so they can be better interchanged

@pytest.fixture(scope='session')
def merfish_stack() -> Stack:
    """retrieve MERFISH testing data from cloudfront and expose it at the module level

    Notes
    -----
    Because download takes time, this fixture runs once per session -- that is, the download is run only once.
    Therefore, methods consuming this fixture should COPY the data using deepcopy before executing code that changes
    the data, as otherwise this can affect other tests and cause failure cascades.

    Returns
    -------
    Stack :
        starfish.io.Stack object containing MERFISH data
    """
    s = Stack()
    s.read('https://s3.amazonaws.com/czi.starfish.data.public/test/MERFISH/fov_001/experiment.json')
    return s


@pytest.fixture(scope='session')
def synthetic_stack() -> ImageStack:
    """generate a synthetic ImageStack

    Returns
    -------
    ImageStack :
        imagestack containing a tensor of (2, 3, 4, 30, 20) whose values are all 1.

    """
    NUM_HYB = 2
    NUM_CH = 3
    NUM_Z = 4
    Y = 30
    X = 20

    img = TileSet(
        {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
        {
            Indices.HYB: NUM_HYB,
            Indices.CH: NUM_CH,
            Indices.Z: NUM_Z,
        },
        default_tile_shape=(Y, X),
    )
    for hyb in range(NUM_HYB):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
                tile = Tile(
                    {
                        Coordinates.X: (0.0, 0.001),
                        Coordinates.Y: (0.0, 0.001),
                        Coordinates.Z: (0.0, 0.001),
                    },
                    {
                        Indices.HYB: hyb,
                        Indices.CH: ch,
                        Indices.Z: z,
                    }
                )
                tile.numpy_array = np.ones(
                    (Y, X))

                img.add_tile(tile)

    stack = ImageStack(img)
    return stack


@pytest.fixture(scope='session')
def gold_standard_dataset() -> Tuple[Stack, list]:

    # set random seed so that data is consistent across tests
    random.seed(1)
    np.random.seed(1)

    NUM_HYB = 4
    NUM_CH = 2
    NUM_Z = 1
    Y = 100
    X = 100

    assert X == Y  # for compatibility with the parameterization of the code

    def choose(n, k):
        if n == k:
            return [[1]*k]
        subsets = [[0] + a for a in choose(n-1, k)]
        if k > 0:
            subsets += [[1] + a for a in choose(n-1, k-1)]
        return subsets

    def graham_sloane_codes(n):
        # n is length of codeword
        # number of on bits is 4
        def code_sum(codeword):
            return sum([i * c for i, c in enumerate(codeword)]) % n
        return [c for c in choose(n, 4) if code_sum(c) == 0]

    # set the image parameters
    p = {
        # number of on bits (not used with current codebook)
        'N_high': 4,
        # length of barcode
        'N_barcode': NUM_CH * NUM_HYB,
        # mean number of flourophores per transcripts - depends on amplification strategy (e.g HCR, bDNA)
        'N_flour': 200,
        # mean number of photons per flourophore - depends on exposure time, bleaching rate of dye
        'N_photons_per_flour': 50,
        # mean number of background photons per pixel - depends on tissue clearing and autoflourescence
        'N_photon_background': 1000,
        # quantum efficiency of the camera detector units number of electrons per photon
        'detection_efficiency': .25,
        # camera read noise per pixel in units electrons
        'N_background_electrons': 1,
        # number of RNA puncta; keep this low to reduce overlap probability
        'N_spots': 20,
        # height and width of image in pixel units
        'N_size': X,
        # standard devitation of gaussian in pixel units
        'psf': 2,
        # dynamic range of camera sensor 37,000 assuming a 16-bit AD converter
        'graylevel': 37000.0/2**16,
        # 16-bit AD converter
        'bits': 16
    }

    codebook = graham_sloane_codes(p['N_barcode'])

    def generate_spot(p):
        position = rand(2)
        gene = random.choice(range(len(codebook)))
        barcode = array(codebook[gene])
        photons = [poisson(p['N_photons_per_flour'])*poisson(p['N_flour'])*b for b in barcode]
        return DataFrame({'position': [position], 'barcode': [barcode], 'photons': [photons], 'gene': gene})

    # right now there is no jitter on x-y positions of the spots, we might want to make it a vector
    spots = concat([generate_spot(p) for _ in range(p['N_spots'])])

    image = zeros((p['N_barcode'], p['N_size'], p['N_size'],))

    for s in spots.itertuples():
        image[:, int(p['N_size']*s.position[0]), int(p['N_size']*s.position[1])] = s.photons

    image_with_background = image + poisson(p['N_photon_background'], size=image.shape)
    filtered = array([gaussian(im, p['psf']) for im in image_with_background])
    filtered = filtered*p['detection_efficiency'] + normal(scale=p['N_background_electrons'], size=filtered.shape)
    signal = [(x/p['graylevel']).astype(int).clip(0, 2**p['bits']) for x in filtered]

    # set up the tile set
    image_data = TileSet(
        {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
        {
            Indices.HYB: NUM_HYB,
            Indices.CH: NUM_CH,
            Indices.Z: NUM_Z,
        },
        default_tile_shape=(Y, X),
    )

    # fill the TileSet
    experiment_indices = list(product(range(NUM_HYB), range(NUM_CH), range(NUM_Z)))
    for i, (hyb, ch, z) in enumerate(experiment_indices):

        tile = Tile(
            {
                Coordinates.X: (0.0, 0.001),
                Coordinates.Y: (0.0, 0.001),
                Coordinates.Z: (0.0, 0.001),
            },
            {
                Indices.HYB: hyb,
                Indices.CH: ch,
                Indices.Z: z,
            }
        )
        tile.numpy_array = signal[i]

        image_data.add_tile(tile)

    data_stack = ImageStack(image_data)

    # make a max projection and pretend that's the dots image, which we'll create another ImageStack for this
    dots_data = TileSet(
        {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
        {
            Indices.HYB: 1,
            Indices.CH: 1,
            Indices.Z: 1,
        },
        default_tile_shape=(Y, X),
    )
    tile = Tile(
        {
            Coordinates.X: (0.0, 0.001),
            Coordinates.Y: (0.0, 0.001),
            Coordinates.Z: (0.0, 0.001),
        },
        {
            Indices.HYB: 0,
            Indices.CH: 0,
            Indices.Z: 0,
        }
    )

    signal_array = np.array(signal)
    tile.numpy_array = np.max(signal_array, 0)

    dots_data.add_tile(tile)
    dots_stack = ImageStack(dots_data)

    # TODO can we mock up a nuclei image somehow?

    # put the data together into a top-level Stack
    results = Stack.from_data(data_stack, aux_dict={'dots': dots_stack})

    # make the codebook(s)
    codebook = []
    for _, code_record in spots.iterrows():
        codeword = []
        for code_value, (hyb, ch, z) in zip(code_record['barcode'], experiment_indices):
            if code_value == 1:
                codeword.append({
                    Indices.HYB: hyb,
                    Indices.CH: ch,
                    Indices.Z: z,
                    "v": 1
                })
        codebook.append(
            {
                'codeword': codeword,
                'gene_name': code_record['gene']
            }
        )

    return results, codebook
