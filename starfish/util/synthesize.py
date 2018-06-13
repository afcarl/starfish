import random
from itertools import product
from typing import List, Dict, Optional

import numpy as np
from numpy import zeros, array
from numpy.random import rand, normal, poisson
from pandas import DataFrame, concat
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from slicedimage import Tile, TileSet

from starfish.constants import Indices, Coordinates
from starfish.pipeline.features.codebook import Codebook
from starfish.image import ImageStack
from starfish.io import Stack


def graham_sloane_codes(n, bits_on: int=4):

    def choose(n, k):
        if n == k:
            return [[1] * k]
        subsets = [[0] + a for a in choose(n - 1, k)]
        if k > 0:
            subsets += [[1] + a for a in choose(n - 1, k - 1)]
        return subsets

    # n is length of codeword
    # number of on bits is 4
    def code_sum(codeword):
        return sum([i * c for i, c in enumerate(codeword)]) % n
    return [c for c in choose(n, bits_on) if code_sum(c) == 0]


def one_hot_code(n_hyb: int, n_channel: int, n_codes: int, gene_names: Optional[List[str]]=None) -> List[Dict]:
    """Generate codes where one channel is "on" in each hybridization round

    Parameters
    ----------
    n_hyb : int
        number of hybridization rounds per code
    n_channel : int
        number of channels per code
    n_codes : int
        number of codes to generate
    gene_names : Optional[List[str]]
        if provided, names for genes in codebook

    Returns
    -------
    List[Dict] :
        list of codewords

    """
    codes = set()
    while len(codes) < n_codes:
        codes.add(tuple([np.random.randint(0, n_channel) for _ in np.arange(n_hyb)]))

    codewords = [
        [
            {Indices.HYB.value: h, Indices.CH.value: c, 'v': 1} for h, c in enumerate(code)
        ] for code in codes
    ]

    if gene_names is None:
        gene_names = np.arange(n_codes)
    assert n_codes == len(gene_names)

    codebook = [{"codeword": w, "gene_name": g} for w, g in zip(codewords, gene_names)]

    return codebook


# TODO sofroniewn: doc me!
def synthesize(
        num_hyb: int=4, num_ch: int=2, num_z: int=1, resolution: int=100,
    ):  # -> Tuple[Stack, list]:
    """

    Parameters
    ----------
    num_hyb
    num_ch
    num_z
    resolution

    Returns
    -------
    Stack :
        starfish Stack containing synthetic spots
    list :
        codebook matching the synthetic data

    """

    # set random seed so that data is consistent across tests
    random.seed(2)
    np.random.seed(2)

    p = {
        # number of on bits (not used with current codebook)
        'N_high': 4,
        # length of barcode
        'N_barcode': num_ch * num_hyb,
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
        'N_size': resolution,

        # standard devitation of gaussian in pixel units
        'psf': 2,
        # dynamic range of camera sensor 37,000 assuming a 16-bit AD converter
        'graylevel': 37000.0 / 2 ** 16,
        # 16-bit AD converter
        'bits': 16
    }

    codebook = graham_sloane_codes(p['N_barcode'])

    def generate_spot(p):
        position = rand(2)
        gene = random.choice(range(len(codebook)))
        barcode = array(codebook[gene])
        photons = [poisson(p['N_photons_per_flour']) * poisson(p['N_flour']) * b for b in barcode]
        return DataFrame({'position': [position], 'barcode': [barcode], 'photons': [photons], 'gene': gene})

    # right now there is no jitter on x-y positions of the spots, we might want to make it a vector
    spots = concat([generate_spot(p) for _ in range(p['N_spots'])])  # type: ignore

    image = zeros((p['N_barcode'], p['N_size'], p['N_size'],))

    for s in spots.itertuples():
        image[:, int(p['N_size'] * s.position[0]), int(p['N_size'] * s.position[1])] = s.photons

    image_with_background = image + poisson(p['N_photon_background'], size=image.shape)
    filtered = array([gaussian(im, p['psf']) for im in image_with_background])
    filtered = filtered * p['detection_efficiency'] + normal(scale=p['N_background_electrons'], size=filtered.shape)
    signal = np.array([(x / p['graylevel']).astype(int).clip(0, 2 ** p['bits']) for x in filtered])

    def select_uint_dtype(array):
        """choose appropriate dtype based on values of an array"""
        max_val = np.max(array)
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if max_val <= dtype(-1):
                return array.astype(dtype)
        raise ValueError('value exceeds dynamic range of largest numpy type')

    corrected_signal = select_uint_dtype(signal)
    rescaled_signal: np.ndarray = rescale_intensity(corrected_signal)

    # set up the tile set
    image_data = TileSet(
        {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
        {
            Indices.HYB: num_hyb,
            Indices.CH: num_ch,
            Indices.Z: num_z,
        },
        default_tile_shape=(resolution, resolution),
    )

    # fill the TileSet
    experiment_indices = list(product(range(num_hyb), range(num_ch), range(num_z)))
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
        tile.numpy_array = rescaled_signal[i]

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
        default_tile_shape=(resolution, resolution),
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

    tile.numpy_array = np.max(rescaled_signal, 0)

    dots_data.add_tile(tile)
    dots_stack = ImageStack(dots_data)

    # TODO can we mock up a nuclei image somehow?

    # put the data together into a top-level Stack
    results = Stack.from_data(data_stack, aux_dict={'dots': dots_stack})

    # make the codebook
    codebook = []
    for _, code_record in spots.iterrows():
        codeword = []
        for code_value, (hyb, ch, z) in zip(code_record['barcode'], experiment_indices):
            if code_value != 0:
                codeword.append({
                    Indices.HYB: hyb,
                    Indices.CH: ch,
                    Indices.Z: z,
                    "v": code_value
                })
        codebook.append(
            {
                'codeword': codeword,
                'gene_name': code_record['gene']
            }
        )
    codebook = Codebook.from_code_array(codebook, n_ch=num_ch, n_hyb=num_hyb)

    return results, codebook, spots
