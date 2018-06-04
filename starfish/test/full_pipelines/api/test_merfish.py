
from starfish.constants import Indices
from starfish.test.dataset_fixtures import merfish_stack
from starfish.pipeline.features.pixels.pixel_spot_detector import PixelSpotDetector
from starfish.pipeline.filter.gaussian_high_pass import GaussianHighPass
from starfish.pipeline.filter.gaussian_low_pass import GaussianLowPass
from starfish.pipeline.filter.richardson_lucy_deconvolution import DeconvolvePSF


def test_merfish_pipeline(merfish_stack):
    s = merfish_stack

    # high pass filter
    ghp = GaussianHighPass(sigma=3)
    ghp.filter(s)

    # deconvolve the point spread function
    dpsf = DeconvolvePSF(num_iter=15, sigma=2)
    dpsf.filter(s)

    # low pass filter
    glp = GaussianLowPass(sigma=1)
    glp.filter(s)

    # scale the data by the scale factors
    scale_factors = {(t[Indices.HYB], t[Indices.CH]): t['scale_factor'] for index, t in s.tile_metadata.iterrows()}
    for indices in s.image._iter_indices():
        data = s.image.get_slice(indices)[0]
        scaled = data / scale_factors[indices[Indices.HYB], indices[Indices.CH]]
        s.image.set_slice(indices, scaled)

    # detect and decode spots
    psd = PixelSpotDetector(
        codebook='https://s3.amazonaws.com/czi.starfish.data.public/MERFISH/codebook.csv',
        distance_threshold=0.5176,
        magnitude_threshold=1,
        area_threshold=2,
        crop_size=40
    )

    spot_attributes, decoded = psd.find(s)

    # import seaborn as sns
    # sns.set_context('talk')
    # sns.set_style('ticks')
    #
    # bench = pd.read_csv(os.path.join('MERFISH', 'benchmark_results.csv'), dtype={'barcode': object})
    # x_cnts = res.groupby('gene').count()['area']
    # y_cnts = bench.groupby('gene').count()['area']
    # tmp = pd.concat([x_cnts, y_cnts], axis=1, join='inner').values
    # r = np.corrcoef(tmp[:, 1], tmp[:, 0])[0, 1]
    #
    # x = np.linspace(50, 2000)
    # plt.scatter(tmp[:, 1], tmp[:, 0], 50, zorder=2)
    # plt.plot(x, x, '-k', zorder=1)
    #
    # plt.xlabel('Gene copy number Benchmark')
    # plt.ylabel('Gene copy number Starfish')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('r = {}'.format(r))
    #
    # sns.despine(offset=2)
