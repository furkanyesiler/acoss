"""
Purpose: To use sublevelset filtrations as an invariant to
parameterization for describing local tempo curves
"""
import matplotlib.pyplot as plt
import deepdish as dd
from ripser import ripser
from persim import plot_diagrams
from scipy import sparse
import scipy
import scipy.io as sio
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.ndimage.filters import gaussian_filter1d as gf1d
from statistics import *
from ..benchmark.utils.cross_recurrence import *
from matplotlib.ticker import FormatStrFormatter


def getLowerStarFiltration(x, infinitymax=True):
    """
    Do a sublevelset filtration
    """
    N = x.size
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    # Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]
    # dgm0 = dgm0[dgm0[:, 1]-dgm0[:, 0] > 1e-3, :]
    if infinitymax:
        dgm0[np.isinf(dgm0[:, 1]), 1] = np.max(x)
    return dgm0


def getPersistenceImage(dgm, plims, res, weightfn = lambda b, l: l, psigma = None):
    """
    Return a persistence image (Adams et al.)
    :param dgm: Nx2 array holding persistence diagram
    :param plims: An array [birthleft, birthright, lifebottom, lifetop] \
        limits of the actual grid will be rounded based on res
    :param res: Width of each pixel
    :param weightfn(b, l): A weight function as a function of birth time\
        and life time
    :param psigma: Standard deviation of each Gaussian.  By default\
        None, which indicates it should be res/2.0
    """
    # Convert to birth time/lifetime
    I = np.array(dgm)
    I[:, 1] = I[:, 1] - I[:, 0]
    
    # Create grid
    lims = np.array([np.floor(plims[0]/res), np.ceil(plims[1]/res), np.floor(plims[2]/res), np.ceil(plims[3]/res)])
    xr = np.arange(int(lims[0]), int(lims[1])+2)*res
    yr = np.arange(int(lims[2]), int(lims[3])+2)*res
    sigma = res/2.0
    if psigma:
        sigma = psigma        
            
    # Add each integrated Gaussian
    PI = np.zeros((len(yr)-1, len(xr)-1))
    for i in range(I.shape[0]):
        [x, y] = I[i, :]
        w = weightfn(x, y)
        if w == 0:
            continue
        # CDF of 2D isotropic Gaussian is separable
        xcdf = scipy.stats.norm.cdf((xr - x)/sigma)
        ycdf = scipy.stats.norm.cdf((yr - y)/sigma)
        X = ycdf[:, None]*xcdf[None, :]
        # Integral image
        PI += weightfn(x, y)*(X[1::, 1::] - X[0:-1, 1::] - X[1::, 0:-1] + X[0:-1, 0:-1])
    return {'PI':PI, 'xr':xr[0:-1], 'yr':yr[0:-1]}


def getOnsetMeans(px, win=20, sigma=1, truncate=4, edge = 10, do_plot=False):
    """
    Do a mollified Gaussian derivative followed by
    a moving average to get smoothed local tempo estimates
    """
    x = px[edge:-edge] # Truncate edges since they seem to be unreliable
    x = gf1d(x, sigma, truncate=truncate, order = 1, mode='reflect')
    x = x[truncate*sigma:-truncate*sigma]
    if do_plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(px)
        plt.subplot(212)
        plt.plot(x)
        plt.show()
    M = x.size-win+1
    X = np.zeros((M, win))
    for k in range(win):
        X[:, k] = x[k:k+M]
    ret = np.mean(X, 1)
    return ret/np.median(ret)


def getAllPersistenceImages():
    pairs, paths = get_cover_pairs(lambda res: res['madmom_features']['onsets'])
    # Assume time series can be in the range [0, 2]
    pilims =  [0.5, 1.5, 0, 1]
    pilimsneg =  [-1.5, -0.5, 0, 1]
    pires = 0.004
    psigma = 0.04
    Is1 = []
    Is2 = []
    for i, pair in enumerate(pairs.keys()):
        print(i)
        for k in range(2):
            y = getOnsetMeans(pairs[pair][k])
            IUp = getLowerStarFiltration(y)
            PIUp = getPersistenceImage(IUp, pilims, pires, psigma=psigma)['PI'].flatten()
            IDown = getLowerStarFiltration(-y)
            PIDown = getPersistenceImage(IUp, pilimsneg, pires, psigma=psigma)['PI'].flatten()
            I = np.concatenate((PIUp, PIDown))
            if k == 0:
                Is1.append(I)
            else:
                Is2.append(I)
    Is1 = np.array(Is1)
    Is2 = np.array(Is2)
    D = get_csm(Is1, Is2)
    N = D.shape[0]
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    dcover = np.diag(D)
    dfalse = D[np.abs(I-J) > 0]
    sio.savemat("onsettiming_%.3g"%psigma, {"dcover":dcover, "dfalse":dfalse})

    bins = np.linspace(0, np.quantile(dfalse, 0.98), 40)
    cutoff = np.quantile(dfalse, 0.95)
    plt.figure(figsize=(5, 2.5))
    sns.distplot(dcover, kde=True, norm_hist=True, bins=bins)
    sns.distplot(dfalse, kde=True, norm_hist=True, bins=bins)
    plt.xlim([0, cutoff])
    plt.xlabel("Persitence Image Distance")
    plt.ylabel("Density")
    plt.title("Persistence Image Distances")
    plt.legend(["True Covers", "False Covers"])
    plt.savefig("OnsetTimings_%.3g.svg"%psigma, bbox_inches='tight')
    print(np.mean(dcover))
    print(np.mean(dfalse))
    print(ks_2samp(dcover, dfalse))


def getAllSTDevs():
    """
    pairs, paths = get_cover_pairs(lambda res: res['madmom_features']['onsets'])
    # Assume time series can be in the range [0, 2]
    N = len(pairs)
    stdevs = np.zeros((N, 2))
    for i, pair in enumerate(pairs.keys()):
        print(i)
        for k in range(2):
            y = getOnsetMeans(pairs[pair][k])
            stdevs[i, k] = np.std(y)
    sio.savemat("stdevs.mat", {"stdevs":stdevs})
    """
    stdevs = sio.loadmat("stdevs.mat")["stdevs"]
    N = stdevs.shape[0]
    y1 = stdevs[:, 0]
    y2 = stdevs[:, 1]
    D = np.abs(y1[:, None] - y2[None, :])
    dcover = np.diag(D)
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    dfalse = D[np.abs(I-J) > 0]

    bins = np.linspace(0, 0.15, 30)
    plt.figure(figsize=(5, 2.5))
    sns.distplot(dcover, kde=True, norm_hist=True, bins=bins)
    sns.distplot(dfalse, kde=True, norm_hist=True, bins=bins)
    print(np.mean(dcover))
    print(np.mean(dfalse))
    print(ks_2samp(dcover, dfalse))
    plt.xlim([np.min(bins), np.max(bins)])
    plt.show()




def makeFigure():
    fin = open('pairs.txt')
    pairs = [f.split() for f in fin.readlines()]
    fin.close()
    pairidx = 1397
    paths = pairs[pairidx]

    get_onsets = lambda res: res['madmom_features']['onsets']
    y1 = get_onsets(dd.io.load(paths[0]))
    y2 = get_onsets(dd.io.load(paths[1]))

    # Assume time series can be in the range [0, 2]
    pilims =  [0.8, 1.2, 0, 0.4]
    pilimsneg =  [-1.2, -0.8, 0, 0.4]
    pires = 0.004
    psigma = 0.03

    y1 = getOnsetMeans(y1)
    y2 = getOnsetMeans(y2)
    prec = 1000
    infinitymax = True
    cutoff = 0.03

    I1 = getLowerStarFiltration(y1, infinitymax)
    res = getPersistenceImage(I1, pilims, pires, psigma=psigma)
    img1 = res['PI']
    dgm0 = I1[np.argsort(I1[:, 0]-I1[:, 1]), :]
    dgm0 = dgm0[0:4, :]
    grid1 = np.unique(dgm0.flatten())
    grid1 = grid1[np.isfinite(grid1)]

    I2 = getLowerStarFiltration(y2, infinitymax)
    res = getPersistenceImage(I2, pilims, pires, psigma=psigma)
    img2 = res['PI']
    dgm0 = I2[np.argsort(I2[:, 0]-I2[:, 1]), :]
    dgm0 = dgm0[0:4, :]
    grid2 = np.unique(dgm0.flatten())
    grid2 = grid2[np.isfinite(grid2)]

    vmin = min(np.min(img1), np.min(img2))
    vmax = max(np.max(img1), np.max(img2))

    grid = np.concatenate((grid1, grid2))
    ylims = None
    if len(grid) > 0:
        ylims = [0.98*np.min(grid), 1.02*np.max(grid)]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(221)
    ax.set_xticks([])
    ax.set_yticks(grid1)
    ax.set_ylim(ylims)
    plt.grid(linewidth=1, linestyle='--')
    ax.plot(y1)
    plt.xlabel("Time")
    plt.ylabel("Tempo Ratio")
    plt.title("'Joy Division' Timing")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax = plt.subplot(222)
    plot_diagrams(I1)
    ax.set_xticks(grid1)
    ax.set_yticks(grid1)
    plt.xlim(ylims)
    if ylims:
        plt.ylim(ylims)
    plt.grid(linewidth=1, linestyle='--')
    plt.title('Persistence Diagrams')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))



    ax = plt.subplot(223)
    ax.set_xticks([])
    ax.set_yticks(grid2)
    if ylims:
        ax.set_ylim(ylims)
    plt.grid(linewidth=1, linestyle='--')
    ax.plot(y2)
    plt.xlabel("Time")
    plt.ylabel("Tempo Ratio")
    plt.title("'Versus' Timing")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax = plt.subplot(224)
    plot_diagrams(I2)
    ax.set_xticks(grid2)
    ax.set_yticks(grid2)
    if ylims:
        plt.ylim(ylims)
    plt.grid(linewidth=1, linestyle='--')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig("OnsetTiming_%i.svg"%pairidx, bbox_inches='tight')

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.imshow(img1, vmin=vmin, vmax=vmax, extent = (res['xr'][0], res['xr'][-1], res['yr'][-1], res['yr'][0]), cmap = 'magma_r', interpolation = 'nearest')
    plt.gca().invert_yaxis()
    plt.title("'Joy Division' Persistence Image")
    plt.xlabel('Birth')
    plt.ylabel('Lifetime')

    plt.subplot(122)
    plt.imshow(img2, vmin=vmin, vmax=vmax, extent = (res['xr'][0], res['xr'][-1], res['yr'][-1], res['yr'][0]), cmap = 'magma_r', interpolation = 'nearest')
    plt.gca().invert_yaxis()
    plt.title("'Versus' Persistence Image")
    plt.xlabel('Birth')
    plt.ylabel('Lifetime')
    plt.savefig("OnsetTiming_%i_PI.svg"%pairidx, bbox_inches='tight')


if __name__ == '__main__':
    getAllSTDevs()
    #getAllPersistenceImages()
    #makeFigure()