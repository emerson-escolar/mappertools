import pathlib
import sklearn.manifold as skm
import scipy.spatial.distance as ssd
import mappertools.outputs.visualization as qs
import numpy as np

def do_mds_one_dim_analysis(data, data_name, dim=2, metric='correlation', output_folder = None, do_outputs=False):
    """
    Performs MDS embedding.

    A convenience wrapper to use sklearn.manifold.MDS with output images and residual analysis.
    Returns a sklearn.manifold.MDS object.

    Parameters
    ----------
    data : ndarray
           An m by n array of m original observations in an n-dimensional space.
    data_name : str
           A name for the data. Used to name output files.
    dim : int
           With what dimension to do MDS.
    metric : str or function, optional
        The distance metric to use. The distance function can
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
        (See scipy.spatial.distance.pdist)
    output_folder : pathlib.Path
    do_outputs : bool
    """

    dissimilarities = ssd.pdist(data, metric=metric)
    diss_matrix = ssd.squareform(dissimilarities)

    mds = skm.MDS(dissimilarity='precomputed', n_components=dim, max_iter=1000).fit(diss_matrix)

    if output_folder == None:
        output_folder = pathlib.Path.cwd()

    if do_outputs and dim in (2,3):
        qs.qs_plot(mds.embedding_,s=1)
        write_file = output_folder.joinpath(data_name + "_mds" + str(dim) + "D.png")
        qs.plt.savefig(str(write_file), dpi=300)
        qs.plt.close()

    mds_distances = ssd.pdist(mds.embedding_, metric='euclidean')

    if do_outputs:
        fig=qs.plt.figure(figsize=(6,6),dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(dissimilarities, mds_distances,s=1)
        ax.set_aspect('equal', 'datalim')
        qs.plt.xlabel('dissimilarities')
        qs.plt.ylabel('mds euclidean distances')
        qs.plt.grid(True)
        qs.plt.title(data_name + " " + str(dim) + "D" + " mds residuals")

        x_vals = np.array(ax.get_xlim())
        qs.plt.plot(x_vals, x_vals, 'r-')

        write_file = output_folder.joinpath(data_name + "_mds_residuals" + str(dim) + "D.png")
        qs.plt.savefig(str(write_file), dpi=1000)
        qs.plt.close()

    return mds


def do_mds_analysis(data, data_name, metric='correlation', output_folder = None, max_dim=9, do_outputs=False):
    dissimilarities = ssd.pdist(data, metric=metric)
    diss_matrix = ssd.squareform(dissimilarities)

    if output_folder == None:
        output_folder = pathlib.Path.cwd()

    mds_results = {}
    dimensions = range(2,max_dim+1)
    for dim in dimensions:
        mds_results[dim] = do_mds_one_dim_analysis(data, data_name, dim=dim,
                                                   metric=metric, output_folder=output_folder,
                                                   do_outputs=do_outputs)
        print("Stress-1 in dimension {:d}: {:f}".format(dim, compute_kruskal_stress_one(mds_results[dim])))

    if do_outputs and len(dimensions) > 1:
        qs.qs_scatter(dimensions, [mds_results[dim].stress_ for dim in dimensions])
        write_file = output_folder.joinpath(data_name + "_mds_scree.png")
        qs.plt.savefig(str(write_file))
        qs.plt.close()

        qs.qs_scatter(dimensions, [compute_kruskal_stress_one(mds_results[dim]) for dim in dimensions])
        write_file = output_folder.joinpath(data_name + "_mds_stress1_scree.png")
        qs.plt.savefig(str(write_file))
        qs.plt.close()

    return mds_results


def compute_kruskal_stress_one(mds):
    # see "Modern Multidimensional Scaling: Theory and Applications, 2nd ed." page 42

    mds_distances = ssd.pdist(mds.embedding_, metric='euclidean')
    return np.sqrt(mds.stress_/np.sum(mds_distances**2))
