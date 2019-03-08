import numpy as np
import itertools


def uniform_cover_fences(x_min, x_max, n, p):
    length = (x_max - x_min) / (n - p * (n-1))
    stride = length-p*length
    lb = np.arange(x_min, x_max, stride)
    ub = np.arange(x_min + length, x_max+stride, stride)

    return (lb,ub)


class EPCover(object):
    """
    Equalized Projection Cover
    """


    def __init__(self, resolution, gain):
        self.resolution = resolution
        self.gain = gain
        pass



    def fit(self, data):
        """ Fit a cover on the data.

        Parameters
        ----------
        data: array-like


            Warning: First column must be an index column.


        """

        lb, ub = uniform_cover_fences(0,100,self.resolution,self.gain)
        print(lb,ub)
        self.lower_bounds = np.vstack(tuple(np.percentile(data[:,1:], perc, axis=0, keepdims=True,interpolation='nearest') for perc in lb))

        self.upper_bounds = np.vstack(tuple(np.percentile(data[:,1:], perc, axis=0, keepdims=True,interpolation='nearest') for perc in ub))

        return self._compute_centers(data)

    def fit_WRONG(self, data):
        data_mins = np.min(data[:,1:], axis=0, keepdims=True)
        data_maxs = np.max(data[:,1:], axis=0, keepdims=True)

        inner_fences = np.vstack(tuple(np.percentile(data[:,1:], i * 100.0/self.resolution, axis=0, keepdims=True) for i in range(1,self.resolution)))
        self.lower_bounds = np.concatenate((data_mins, inner_fences), axis=0)
        self.upper_bounds = np.concatenate((inner_fences, data_maxs), axis=0)

        gains = self.gain * (self.upper_bounds - self.lower_bounds)

        self.lower_bounds -= gains
        self.upper_bounds += gains

        return self._compute_centers(data)

    def _compute_centers(self,data):
        data_dim = data.shape[1] - 1
        ans = []
        for rect in itertools.product(range(self.resolution), repeat=data_dim):
            rect_lb = self.lower_bounds[rect, range(data_dim)]
            rect_ub = self.upper_bounds[rect, range(data_dim)]
            ans.append(0.5 * (rect_lb + rect_ub))
        return ans


    def transform(self, data):
        """ Fit a cover on the data.

        Parameters
        ----------
        data: array-like


            Warning: First column must be an index column.
        """


        data_dim = data.shape[1] - 1
        ans = []
        for rect in itertools.product(range(self.resolution), repeat=data_dim):
            rect_lb = self.lower_bounds[rect, range(data_dim)]
            rect_ub = self.upper_bounds[rect, range(data_dim)]

            logical = np.logical_and(rect_lb <= data[:,1:], data[:,1:] <= rect_ub)
            member_bool = np.all(logical, axis=1, keepdims=False)

            members = data[member_bool,:]
            ans.append(members)

            print("{} members in cube {}".format(len(members),str(rect)))


        return ans
