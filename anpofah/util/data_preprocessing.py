import numpy as np

def is_outlier(points, thresh=4.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def is_outlier_percentile(points, percentile=99.999):
    diff = (100 - percentile) / 2.0
    minval, maxval = np.percentile(points, [diff, 100 - diff])
    return (points < minval) | (points > maxval)

def clip_outlier(data):
    idx = is_outlier_percentile(data)
    return data[~idx]
