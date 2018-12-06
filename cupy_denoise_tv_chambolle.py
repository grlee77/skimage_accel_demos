#!/usr/bin/env python
"""
Demo of denoise_tv_chambolle, with very minor refactoring to support CuPy
backend for GPU-based computation.

The cupy implementation of diff is pending a PR upstream, so is replicated
here for now.

_denoise_tv_chambolle_nd is a near-exact copy of
skimage.restoration.denoise_tv_chambolle. It differs primarily in the use of
the function `get_array_module` to determine whether numpy or cupy will be
used as the array backend. The name ``xp`` is then used to stand in for either
``cupy`` or ``numpy``.

The nice thing about CuPy as opposed to OpenCV is that it has an increasingly
complete numpy-like API, so it can deal with nearly all numpy dtypes and
any number of array dimensions.

"""
import numpy as np

from numpy.core.multiarray import normalize_axis_index

try:
    import cupy
    have_cupy = True
except ImportError:
    have_cupy = False


def get_array_module(arr):
    """ Check if the array is a cupy GPU array and return the array module.

    This is like cupy.get_array_module, but return numpy when cupy is
    unavailable.

    Paramters
    ---------
    arr : numpy.ndarray or cupy.core.core.ndarray
        The array to check.

    Returns
    -------
    array_module : python module
        This will be cupy when ``arr`` is a ``cupy.ndarray`` and ``False``
        otherwise.
    """
    # TODO: also check for __array_interface__ attribute and not
    #       __cuda_array_interface__?
    if have_cupy:
        return cupy.get_array_module(arr)
    else:
        return np


# TODO: this diff function will be submitted as an upstream PR to CuPy
def diff(a, n=1, axis=-1):
    """Calculate the n-th discrete difference along the given axis.

    The first difference is given by ``out[n] = a[n+1] - a[n]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.

    Args:
        a (array_like): Input array
        n (int, optional):
            The number of times values are differenced. If zero, the input is
            returned as-is.
        axis (int, optional):
            The axis along which the difference is taken, default is the last
            axis.

    Returns:
        diff (ndarray):
            The n-th differences. The shape of the output is the same as `a`
            except along `axis` where the dimension is smaller by `n`. The
            type of the output is the same as the type of the difference
            between any two elements of `a`. This is the same as the type of
            `a` in most cases. A notable exception is `datetime64`, which
            results in a `timedelta64` output array.

    .. seealso:: :func:`numpy.diff`
    """
    a = cupy.asanyarray(a)
    nd = a.ndim
    axis = normalize_axis_index(axis, nd)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = cupy.not_equal if a.dtype == cupy.bool_ else cupy.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a


# like the skimage version, but with module `xp` standing in for either cupy
# or numpy.
def _denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, n_iter_max=200,
                             xp=None):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.

    """

    ndim = image.ndim
    p = xp.zeros((image.ndim, ) + image.shape, dtype=image.dtype)
    g = xp.zeros_like(p)
    d = xp.zeros_like(image)
    i = 0
    slices_g = [slice(None), ] * (ndim + 1)
    slices_d = [slice(None), ] * ndim
    slices_p = [slice(None), ] * (ndim + 1)
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
            E = (d * d).sum()
        else:
            out = image
            E = 0.

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            if xp == np:
                g[tuple(slices_g)] = xp.diff(out, axis=ax)
            else:
                g[tuple(slices_g)] = diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = (g * g).sum(axis=0, keepdims=True)
        xp.sqrt(norm, out=norm)
        E += weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


def denoise_tv_chambolle(image, weight=0.1, eps=2.e-4, n_iter_max=200,
                         multichannel=False):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension.

    Returns
    -------
    out : ndarray
        Denoised image.

    Notes
    -----
    Make sure to set the multichannel parameter appropriately for color images.

    The principle of total variation denoising is explained in
    https://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This code is an implementation of the algorithm of Rudin, Fatemi and Osher
    that was proposed by Chambolle in [1]_.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.

    Examples
    --------
    2D example on astronaut image:

    >>> from skimage import color, data
    >>> img = color.rgb2gray(data.astronaut())[:50, :50]
    >>> img += 0.5 * img.std() * np.random.randn(*img.shape)
    >>> denoised_img = denoise_tv_chambolle(img, weight=60)

    3D example on synthetic data:

    >>> x, y, z = np.ogrid[0:20, 0:20, 0:20]
    >>> mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    >>> mask = mask.astype(np.float)
    >>> mask += 0.2*np.random.randn(*mask.shape)
    >>> res = denoise_tv_chambolle(mask, weight=100)

    """

    xp = get_array_module(image)
    im_type = image.dtype

    # TODO: fix cupy support for img_as_float
    if not im_type.kind == 'f':
        raise NotImplementedError(
            "please convert input to floating point type")
    #    image = img_as_float(image)

    if multichannel:
        out = xp.zeros_like(image)
        for c in range(image.shape[-1]):
            out[..., c] = _denoise_tv_chambolle_nd(image[..., c], weight, eps,
                                                   n_iter_max, xp=xp)
    else:
        out = _denoise_tv_chambolle_nd(image, weight, eps, n_iter_max, xp=xp)
    return out


def demo_timing_cpu_vs_gpu():
    from skimage import data
    from time import time
    for dtype in [np.float32, np.float64]:
        cat = data.chelsea().astype(dtype)
        cat /= cat.max()
        cat = cat + 0.2 * np.random.randn(*cat.shape).astype(dtype)

        # increase dimensions

        # The relative benefit of the GPU is higher for larger inputs.
        cat = np.tile(cat, (4, 4, 1))

        nreps = 2

        tstart = time()
        for n in range(nreps):
            out = denoise_tv_chambolle(cat, multichannel=True)
        dur_cpu = (time() - tstart) / nreps
        print("Duration on CPU ({}): {}".format(cat.dtype, dur_cpu))
        # 2.73 seconds on CPU

        catg = cupy.asarray(cat)
        # dummy run to make sure all CuPy kernels are compiled prior to timing
        outg = denoise_tv_chambolle(catg, multichannel=True)

        tstart = time()
        for n in range(nreps):
            outg = denoise_tv_chambolle(catg, multichannel=True)
        dur_gpu = (time() - tstart) / nreps
        print("Duration on GPU ({}): {} (accel. = {})".format(cat.dtype, dur_gpu, dur_cpu/dur_gpu))

        tstart = time()
        for n in range(nreps):
            catg = cupy.asarray(cat)
            outg = denoise_tv_chambolle(catg, multichannel=True)
            out = outg.get()
        dur_gpu = (time() - tstart) / nreps
        print("Duration on GPU ({}) (including host/device transfers): {} (accel. = {})".format(cat.dtype, dur_gpu, dur_cpu/dur_gpu))
        # 234 ms on GPU


if __name__ == '__main__':
    demo_timing_cpu_vs_gpu()
