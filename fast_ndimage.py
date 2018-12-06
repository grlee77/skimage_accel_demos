#!/usr/bin/env python
"""
Implementation of several functions from scipy.ndimage with support for
additional backends. Here each function has a new argument ``backend`` that
by default will prefer OpenCV when it is possible, although the user can
override to choose either OpenCV or SciPy.

By importing from this module rather than scipy.ndimage itself, acceleration
of these functions is possible without change to code in downstream libraries.

The OpenCV backends usually only support 2D and a limited number of dtypes
such as uint8, uint16, float32 and/or float64,
"""

import warnings
import numpy as np
from scipy import ndimage as ndi

"""
OpenCV uses (width, height, color) dimension
scikit-image uses (row, column, color)

OpenCV colorspace is BGR while scikit-image's is RGB


getNumThreads/setNumThreads
see explanation at:
https://docs.opencv.org/master/db/de0/group__core__utils.html#ga2db334ec41d98da3129ef4a2342fc4d4

"""

__all__ = ['gaussian_filter',
           'gaussian_filter1d',
           'median_filter',
           'uniform_filter',
           'uniform_filter1d',
           'convolve',
           'correlate',
           'sobel', ]

try:
    import cv2
    have_opencv = True
    ndi_to_opencv_modes = {'reflect': cv2.BORDER_REFLECT,
                           'mirror': cv2.BORDER_REFLECT_101,
                           'wrap': cv2.BORDER_WRAP,  # Warning will not match
                           'nearest': cv2.BORDER_REPLICATE,
                           'constant': cv2.BORDER_CONSTANT}
except ImportError:
    have_opencv = False
    ndi_to_opencv_border = None


def _get_opencv_anchor(origin, kernel_shape):
    if np.isscalar(origin):
        origin = (origin, origin)

    if ((origin[0] > 0 and origin[1] < 0) or
            (origin[1] > 0 and origin[0] < 0)):
        raise NotImplementedError(
            "mixed positive/negative origin not currently supported")

    # have to swap order to match OpenCV result
    origin = (origin[1], origin[0])

    if np.isscalar(kernel_shape):
        kernel_shape = (kernel_shape, kernel_shape)
    kernel_center = tuple([k//2 for k in kernel_shape])
    anchor = tuple([o + c for o, c in zip(origin, kernel_center)])
    return anchor


def _get_opencv_mode(mode, cval):
    if mode not in ndi_to_opencv_modes:
        raise ValueError("Unrecognized mode: {}".format(mode))
    mode = ndi_to_opencv_modes[mode]
    if mode == 'constant' and cval != 0.0:
        # OpenCV functions like filter2D seem to require the user to manually
        # add a border using copyMakeBorder if non-zero cval is desired.
        raise ValueError("constant mode is currently only supported with "
                         "cval = 0")
    if mode == 'wrap':
        # bug in scipy.ndimage functions when mode = 'warp':
        # https://github.com/scikit-image/scikit-image/pull/1583
        warnings.warn("For mode='warp', the OpenCV result may not match "
                      "ndimage.")
    return mode


def _get_backend(ndim, backend, allow_1d=False):
    if backend is None:
        if have_opencv and ndim == 2:
            backend = 'opencv'
        else:
            backend = 'ndimage'
    if backend not in ['opencv', 'ndimage']:
        raise ValueError("Backend must be opencv or ndimage")
    if backend == 'opencv':
        if ndim != 2 and not allow_1d:
            # TODO: handle multichannel
            raise ValueError("OpenCV backend only compatible with 2D images")
    return backend


def uniform_filter(img, size=3, output=None, mode='reflect', cval=0.0,
                   origin=0, backend=None, normalize=True, threads=None,
                   squared=False):
    """Multi-dimensional uniform filter.

    Parameters
    ---------
    see scipy.ndimage.uniform_filter

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    normalize : bool, optional
        Controls whether or not the uniform filter coefficients are normalized
        so that they sum to one.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.
    squared : bool, optional
        If True, this returns uniform_filter(img**2, ...).

    Notes
    -----
    cv2.boxFilter  when `squared == False`
    cv2.sqrBoxFilter when `squared == True`
    cv2.blur correspnds to `normalize == True` and `squared == False`

    Underlying OpenCV functions are defined for dtypes CV_8U, CV_16U, CV_16S,
    CV_32F or CV_64F.

    See Also
    --------
    cv2.boxFilter, cv2.sqrBoxFilter

    """
    backend = _get_backend(img.ndim, backend)
    if mode == 'wrap' and backend == 'opencv':
        warnings.warn("mode='wrap' is unsupported by the underlying OpenCV "
                      "function... falling back to ndimage")
        backend = 'ndimage'

    if backend == 'opencv':
        if threads is not None:
            if threads < 1 and threads != -1:
                raise ValueError(
                    "Invalid number of threads: {}".format(threads))
            threads_orig = cv2.getNumThreads()
            cv2.setNumThreads(threads)
        try:
            opencv_mode = _get_opencv_mode(mode, cval)
            if np.isscalar(size):
                size = (size, size)
            else:
                if len(size) != 2:
                    raise ValueError(
                        "size doesn't match number of image dimensions")
                size = (size[1], size[0])

            if squared:
                func = cv2.sqrBoxFilter
                kwargs = dict(_dst=output)
            else:
                func = cv2.boxFilter
                kwargs = dict(dst=output)

            result = func(img,
                          ddepth=-1,
                          ksize=size,
                          anchor=_get_opencv_anchor(origin, size),
                          normalize=normalize,
                          borderType=opencv_mode,
                          **kwargs)
        finally:
            if threads is not None:
                cv2.setNumThreads(threads_orig)
    elif backend == 'ndimage':
        if squared:
            img = img * img
        result = ndi.uniform_filter(img, size=size, output=output, mode=mode,
                                    cval=cval, origin=origin)
        if not normalize:
            # multiply output by the kernel size
            if np.isscalar(size):
                result *= size**img.ndim
            else:
                result *= np.prod(size)
    return result


def median_filter(img, size=3, footprint=None, output=None, mode='reflect',
                  cval=0.0, origin=0, backend=None, threads=None):
    """Multi-dimensional median filter.

    Parameters
    ---------
    see scipy.ndimage.median_filter

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.


    Notes
    -----
    The OpenCV backend only supports odd-integer ``size`` and does not support
    ``footprint``.  When ``size`` is 3 or 5, filtering for uint8, uint16 and
    float32 is available.  For other sizes, only uint8 filtering can be
    performed.

    See Also
    --------
    cv2.medianBlur  (opeates on uint8, uint16 or float32)

    """
    backend = _get_backend(img.ndim, backend)
    if backend == 'opencv':
        dtype_in = img.dtype
        if footprint is not None:
            if (np.all(footprint == 1) and
                    (footprint.shape[0] == footprint.shape[1])):
                size = footprint.shape[0]
                footprint = None
            else:
                warnings.warn(
                    "footprint is unsupported by the underlying OpenCV "
                    "function... falling back to ndimage")
                backend = 'ndimage'
        if not np.isscalar(size):
            if size[0] == size[1]:
                size = size[0]
            else:
                warnings.warn(
                    "non-square size is unsupported by the underlying "
                    "OpenCV function... falling back to ndimage")
                backend = 'ndimage'

        # check for odd kernel size
        if size % 2 == 0:
            raise ValueError("OpenCV medianBlur requires odd size")

        # check for or convert to compatible dtype
        if size == 3 or size == 5:
            # uint16 and float32 only available for kernel sizes of 3 and 5
            if dtype_in in [np.uint8, np.uint16, np.float32]:
                dtype = dtype_in
            else:
                warnings.warn(
                    "OpenCV median filtering will be performed using float32 "
                    "dtype")
                dtype = np.float32
        else:
            if dtype_in in [np.uint8, ]:
                dtype = dtype_in
            else:
                raise ValueError(
                    ("OpenCV median filter with size={} can only be performed "
                     "for uint8 dtype").format(size))
        img = np.asarray(img, dtype=dtype)

        opencv_mode = _get_opencv_mode(mode, cval)
        if opencv_mode != cv2.BORDER_REFLECT:
            warnings.warn(
                "only mode == 'reflect' is supported by the underlying "
                "OpenCV function... falling back to ndimage")
            backend = 'ndimage'
        if not np.all(np.asarray(origin) == 0):
            warnings.warn(
                "non-zero origin is unsupported by the underlying "
                "OpenCV function... falling back to ndimage")
            backend = 'ndimage'
    if backend == 'opencv':
        if threads is not None:
            if threads < 1 and threads != -1:
                raise ValueError(
                    "Invalid number of threads: {}".format(threads))
            threads_orig = cv2.getNumThreads()
            cv2.setNumThreads(threads)
        try:
            result = cv2.medianBlur(img,
                                    ksize=size,
                                    dst=output)
        finally:
            if threads is not None:
                cv2.setNumThreads(threads_orig)
    elif backend == 'ndimage':
        result = ndi.median_filter(img, size=size, footprint=footprint,
                                   output=output, mode=mode, cval=cval,
                                   origin=origin)
    return result


def uniform_filter1d(img, size=3, axis=-1, output=None, mode='reflect',
                     cval=0.0, origin=0, backend=None, normalize=True,
                     threads=None, squared=False):
    """Uniform filter along a single axis.


    Parameters
    ---------
    see scipy.ndimage.uniform_filter1d

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    normalize : bool, optional
        Controls whether or not the uniform filter coefficients are normalized
        so that they sum to one.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.
    squared : bool, optional
        If True, this returns uniform_filter(img**2, ...).

    See Also
    --------
    cv2.boxFilter, cv2.sqrBoxFilter
    """
    backend = _get_backend(img.ndim, backend)
    if backend == 'opencv':
        axis = axis % img.ndim
        if axis == 0:
            size = (size, 1)
            origin = (origin, 0)
        else:
            size = (1, size)
            origin = (0, origin)
        result = uniform_filter(img, size=size, output=output, mode=mode,
                                cval=cval, origin=origin, backend='opencv',
                                threads=threads)
    else:
        result = ndi.uniform_filter1d(img, size=size, axis=axis, output=output,
                                      mode=mode, cval=cval, origin=origin)
    return result


def convolve(img, weights, output=None, mode='reflect', cval=0.0, origin=0,
             backend=None, delta=0, threads=None):
    """Multidimensional convolution.

    Parameters
    ---------
    see scipy.ndimage.convolve

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    delta : float, optional
        Add this value to the filtered output.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.

    Notes
    -----
    cv2.filter2D supports
        CV_8U input to CV_16S, CV_32F or CV_64F output
        CV_16U or CV_16S input to CV_32F or CV_64F output
        CV_32F input to CV_32F or CV_64F output
        CV_64F input to CV_64F output
        User-defined ddepth is not yet suppported in this wrapper, so the
        output will have the autoselected output depth given by ``ddepth=-1``.

    See Also
    --------
    cv2.filter2D
    """
    backend = _get_backend(img.ndim, backend)
    if mode == 'wrap' and backend == 'opencv':
        warnings.warn("mode='wrap' is unsupported by the underlying OpenCV "
                      "function... falling back to ndimage")
        backend = 'ndimage'

    if backend == 'opencv':
        if threads is not None:
            if threads < 1 and threads != -1:
                raise ValueError(
                    "Invalid number of threads: {}".format(threads))
            threads_orig = cv2.getNumThreads()
            cv2.setNumThreads(threads)
        try:
            opencv_mode = _get_opencv_mode(mode, cval)
            anchor = _get_opencv_anchor(origin, weights.shape)

            if np.isscalar(origin):
                origin = (origin, origin)
            if origin[0] != origin[1]:
                # TODO: fix: does not match ndimage if origin[0] != origin[1]
                raise NotImplementedError(
                    "origin[0] != origin[1] is not supported in opencv mode")

            """
            It is necessary to adjust the kernel and anchor for the fact that
            filter2D actually performs correlation, not convolution.

            To get a true convolution, we must flip the kernel and adjust the
            anchor point as described in the OpenCV documentation of filter2D.
            """
            kernel = weights[::-1, ::-1]
            anchor = (kernel.shape[1] - anchor[1] - 1,
                      kernel.shape[0] - anchor[0] - 1)

            result = cv2.filter2D(img,
                                  dst=output,
                                  ddepth=-1,
                                  kernel=kernel,
                                  anchor=anchor,
                                  delta=delta,
                                  borderType=opencv_mode)
        finally:
            if threads is not None:
                cv2.setNumThreads(threads_orig)
    elif backend == 'ndimage':
        result = ndi.convolve(img, weights, output=output, mode=mode,
                              cval=cval, origin=origin)
        if delta != 0:
            result += delta
    return result


def sobel(img, axis=-1, output=None, mode='reflect', cval=0.0, backend=None,
          threads=None, delta=0, scale=None):
    """Sobel filter along a specific axis.

    Parameters
    ---------
    see scipy.ndimage.sobel

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.
    delta : float, optional
        Add this value to the filtered output.
    scale : float or None, optional
        Scale the filtered output by this amount.

    Notes
    -----
    cv2.Sobel supports
        CV_8U input to CV_16S, CV_32F or CV_64F output
        CV_16U or CV_16S input to CV_32F or CV_64F output
        CV_32F input to CV_32F or CV_64F output
        CV_64F input to CV_64F output
        User-defined ddepth is not yet suppported in this wrapper, so the
        output will have the autoselected output depth given by ``ddepth=-1``.

    See Also
    --------
    cv2.Sobel
    """
    backend = _get_backend(img.ndim, backend, allow_1d=True)
    if mode == 'wrap' and backend == 'opencv':
        warnings.warn("mode='wrap' is unsupported by the underlying OpenCV "
                      "function... falling back to ndimage")
        backend = 'ndimage'
    if backend == 'opencv':
        shape_in = img.shape
        if scale is None:
            scale = 1
        if img.ndim == 1:
            img = img[:, np.newaxis]
            axis = 0
            if mode == 'constant':
                scale = 0.5 * scale
            else:
                scale = 0.25 * scale
        axis = axis % img.ndim
        # set order to 0 on the axis that is not filtered
        if axis == 0:
            dx, dy = 0, 1
        else:
            dx, dy = 1, 0
        opencv_mode = _get_opencv_mode(mode, cval)
        result = cv2.Sobel(img, ddepth=-1, dx=dx, dy=dy, ksize=3, dst=output,
                           scale=scale,
                           delta=delta,
                           borderType=opencv_mode)
        result = result.reshape(shape_in)
    else:
        result = ndi.sobel(img, axis=axis, output=output, mode=mode, cval=cval)
        if scale is not None:
            result *= scale
        if delta != 0:
            result += delta

    return result


def correlate(img, weights, output=None, mode='reflect', cval=0.0, origin=0,
              backend=None, delta=0, threads=None):
    """Multidimensional correlation.

    Parameters
    ---------
    see scipy.ndimage.correlate

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    delta : float, optional
        Add this value to the filtered output.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.

    See Also
    --------
    cv2.filter2D
    """
    backend = _get_backend(img.ndim, backend)
    if mode == 'wrap' and backend == 'opencv':
        warnings.warn("mode='wrap' is unsupported by the underlying OpenCV "
                      "function... falling back to ndimage")
        backend = 'ndimage'

    if backend == 'opencv':
        if threads is not None:
            if threads < 1 and threads != -1:
                raise ValueError(
                    "Invalid number of threads: {}".format(threads))
            threads_orig = cv2.getNumThreads()
            cv2.setNumThreads(threads)
        try:
            opencv_mode = _get_opencv_mode(mode, cval)
            anchor = _get_opencv_anchor(origin, weights.shape)

            if np.isscalar(origin):
                origin = (origin, origin)
            if origin[0] != origin[1]:
                # TODO: fix: does not match ndimage if origin[0] != origin[1]
                raise NotImplementedError(
                    "origin[0] != origin[1] is not supported in opencv mode")

            kernel = weights

            # TODO: why is this coordinate swap necessary for correlate, but not
            #       for convolve?
            anchor = (anchor[1], anchor[0])

            result = cv2.filter2D(img,
                                  dst=output,
                                  ddepth=-1,
                                  kernel=kernel,
                                  anchor=anchor,
                                  delta=delta,
                                  borderType=opencv_mode)
        finally:
            if threads is not None:
                cv2.setNumThreads(threads_orig)
    elif backend == 'ndimage':
        result = ndi.correlate(img, weights, output=output, mode=mode,
                               cval=cval, origin=origin)
        if delta != 0:
            result += delta
    return result


def gaussian_filter(img, sigma, order=0, output=None, mode='reflect', cval=0.0,
                    truncate=4.0, backend=None, threads=None):
    """Multidimensional Gaussian filter.

    Parameters
    ---------
    see scipy.ndimage.gaussian_filter

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.

    Notes
    -----
    cv2.GaussianBlur implemented for CV_8U, CV_16U, CV_16S, CV_32F, CV_64F and
    for any number of channels.

    See Also
    --------
    cv2.GaussianBlur
    """
    backend = _get_backend(img.ndim, backend)
    if backend == 'opencv':
        if mode == 'wrap':
            warnings.warn(
                "mode == 'wrap' is unsupported by the underlying OpenCV "
                "function... falling back to ndimage")
            backend = 'ndimage'
        if order != 0:
            warnings.warn(
                "order != 0 is unsupported by the underlying OpenCV "
                "function... falling back to ndimage")
            backend = 'ndimage'

    if backend == 'opencv':
        if threads is not None:
            if threads < 1 and threads != -1:
                raise ValueError(
                    "Invalid number of threads: {}".format(threads))
            threads_orig = cv2.getNumThreads()
            cv2.setNumThreads(threads)
        try:
            opencv_mode = _get_opencv_mode(mode, cval)
            if np.isscalar(sigma):
                sigma = (sigma, sigma)
            if np.isscalar(truncate):
                truncate = (truncate, truncate)

            # determine ksize from sigma & truncate
            # the equation used is from scipy.ndimage.gaussian_filter1d
            wx = (2 * int(truncate[1] * sigma[1] + 0.5) + 1)
            wy = (2 * int(truncate[0] * sigma[0] + 0.5) + 1)

            result = cv2.GaussianBlur(img,
                                      dst=output,
                                      ksize=(wx, wy),
                                      sigmaX=sigma[1],
                                      sigmaY=sigma[0],
                                      borderType=opencv_mode)
        finally:
            if threads is not None:
                cv2.setNumThreads(threads_orig)
    elif backend == 'ndimage':
        result = ndi.gaussian_filter(img, sigma, order=order, output=output,
                                     mode=mode, cval=cval, truncate=truncate)
    return result


def gaussian_filter1d(img, sigma, axis=-1, order=0, output=None,
                      mode='reflect', cval=0.0, truncate=4.0, backend=None,
                      threads=None):
    """Gaussian filter along a single axis.

    Parameters
    ---------
    see scipy.ndimage.gaussian_filter1d

    Additional Parameters
    --------------------
    backend : {None, 'ndimage', 'opencv'}, optional
        If None, defaults to OpenCV for 2D images when possible.  If OpenCV is
        not available or input.ndim != 2, ndimage is always used.
    threads : int or None, optional
        The number of threads the OpenCV backend will use.  If None, the number
        of threads is not set internally (the value returned by
        cv2.getNumThreads() is used).  ``threads=-1`` can be used to specify
        that all available threads should be used.

    See Also
    --------
    cv2.GaussianBlur
    """
    backend = _get_backend(img.ndim, backend)
    if backend == 'opencv':
        axis = axis % img.ndim
        # Trick: set truncate so that filter will have size one on the axis
        # that is not filtered.
        if axis == 0:
            truncate = (truncate, 0)
        else:
            truncate = (0, truncate)
        result = gaussian_filter(img, sigma, order=order, output=output,
                                 mode=mode, cval=cval, truncate=truncate,
                                 backend='opencv', threads=threads)
    else:
        result = ndi.gaussian_filter1d(img, sigma, axis=axis, order=order,
                                       output=output, mode=mode, cval=cval,
                                       truncate=truncate)
    return result



def demo_timing_median():
    from skimage import data
    from time import time
    cat = data.chelsea().astype(np.float32)
    cat /= cat.max()
    cat = cat + 0.2 * np.random.randn(*cat.shape).astype(cat.dtype)
    cat_grey = cat[..., 0]

    # increase dimensions

    cat_grey = np.tile(cat_grey, (4, 4))

    nreps = 5

    tstart = time()
    for n in range(nreps):
        out = median_filter(cat_grey, size=5, backend='ndimage')
    dur_ndi = (time() - tstart)/nreps
    print("Duration (median, dtype={}) with scipy.ndimage: {}".format(cat_grey.dtype, dur_ndi))

    tstart = time()
    for n in range(nreps):
        out = median_filter(cat_grey, size=5, backend='opencv')
    dur_cv = (time() - tstart)/nreps
    print("Duration (median, dtype={}) with opencv: {} (accel. = {})".format(cat_grey.dtype, dur_cv, dur_ndi/dur_cv))

    tstart = time()
    for n in range(nreps):
        out = gaussian_filter(cat_grey, sigma=2, backend='ndimage')
    dur_ndi = (time() - tstart)/nreps
    print("Duration (gaussian, dtype={}) with scipy.ndimage: {}".format(cat_grey.dtype , dur_ndi))

    tstart = time()
    for n in range(nreps):
        out = gaussian_filter(cat_grey, sigma=2, backend='opencv')
    dur_cv = (time() - tstart)/nreps
    print("Duration (gaussian, dtype={}) with opencv: {} (accel. = {})".format(cat_grey.dtype , dur_cv, dur_ndi/dur_cv))

    tstart = time()
    for n in range(nreps):
        out = uniform_filter(cat_grey, size=5, backend='ndimage')
    dur_ndi = (time() - tstart)/nreps
    print("Duration (uniform, dtype={}) with scipy.ndimage: {}".format(cat_grey.dtype , dur_ndi))

    tstart = time()
    for n in range(nreps):
        out = uniform_filter(cat_grey, size=5, backend='opencv')
    dur_cv = (time() - tstart)/nreps
    print("Duration (uniform, dtype={}) with opencv: {} (accel. = {})".format(cat_grey.dtype , dur_cv, dur_ndi/dur_cv))

    tstart = time()
    for n in range(nreps):
        out = sobel(cat_grey, backend='ndimage')
    dur_ndi = (time() - tstart)/nreps
    print("Duration (sobel, dtype={}) with scipy.ndimage: {}".format(cat_grey.dtype , dur_ndi))

    tstart = time()
    for n in range(nreps):
        out = sobel(cat_grey, backend='opencv')
    dur_cv = (time() - tstart)/nreps
    print("Duration (sobel, dtype={}) with opencv: {} (accel. = {})".format(cat_grey.dtype , dur_cv, dur_ndi/dur_cv))

    weights = np.random.rand(5, 5).astype(cat_grey.dtype)
    tstart = time()
    for n in range(nreps):
        out = correlate(cat_grey, weights, backend='ndimage')
    dur_ndi = (time() - tstart)/nreps
    print("Duration (correlate, dtype={}) with scipy.ndimage: {}".format(cat_grey.dtype, dur_ndi))

    for n in range(nreps):
        out = correlate(cat_grey, weights, backend='opencv')
    dur_ndi = (time() - tstart)/nreps
    print("Duration (correlate, dtype={}) with opencv: {} (accel. = {})".format(cat_grey.dtype, dur_cv, dur_ndi/dur_cv))

if __name__ == '__main__':
    demo_timing_median()
