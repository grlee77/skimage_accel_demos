import numpy as np
from numpy.testing import assert_allclose, run_module_suite
from fast_ndimage import (
    median_filter, sobel, convolve, correlate, gaussian_filter,
    gaussian_filter1d, uniform_filter, uniform_filter1d)


def test_median_filter():
    rtol = atol = 1e-7
    shape = (63, 64)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape).astype(np.float32)

    for mode in ['reflect', ]:
        kwargs = dict(mode=mode, size=3)
        result_ndi = median_filter(x, backend='ndimage', **kwargs)
        result_opencv = median_filter(x, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)


def test_sobel_filter():
    rtol = atol = 1e-7
    shape = (63, 64)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape)

    # TODO: OpenCV 3.3 currently crashing for mode 'wrap':
    # error: ~/miniconda3/conda-bld/opencv_1513818334462/work/opencv-3.3.0/modules/imgproc/src/filter.cpp:127: error: (-215) columnBorderType != BORDER_WRAP in function init
    for axis in [0, 1]:
        for mode in ['reflect', 'mirror', 'constant', 'nearest']:
            kwargs = dict(mode=mode, axis=axis)
            result_ndi = sobel(x, backend='ndimage', **kwargs)
            result_opencv = sobel(x, backend='opencv', **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    axis = 0
    mode = 'reflect'
    for scale in [0.5, 1, 2, None]:
        for delta in [0, 0.5, 2]:
            kwargs = dict(mode=mode, axis=axis, scale=scale, delta=delta)
            result_ndi = sobel(x[:, 0], backend='ndimage', **kwargs)
            result_opencv = sobel(x[:, 0], backend='opencv', **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)


def test_uniform_filter():
    rtol = atol = 1e-7
    shape = (63, 64)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape)

    # TODO: OpenCV 3.3 currently crashing for mode 'wrap':
    # error: ~/miniconda3/conda-bld/opencv_1513818334462/work/opencv-3.3.0/modules/imgproc/src/filter.cpp:127: error: (-215) columnBorderType != BORDER_WRAP in function init
    for mode in ['reflect', 'mirror', 'constant', 'nearest']:
        kwargs = dict(mode=mode, size=(2, 3))
        result_ndi = uniform_filter(x, backend='ndimage', **kwargs)
        result_opencv = uniform_filter(x, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    for squared in [False, True]:
        for normalize in [False, True]:
            kwargs = dict(size=3, mode='reflect', normalize=normalize,
                          squared=squared)
            result_ndi = uniform_filter(x, backend='ndimage', **kwargs)
            result_opencv = uniform_filter(x, backend='opencv', **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    for size in [5, (5, 6), (6, 5), 6]:
        for origin in [-2, -1, 0, 1, 2, (0, 0), (1, 1), (0, 1), (2, 1),
                       (-1, -2)]:
            kwargs = dict(mode='reflect', size=size, origin=origin)
            result_ndi = uniform_filter(x, backend='ndimage', **kwargs)
            result_opencv = uniform_filter(x, backend='opencv', **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)


def test_uniform_filter1d():
    rtol = atol = 1e-7
    shape = (63, 64)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape)

    size = 3
    for axis in [0, 1, -1]:
        for mode in ['reflect', 'mirror', 'constant', 'nearest']:
            kwargs = dict(mode=mode)
            result_ndi = uniform_filter1d(x, size, axis, backend='ndimage', **kwargs)
            result_opencv = uniform_filter1d(x, size, axis, backend='opencv', **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

        for squared in [False, True]:
            for normalize in [False, True]:
                kwargs = dict(mode='reflect', normalize=normalize,
                              squared=squared)
                result_ndi = uniform_filter1d(x, size, axis, backend='ndimage', **kwargs)
                result_opencv = uniform_filter1d(x, size, axis, backend='opencv', **kwargs)
                assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

        for origin in [-1, 0, 1]:
            kwargs = dict(mode='reflect', origin=origin)
            result_ndi = uniform_filter1d(x, size, axis, backend='ndimage', **kwargs)
            result_opencv = uniform_filter1d(x, size, axis, backend='opencv', **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)


def test_gaussian_filter():
    rtol = atol = 1e-12
    shape = (63, 64)
    sigma = (1.5, 3)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape)

    # TODO: OpenCV 3.3 currently crashing for mode 'wrap':
    # error: ~/miniconda3/conda-bld/opencv_1513818334462/work/opencv-3.3.0/modules/imgproc/src/filter.cpp:127: error: (-215) columnBorderType != BORDER_WRAP in function init
    for mode in ['reflect', 'mirror', 'constant', 'nearest']:
        kwargs = dict(mode=mode)
        result_ndi = gaussian_filter(x, sigma, backend='ndimage', **kwargs)
        result_opencv = gaussian_filter(x, sigma, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    mode = 'reflect'
    for truncate in [1, 1.1, 1.5, 2, 4, 5]:
        kwargs = dict(mode=mode, truncate=truncate)
        result_ndi = gaussian_filter(x, sigma, backend='ndimage', **kwargs)
        result_opencv = gaussian_filter(x, sigma, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)


def test_gaussian_filter1d():
    rtol = atol = 1e-12
    shape = (63, 64)
    sigma = 2.5
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape)

    # TODO: OpenCV 3.3 currently crashing for mode 'wrap':
    # error: ~/miniconda3/conda-bld/opencv_1513818334462/work/opencv-3.3.0/modules/imgproc/src/filter.cpp:127: error: (-215) columnBorderType != BORDER_WRAP in function init
    for axis in [0, 1, -1]:
        for mode in ['reflect', 'mirror', 'constant', 'nearest']:
            kwargs = dict(mode=mode)
            result_ndi = gaussian_filter1d(x, sigma, axis, backend='ndimage',
                                           **kwargs)
            result_opencv = gaussian_filter1d(x, sigma, axis, backend='opencv',
                                              **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

        mode = 'reflect'
        for truncate in [1, 2]:
            kwargs = dict(mode=mode, truncate=truncate, axis=axis)
            result_ndi = gaussian_filter1d(x, sigma, backend='ndimage',
                                           **kwargs)
            result_opencv = gaussian_filter1d(x, sigma, backend='opencv',
                                              **kwargs)
            assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)



def test_convolve():
    rtol = atol = 1e-12
    shape = (63, 64)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape)
    weights = rstate.standard_normal((3, 6))

    func = convolve

    # TODO: OpenCV 3.3 currently crashing for mode 'wrap':
    # error: ~/miniconda3/conda-bld/opencv_1513818334462/work/opencv-3.3.0/modules/imgproc/src/filter.cpp:127: error: (-215) columnBorderType != BORDER_WRAP in function init
    for mode in ['reflect', 'mirror', 'constant', 'nearest']:
        kwargs = dict(mode=mode)
        result_ndi = func(x, weights, backend='ndimage', **kwargs)
        result_opencv = func(x, weights, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    for delta in [0, -0.5, 2]:
        kwargs = dict(mode='reflect', delta=delta)
        result_ndi = func(x, weights, backend='ndimage', **kwargs)
        result_opencv = func(x, weights, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    for origin in [-1, 0, 1, (0, 0), (1, 1)]:
        kwargs = dict(mode='reflect', origin=origin)
        result_ndi = func(x, weights, backend='ndimage', **kwargs)
        result_opencv = func(x, weights, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)


# TODO: test threading


def test_correlate():
    rtol = atol = 1e-12
    shape = (63, 64)
    rstate = np.random.RandomState(0)
    x = rstate.standard_normal(shape)
    weights = rstate.standard_normal((4, 4))

    func = correlate

    # TODO: OpenCV 3.3 currently crashing for mode 'wrap':
    # error: ~/miniconda3/conda-bld/opencv_1513818334462/work/opencv-3.3.0/modules/imgproc/src/filter.cpp:127: error: (-215) columnBorderType != BORDER_WRAP in function init
    for mode in ['reflect', 'mirror', 'constant', 'nearest']:
        kwargs = dict(mode=mode)
        result_ndi = func(x, weights, backend='ndimage', **kwargs)
        result_opencv = func(x, weights, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    for delta in [0, -0.5, 2]:
        kwargs = dict(mode='reflect', delta=delta)
        result_ndi = func(x, weights, backend='ndimage', **kwargs)
        result_opencv = func(x, weights, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    for origin in [-1, 0, 1, (0, 0), (1, 1)]:
        kwargs = dict(mode='reflect', origin=origin)
        result_ndi = func(x, weights, backend='ndimage', **kwargs)
        result_opencv = func(x, weights, backend='opencv', **kwargs)
        assert_allclose(result_ndi, result_opencv, rtol=rtol, atol=atol)

    # TODO: assert_raises ValueError on origin=(-1, 1) etc.


if __name__ == "__main__":
    run_module_suite()
