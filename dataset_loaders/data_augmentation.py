# Based on
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
import os
import shutil
import warnings

import numpy as np
import scipy.misc
import scipy.ndimage as ndi
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float


def farn_optical_flow(dataset):
    '''Farneback optical flow

    Takes a 4D array of sequences and returns a 4D array with
    an RGB optical flow image for each frame in the input'''
    import cv2
    warnings.warn('Farneback optical flow not stored on disk. It will now be '
                  'computed on the whole dataset and stored on disk.'
                  'Time to sit back and get a coffee!')

    # Create a copy of the dataset to iterate on
    dataset = dataset.__class__(batch_size=1,
                                return_01c=True,
                                return_0_255=True,
                                shuffle_at_each_epoch=False,
                                infinite_iterator=False)

    ret = dataset.next()
    frame0 = ret['data']
    prefix0 = ret['subset'][0]
    if frame0.ndim != 4:
        raise RuntimeError('Optical flow expected 4 dimensions, got %d' %
                           frame0.ndim)
    frame0 = frame0[0, ..., ::-1]  # go BGR for OpenCV + remove batch dim
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)  # Go gray

    flow = None
    of_path = os.path.join(dataset.path, 'OF', 'Farn')
    of_shared_path = os.path.join(dataset.shared_path, 'OF', 'Farn')

    for ret in dataset:
        frame1 = ret['data']
        filename1 = ret['filenames'][0, 0]
        # Strip extension, if any
        filename1 = filename1[:-4] + '.'.join(filename1[-4:].split('.')[:-1])
        prefix1 = ret['subset'][0]

        if frame1.ndim != 4:
            raise RuntimeError('Optical flow expected 4 dimensions, got %d' %
                               frame1.ndim)

        frame1 = frame1[0, ..., ::-1]  # go BGR for OpenCV + remove batch dim
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Go gray

        if prefix1 != prefix0:
            # First frame of a new subset
            frame0 = frame1
            prefix0 = prefix1
            continue

        # Compute displacement
        flow = cv2.calcOpticalFlowFarneback(prev=frame0,
                                            next=frame1,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=10,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.1,
                                            flags=0,
                                            flow=flow)

        # Save in the local path
        if not os.path.exists(os.path.join(of_path, prefix1)):
            os.makedirs(os.path.join(of_path, prefix1))
        # Save the flow as dy, dx
        np.save(os.path.join(of_path, prefix1, filename1), flow[..., ::-1])
        # cv2.imwrite(os.path.join(of_path, prefix1, filename1 + '.png'), flow)
        frame0 = frame1
        prefix0 = prefix1

    # Store a copy in shared_path
    # TODO there might be a race condition when multiple experiments are
    # run and one checks for the existence of the shared path OF dir
    # while this copy is happening.
    if of_path != of_shared_path:
        shutil.copytree(of_path, of_shared_path)


def my_label2rgb(labels, cmap, bglabel=None, bg_color=(0., 0., 0.)):
    '''Convert a label mask to RGB applying a color map'''
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(cmap)):
        if i != bglabel:
            output[(labels == i).nonzero()] = cmap[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


def my_label2rgboverlay(labels, cmap, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    '''Superimpose a mask over an image

    Convert a label mask to RGB applying a color map and superimposing it
    over an image as a transparent overlay'''
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, cmap, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


def save_img2(x, y, fname, cmap, void_label, rows_idx, cols_idx,
              chan_idx):
    '''Save a mask and an image side to side

    Convert a label mask to RGB applying a color map and superimposing it
    over an image as a transparent overlay. Saves the original image and
    the image with the mask overlay in a file'''
    pattern = [el for el in range(x.ndim) if el not in [rows_idx, cols_idx,
                                                        chan_idx]]
    pattern += [rows_idx, cols_idx, chan_idx]
    x_copy = x.transpose(pattern)
    if y is not None and len(y) > 0:
        y_copy = y.transpose(pattern)

    # Take the first batch and drop extra dim on y
    x_copy = x_copy[0]
    if y is not None and len(y) > 0:
        y_copy = y_copy[0, ..., 0]

    label_mask = my_label2rgboverlay(y,
                                     colors=cmap,
                                     image=x,
                                     bglabel=void_label,
                                     alpha=0.2)
    combined_image = np.concatenate((x, label_mask),
                                    axis=1)
    scipy.misc.toimage(combined_image).save(fname)


def transform_matrix_offset_center(matrix, x, y):
    '''Shift the transformation matrix to be in the center of the image

    Apply an offset to the transformation matrix so that the origin of
    the axis is in the center of the image.'''
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.,
                    order=0, rows_idx=1, cols_idx=2):
    '''Apply an affine transformation on each channel separately.'''
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    # Reshape to (*, 0, 1)
    pattern = [el for el in range(x.ndim) if el != rows_idx and el != cols_idx]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[-2:])  # squash everything on the first axis

    # Apply the transformation on each channel, sequence, batch, ..
    for i in range(x.shape[0]):
        x[i] = ndi.interpolation.affine_transform(x[i], final_affine_matrix,
                                                  final_offset, order=order,
                                                  mode=fill_mode, cval=cval)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def random_channel_shift(x, shift_range, rows_idx, cols_idx, chan_idx):
    '''Shift the intensity values of each channel uniformly.

    Channel by channel, shift all the intensity values by a random value in
    [-shift_range, shift_range]'''
    pattern = [chan_idx]
    pattern += [el for el in range(x.ndim) if el not in [rows_idx, cols_idx,
                                                         chan_idx]]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # channel first
    x_shape = list(x.shape)
    # squash rows and cols together and everything else on the 1st
    x = x.reshape((-1, x_shape[-2] * x_shape[-1]))
    # Loop on the channels/batches/etc
    for i in range(x.shape[0]):
        min_x, max_x = np.min(x), np.max(x)
        x[i] = np.clip(x[i] + np.random.uniform(-shift_range, shift_range),
                       min_x, max_x)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def flip_axis(x, flipping_axis):
    '''Flip an axis by inverting the position of its elements'''
    pattern = [flipping_axis]
    pattern += [el for el in range(x.ndim) if el != flipping_axis]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # "flipping_axis" first
    x = x[::-1, ...]
    x = x.transpose(inv_pattern)
    return x


def pad_image(x, pad_amount, mode='reflect', constant=0.):
    '''Pad an image

    Pad an image by pad_amount on each side.

    Parameters
    ----------
    x: numpy ndarray
        The array to be padded.
    pad_amount: int
        The number of pixels of the padding.
    mode: string
        The padding mode. If "constant" a constant value will be used to
        fill the padding; if "reflect" the border pixels will be used in
        inverse order to fill the padding; if "nearest" the border pixel
        closer to the padded area will be used to fill the padding; if
        "zero" the padding will be filled with zeros.
    constant: int
        The value used to fill the padding when "constant" mode is
        selected.
        '''
    e = pad_amount
    shape = list(x.shape)
    shape[:2] += 2*e
    if mode == 'constant':
        x_padded = np.ones(shape, dtype=np.float32)*constant
        x_padded[e:-e, e:-e] = x.copy()
    else:
        x_padded = np.zeros(shape, dtype=np.float32)
        x_padded[e:-e, e:-e] = x.copy()

    if mode == 'reflect':
        # Edges
        x_padded[:e, e:-e] = np.flipud(x[:e, :])  # left
        x_padded[-e:, e:-e] = np.flipud(x[-e:, :])  # right
        x_padded[e:-e, :e] = np.fliplr(x[:, :e])  # top
        x_padded[e:-e, -e:] = np.fliplr(x[:, -e:])  # bottom
        # Corners
        x_padded[:e, :e] = np.fliplr(np.flipud(x[:e, :e]))  # top-left
        x_padded[-e:, :e] = np.fliplr(np.flipud(x[-e:, :e]))  # top-right
        x_padded[:e, -e:] = np.fliplr(np.flipud(x[:e, -e:]))  # bottom-left
        x_padded[-e:, -e:] = np.fliplr(np.flipud(x[-e:, -e:]))  # bottom-right
    elif mode == 'zero' or mode == 'constant':
        pass
    elif mode == 'nearest':
        # Edges
        x_padded[:e, e:-e] = x[[0], :]  # left
        x_padded[-e:, e:-e] = x[[-1], :]  # right
        x_padded[e:-e, :e] = x[:, [0]]  # top
        x_padded[e:-e, -e:] = x[:, [-1]]  # bottom
        # Corners
        x_padded[:e, :e] = x[[0], [0]]  # top-left
        x_padded[-e:, :e] = x[[-1], [0]]  # top-right
        x_padded[:e, -e:] = x[[0], [-1]]  # bottom-left
        x_padded[-e:, -e:] = x[[-1], [-1]]  # bottom-right
    else:
        raise ValueError("Unsupported padding mode \"{}\"".format(mode))
    return x_padded


def gen_warp_field(shape, sigma=0.1, grid_size=3):
    '''Generate an spline warp field'''
    import SimpleITK as sitk
    # Initialize bspline transform
    args = shape+(sitk.sitkFloat32,)
    ref_image = sitk.Image(*args)
    tx = sitk.BSplineTransformInitializer(ref_image, [grid_size, grid_size])

    # Initialize shift in control points:
    # mesh size = number of control points - spline order
    p = sigma * np.random.randn(grid_size+3, grid_size+3, 2)

    # Anchor the edges of the image
    p[:, 0, :] = 0
    p[:, -1:, :] = 0
    p[0, :, :] = 0
    p[-1:, :, :] = 0

    # Set bspline transform parameters to the above shifts
    tx.SetParameters(p.flatten())

    # Compute deformation field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_image)
    displacement_field = displacement_filter.Execute(tx)

    return displacement_field


def apply_warp(x, warp_field, fill_mode='reflect',
               interpolator=None,
               fill_constant=0, rows_idx=1, cols_idx=2):
    '''Apply an spling warp field on an image'''
    import SimpleITK as sitk
    if interpolator is None:
        interpolator = sitk.sitkLinear
    # Expand deformation field (and later the image), padding for the largest
    # deformation
    warp_field_arr = sitk.GetArrayFromImage(warp_field)
    max_deformation = np.max(np.abs(warp_field_arr))
    pad = np.ceil(max_deformation).astype(np.int32)
    warp_field_padded_arr = pad_image(warp_field_arr, pad_amount=pad,
                                      mode='nearest')
    warp_field_padded = sitk.GetImageFromArray(warp_field_padded_arr,
                                               isVector=True)

    # Warp x, one filter slice at a time
    pattern = [el for el in range(0, x.ndim) if el not in [rows_idx, cols_idx]]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # batch, channel, ...
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[2:])  # *, r, c
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetEdgePaddingValue(np.min(x).astype(np.double))
    for i in range(x.shape[0]):
        bc_pad = pad_image(x[i], pad_amount=pad, mode=fill_mode,
                           constant=fill_constant).T
        bc_f = sitk.GetImageFromArray(bc_pad)
        bc_f_warped = warp_filter.Execute(bc_f, warp_field_padded)
        bc_warped = sitk.GetArrayFromImage(bc_f_warped)
        x[i] = bc_warped[pad:-pad, pad:-pad].T
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def random_transform(dataset,
                     seq,
                     prefix_and_fnames=None,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     cval_mask=0.,
                     horizontal_flip=0.,  # probability
                     vertical_flip=0.,  # probability
                     rescale=None,
                     spline_warp=False,
                     warp_sigma=0.1,
                     warp_grid_size=3,
                     crop_size=None,
                     nclasses=None,
                     gamma=0.,
                     gain=1.,
                     return_optical_flow=None,
                     optical_flow_type='Farn',
                     chan_idx=3,  # No batch yet: (s, 0, 1, c)
                     rows_idx=1,  # No batch yet: (s, 0, 1, c)
                     cols_idx=2,  # No batch yet: (s, 0, 1, c)
                     void_label=None):
    '''Random Transform.

    A function to perform data augmentation of images and masks during
    the training  (on-the-fly). Based on [RandomTransform1]_.

    Parameters
    ----------
    dataset: a :class:`Dataset` instance
        The instance of the current dataset. First step towards making
        this a class method.
    seq: a dictionary of numpy array
        A dictionary with at least these keys: 'data', 'labels', 'filenames',
        'subset'.
    prefix_and_fnames: list
        A list of prefix and names for the current sequence
    rotation_range: int
        Degrees of rotation (0 to 180).
    width_shift_range: float
        The maximum amount the image can be shifted horizontally (in
        percentage).
    height_shift_range: float
        The maximum amount the image can be shifted vertically (in
        percentage).
    shear_range: float
        The shear intensity (shear angle in radians).
    zoom_range: float or list of floats
        The amout of zoom. If set to a scalar z, the zoom range will be
        randomly picked in the range [1-z, 1+z].
    channel_shift_range: float
        The shift range for each channel.
    fill_mode: string
        Some transformations can return pixels that are outside of the
        boundaries of the original image. The points outside the
        boundaries are filled according to the given mode (`constant`,
        `nearest`, `reflect` or `wrap`). Default: `nearest`.
    cval: int
        Value used to fill the points of the image outside the boundaries when
        fill_mode is `constant`. Default: 0.
    cval_mask: int
        Value used to fill the points of the mask outside the boundaries when
        fill_mode is `constant`. Default: 0.
    horizontal_flip: float
        The probability to randomly flip the images (and masks)
        horizontally. Default: 0.
    vertical_flip: bool
        The probability to randomly flip the images (and masks)
        vertically. Default: 0.
    rescale: float
        The rescaling factor. If None or 0, no rescaling is applied, otherwise
        the data is multiplied by the value provided (before applying
        any other transformation).
    spline_warp: bool
        Whether to apply spline warping.
    warp_sigma: float
        The sigma of the gaussians used for spline warping.
    warp_grid_size: int
        The grid size of the spline warping.
    crop_size: tuple
        The size of crop to be applied to images and masks (after any
        other transformation).
    nclasses: int
        The number of classes of the dataset.
    gamma: float
        Controls gamma in Gamma correction.
    gain: float
        Controls gain in Gamma correction.
    return_optical_flow: string
        Either 'displacement' or 'rbg'.
        If set, a dense optical flow will be retrieved from disk (or
        computed when missing) and returned as a 'flow' key.
        If 'displacement', the optical flow will be returned as a
        two-dimensional array of (dx, dy) displacement. If 'rgb', a
        three dimensional RGB array with values in [0, 255] will be
        returned. Default: None.
    optical_flow_type: string
        Indicates the method used to generate the optical flow. The
        optical flow is loaded from a specific directory based on this
        type.
    chan_idx: int
        The index of the channel axis.
    rows_idx: int
        The index of the rows of the image.
    cols_idx: int
        The index of the cols of the image.
    void_label: int
        The index of the void label, if any. Used for padding.

    References
    ----------
    .. [RandomTransform1]
       https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    '''
    # Set this to a dir, if you want to save augmented images samples
    save_to_dir = None
    nclasses = dataset.nclasses
    void_label = dataset.void_labels

    if rescale:
        raise NotImplementedError()

    # Make sure we do not modify the original images
    seq['data'] = seq['data'].copy()
    if seq['labels'] is not None and len(seq['labels']) > 0:
        seq['labels'] = seq['labels'].copy()
        # Add extra dim to y to simplify computation
        seq['labels'] = seq['labels'][..., None]
    sh = seq['data'].shape

    # listify zoom range
    if np.isscalar(zoom_range):
        if zoom_range > 1.:
            raise RuntimeError('Zoom range should be between 0 and 1. '
                               'Received: ', zoom_range)
        zoom_range = [1 - zoom_range, 1 - zoom_range]
    elif len(zoom_range) == 2:
        if any(el > 1. for el in zoom_range):
            raise RuntimeError('Zoom range should be between 0 and 1. '
                               'Received: ', zoom_range)
        zoom_range = [1-el for el in zoom_range]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    # Channel shift
    if channel_shift_range != 0:
        seq['data'] = random_channel_shift(seq['data'], channel_shift_range,
                                           rows_idx, cols_idx, chan_idx)

    # Gamma correction
    if gamma > 0:
        scale = float(1)
        seq['data'] = ((seq['data'] / scale) ** gamma) * scale * gain

    # Affine transformations (zoom, rotation, shift, ..)
    if (rotation_range or height_shift_range or width_shift_range or
            shear_range or zoom_range != [1, 1]):

        # --> Rotation
        if rotation_range:
            theta = np.pi / 180 * np.random.uniform(-rotation_range,
                                                    rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        # --> Shift/Translation
        if height_shift_range:
            tx = (np.random.uniform(-height_shift_range, height_shift_range) *
                  sh[rows_idx])
        else:
            tx = 0
        if width_shift_range:
            ty = (np.random.uniform(-width_shift_range, width_shift_range) *
                  sh[cols_idx])
        else:
            ty = 0
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        # --> Shear
        if shear_range:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        # --> Zoom
        if zoom_range == [1, 1]:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        # Use a composition of homographies to generate the final transform
        # that has to be applied
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                translation_matrix),
                                         shear_matrix), zoom_matrix)
        h, w = sh[rows_idx], sh[cols_idx]
        transform_matrix = transform_matrix_offset_center(transform_matrix,
                                                          h, w)
        # Apply all the transformations together
        seq['data'] = apply_transform(seq['data'], transform_matrix,
                                      fill_mode=fill_mode, cval=cval, order=1,
                                      rows_idx=rows_idx, cols_idx=cols_idx)
        if seq['labels'] is not None and len(seq['labels']) > 0:
            seq['labels'] = apply_transform(seq['labels'],
                                            transform_matrix,
                                            fill_mode=fill_mode,
                                            cval=cval_mask,
                                            order=0,
                                            rows_idx=rows_idx,
                                            cols_idx=cols_idx)

    # Horizontal flip
    if np.random.random() < horizontal_flip:  # 0 = disabled
        seq['data'] = flip_axis(seq['data'], cols_idx)
        if seq['labels'] is not None and len(seq['labels']) > 0:
            seq['labels'] = flip_axis(seq['labels'], cols_idx)

    # Vertical flip
    if np.random.random() < vertical_flip:  # 0 = disabled
        seq['data'] = flip_axis(seq['data'], rows_idx)
        if seq['labels'] is not None and len(seq['labels']) > 0:
            seq['labels'] = flip_axis(seq['labels'], rows_idx)

    # Spline warp
    if spline_warp:
        import SimpleITK as sitk
        warp_field = gen_warp_field(shape=(sh[rows_idx],
                                           sh[cols_idx]),
                                    sigma=warp_sigma,
                                    grid_size=warp_grid_size)
        seq['data'] = apply_warp(seq['data'], warp_field,
                                 interpolator=sitk.sitkLinear,
                                 fill_mode=fill_mode, fill_constant=cval,
                                 rows_idx=rows_idx, cols_idx=cols_idx)
        if seq['labels'] is not None and len(seq['labels']) > 0:
            # TODO is this round right??
            seq['labels'] = np.round(
                apply_warp(seq['labels'], warp_field,
                           interpolator=sitk.sitkNearestNeighbor,
                           fill_mode=fill_mode, fill_constant=cval_mask,
                           rows_idx=rows_idx, cols_idx=cols_idx))

    # Optical flow
    if return_optical_flow:
        return_optical_flow = return_optical_flow.lower()
        if return_optical_flow not in ['rgb', 'displacement']:
            raise RuntimeError('Unknown return_optical_flow value: %s' %
                               return_optical_flow)
        if prefix_and_fnames is None:
            raise RuntimeError('You should specify a list of prefixes '
                               'and filenames')
        # Find the filename of the first frame of this prefix
        first_frame_of_prefix = sorted(dataset.get_names()[seq['subset']])[0]

        of_base_path = os.path.join(dataset.path, 'OF', optical_flow_type)
        if not os.path.isdir(of_base_path):
            # The OF is not on disk: compute it and store it
            if optical_flow_type != 'Farn':
                raise RuntimeError('Unknown optical flow type: %s. For '
                                   'optical_flow_type other than Farn '
                                   'please run your own implementation '
                                   'manually and save it in %s' %
                                   optical_flow_type, of_base_path)
            farn_optical_flow(dataset)  # Compute and store on disk

        # Load the OF from disk
        import skimage
        flow = []
        for frame in prefix_and_fnames:
            if frame[1] == first_frame_of_prefix:
                # It's the first frame of the prefix, there is no
                # previous frame to compute the OF with, return a blank one
                of = np.zeros(sh[1:], seq['data'].dtype)
                flow.append(of)
                continue

            # Read from disk
            of_path = os.path.join(of_base_path, frame[0],
                                   frame[1].rstrip('.') + '.npy')
            if os.path.exists(of_path):
                of = np.load(of_path)
            else:
                raise RuntimeError('Optical flow not found for this '
                                   'file: %s' % of_path)

            if return_optical_flow == 'rgb':
                # of = of[..., ::-1]

                def cart2pol(x, y):
                    mag = np.sqrt(x**2 + y**2)
                    ang = np.arctan2(y, x)  # note, in [-pi, pi]
                    return mag, ang
                mag, ang = cart2pol(of[..., 0], of[..., 1])

                # Normalize to [0, 1]
                sh = of.shape[:2]
                two_pi = 2 * np.pi
                ang = (ang + two_pi) % two_pi / two_pi
                mag = mag - mag.min()
                mag /= np.float(mag.max())

                # Convert to RGB [0, 1]
                hsv = np.ones((sh[0], sh[1], 3))
                hsv[..., 0] = ang
                hsv[..., 2] = mag
                of = skimage.color.hsv2rgb(hsv)  # HSV --> RGB [0, 1]
                of = (of * 255).astype('uint8')
                from PIL import Image
                import ipdb; ipdb.set_trace()
                Image.fromarray(of).show()

            flow.append(np.array(of))
        flow = np.array(flow)

    # Crop
    # Expects axes with shape (..., 0, 1)
    # TODO: Add center crop
    if crop_size:
        # Reshape to (..., 0, 1)
        pattern = [el for el in range(seq['data'].ndim) if el != rows_idx and
                   el != cols_idx] + [rows_idx, cols_idx]
        inv_pattern = [pattern.index(el) for el in range(seq['data'].ndim)]
        seq['data'] = seq['data'].transpose(pattern)

        crop = list(crop_size)
        pad = [0, 0]
        h, w = seq['data'].shape[-2:]

        # Compute amounts
        if crop[0] < h:
            # Do random crop
            top = np.random.randint(h - crop[0])
        else:
            # Set pad and reset crop
            pad[0] = crop[0] - h
            top, crop[0] = 0, h
        if crop[1] < w:
            # Do random crop
            left = np.random.randint(w - crop[1])
        else:
            # Set pad and reset crop
            pad[1] = crop[1] - w
            left, crop[1] = 0, w

        # Cropping
        seq['data'] = seq['data'][..., top:top+crop[0], left:left+crop[1]]
        if seq['labels'] is not None and len(seq['labels']) > 0:
            seq['labels'] = seq['labels'].transpose(pattern)
            seq['labels'] = seq['labels'][..., top:top+crop[0],
                                          left:left+crop[1]]
        if return_optical_flow:
            flow = flow.transpose(pattern)
            flow = flow[..., top:top+crop[0], left:left+crop[1]]
        # Padding
        if pad != [0, 0]:
            pad_pattern = ((0, 0),) * (seq['data'].ndim - 2) + (
                (pad[0]//2, pad[0] - pad[0]//2),
                (pad[1]//2, pad[1] - pad[1]//2))
            seq['data'] = np.pad(seq['data'], pad_pattern, 'constant')
            seq['labels'] = np.pad(seq['labels'], pad_pattern, 'constant',
                                   constant_values=void_label)
            if return_optical_flow:
                flow = np.pad(flow, pad_pattern, 'constant')  # pad with zeros

        # Reshape to original shape
        seq['data'] = seq['data'].transpose(inv_pattern)
        if seq['labels'] is not None and len(seq['labels']) > 0:
            seq['labels'] = seq['labels'].transpose(inv_pattern)
        if return_optical_flow:
            flow = flow.transpose(inv_pattern)

    # Save augmented images
    if save_to_dir:
        import seaborn as sns
        fname = 'data_augm_{}.png'.format(np.random.randint(1e4))
        print ('Save to dir'.format(fname))
        cmap = sns.hls_palette(nclasses)
        save_img2(seq['data'], seq['labels'], os.path.join(save_to_dir, fname),
                  cmap, void_label, rows_idx, cols_idx, chan_idx)

    # Undo extra dim
    if seq['labels'] is not None and len(seq['labels']) > 0:
        seq['labels'] = seq['labels'][..., 0]

    if return_optical_flow:
        seq['flow'] = np.array(flow)


def cart2polar(x, y):
    '''Roughly equivalent to cv2.cartToPolar'''
    mag = np.sqrt(x**2 + y**2)
    ang = np.arctan2(y, x)  # note, in [-pi, pi]
    return mag, ang


def flow2rgb(flow, frame=None, return_vec_field=False, return_0_255=True):
    '''
    Convert optical flow to RGB image
    From:
    https://github.com/stefanoalletto/TransFlow/blob/master/
    flowToColor.py

    Parameters
    ----------
    flow: ndarray
        A 3D array with the X, Y displacement per pixel
    frame: ndarray
        An image, used to overlay the vector field if return_vec_field
        is True
    return_vec_field: bool
        If True an image with an overlay of the optical flow vector
        field will be returned. Otherwise an RGB image representation of
        the optical flow will be returned. Default: False
    return_0_255: bool
        If True the returned RGB optical flow will be in [0, 255],
        otherwise in [0, 1]. Ignored if return_vec_field is True.
    '''
    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise ValueError('The flow should be an array (x, y, c) with c '
                         'containing the 2D XY-displacement.')

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = maxv = -999.
    minu = minv = 999.
    maxrad = -1.

    maxu = max(maxu, np.max(u))
    minu = max(minu, np.max(u))
    maxv = max(maxv, np.max(v))
    minv = max(minv, np.max(v))
    rad = np.sqrt((u ** 2. + v ** 2.))
    maxrad = max(maxrad, np.max(rad))
    u = u / (maxrad + 1e-5)
    v = v / (maxrad + 1e-5)
    if return_vec_field:
        mag, _ = cart2polar(flow[..., 0], flow[..., 1])
        img = drawVectorField(frame, mag, rad)
    else:
        img = computeColor(u, v)
        if return_0_255:
            img = img.astype('uint8')
        else:
            img = img / 255.
    return img


def drawVectorField(frame, mag, rad):
    import cv2
    magnitude_hsv = np.zeros(shape=rad.shape + tuple([3]))
    magnitude_hsv[..., 2] = np.clip(mag, 0, 10) / 10.
    magnitude_rgb = cv2.cvtColor(np.uint8(magnitude_hsv * 255),
                                 cv2.COLOR_HSV2RGB)
    magnitude_rgb[..., 1] = 255

    white_background = np.ones_like(frame) * 255
    cv2.addWeighted(frame, 0.4, white_background, 0.6, 0, white_background)

    height = rad.shape[0]
    width = rad.shape[1]

    divisor = 12
    vector_length = 10

    for i in range(height / divisor):
        for j in range(width / divisor):
            y1 = i * divisor
            x1 = j * divisor
            vector_length = magnitude_hsv[y1, x1, 2] * 10
            dy = vector_length * np.sin(rad[y1, x1])
            dx = vector_length * np.cos(rad[y1, x1])
            x2 = int(x1 + dx)
            y2 = int(y1 + dy)
            x2 = np.clip(x2, 0, width)
            y2 = np.clip(y2, 0, height)
            arrow_color = magnitude_rgb[y1, x1].tolist()
            white_background = cv2.arrowedLine(
                white_background, (x1, y1), (x2, y2),
                arrow_color, 1, tipLength=0.4)
    return white_background


def computeColor(u, v):
    img = np.zeros((u.shape[0], u.shape[1], 3))
    # nanIdx = np.logical_or(np.isnan(u), np.isnan(v))
    # u[int(nanIdx)-1] = 0.
    # v[int(nanIdx)-1] = 0.
    colorwheel, ncols = makeColorwheel()
    rad = np.sqrt((u ** 2. + v ** 2.))
    a = np.arctan2((-v), (-u)) / np.pi
    fk = np.dot((a + 1.) / 2., ncols - 1.)
    # % -1~1 maped to 1~ncols
    k0 = np.floor(fk).astype(np.int32)
    # % 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in np.arange(colorwheel.shape[-1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.
        col1 = tmp[k1] / 255.
        col = (1. - f) * col0 + f * col1
        idx = rad <= 1.
        col[idx] = 1. - rad[idx] * (1. - col[idx])
        # % increase saturation with radius
        col[rad > 1] = col[rad > 1] * 0.75
        # % out of range
        img[:, :, i] = np.floor(255. * col)
    return img


def makeColorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY+YG+GC+CB+BM+MR
    colorwheel = np.zeros((int(ncols), 3))
    # % r g b
    col = 0
    # %RY
    colorwheel[0:RY, 0] = 255.
    colorwheel[0:RY, 1] = np.floor(255. * np.arange(0., RY) / RY)
    col = col + RY
    # %YG
    colorwheel[col:col+YG, 0] = 255. - np.floor(255. * np.arange(0., YG) / YG)
    colorwheel[col:col+YG, 1] = 255.
    col = col + YG
    # %GC
    colorwheel[col+0:col+GC, 1] = 255.
    colorwheel[col+0:col+GC, 2] = np.floor(255. * np.arange(0., GC) /
                                           GC)
    col = col + GC
    # %CB
    colorwheel[col+0:col+CB, 1] = 255. - np.floor(255. * np.arange(0., CB) /
                                                  CB)
    colorwheel[col+0:col+CB, 2] = 255.
    col = col + CB
    # %BM
    colorwheel[col+0:col+BM, 2] = 255.
    colorwheel[col+0:col+BM, 0] = np.floor(255. * np.arange(0., BM) /
                                           BM)
    col = col + BM
    # %MR
    colorwheel[col+0:col+MR, 2] = 255. - np.floor(255. * np.arange(0., MR) /
                                                  MR)
    colorwheel[col+0:col+MR, 0] = 255.
    return colorwheel, ncols
