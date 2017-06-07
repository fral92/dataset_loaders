import numpy as np
import os
from scipy.ndimage.interpolation import zoom

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys


class MovingMNISTDataset(ThreadedDataset):
    # Adapted from
    # https://github.com/dbbert/dfn

    """ The moving MNIST dataset

    Parameters
    ----------
    which_set: string
        A string in ['train', 'valid', 'test'], corresponding to the set
        to be returned.
    nvids: int
        The number of video sequences to be generated for this set
    frame_size: int
        The size of the frames
    num_digits: int
        The number of moving digits in the video sequence
    digits_sizes: list of int
        A list of sizes for each digit
    background: string
        If is 'zeros' the background will have zero values, otherwhise
        it is initialized randomly
    digits_speed: float, string or list of float
        The digits speed. If it is a float instance then the same speed
        is applied to each digit. If it is 'random' then a random speed
        is generated for each digit
    change_dir_prob: list of float
        The probability for each digit to change direction between two
        consecutive frames
    """
    name = 'movingMNIST'
    # The number of *non void* classes
    non_void_nclasses = None

    # A list of the ids of the void labels
    _void_labels = []

    data_shape = (64, 64, 1)

    def __init__(self, which_set='train', nvids=1000, image_size=64,
                 num_digits=1, digits_sizes=[28, 28], background='zeros',
                 digits_speed=0.3, change_dir_prob=[0., 0.], rng=None,
                 seed=1, *args, **kwargs):

        self.data_shape = (image_size, image_size, 1)

        self.which_set = 'validation' if which_set == 'valid' else which_set
        self.nvids = nvids
        self.frame_size = image_size
        self.curr_data_idx = 0
        self.num_digits = num_digits
        self.digits_sizes = digits_sizes
        self.background = background
        self.set_has_GT = False
        self.seed = seed
        self.digits_speed = digits_speed
        self.change_dir_prob = change_dir_prob
        self.vids_indices = range(nvids)
        self._rng = rng if rng else np.random.RandomState(seed)
        # self.path = self.shared_path
        import h5py
        try:
            f = h5py.File(os.path.join(self.path, 'mnist.h5'))
        except:
            raise RuntimeError('Failed to load dataset file from: %s' %
                               os.path.join(self.path, 'mnist.h5'))

        self._MNIST_data = f[self.which_set].value[:nvids].reshape(-1, 28, 28)
        f.close()

        super(MovingMNISTDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset.
           Note: The names will be ignored in load_sequence."""
        return {'default': [None for _ in range(self.nvids * self.seq_length)]}

    def _get_random_trajectory(self):
        # Here add one since the frame after the sequence will be used
        # as target
        ext_seq_length = self.seq_length + 1
        canvas_size = self.frame_size - np.max(self.digits_sizes)

        # Initial position uniform random inside the box.
        y = self._rng.rand(self.num_digits)
        x = self._rng.rand(self.num_digits)
        # Choose speed.
        speed = self.digits_speed
        if isinstance(self.digits_speed, float):
            speed = [self.digits_speed for el in range(self.num_digits)]
        elif self.digits_speed == 'random':
            speed = self._rng.rand(self.num_digits)
        v_y = np.asarray(speed)  # np.sin(theta)
        v_x = np.asarray(speed)  # np.cos(theta)

        start_y = np.zeros((ext_seq_length, self.num_digits))
        start_x = np.zeros((ext_seq_length, self.num_digits))
        # Random directions
        y_rand = np.asarray([self._rng.uniform(-1, 1) for _ in
                             range(self.num_digits)])
        x_rand = np.asarray([self._rng.uniform(-1, 1) for _ in
                             range(self.num_digits)])

        # For each digit for each frame compute the decision to change
        # direction following a binomial distribution
        change_dir = []
        for i in range(self.num_digits):
            sampling = [self._rng.binomial(1, self.change_dir_prob[i])
                        for _ in range(ext_seq_length)]
            change_dir.append(sampling)
        change_dir = zip(*change_dir)

        for frame_id in range(ext_seq_length):

            # Which digits change the direction in this frame
            c_digits = np.where(change_dir[frame_id])
            # Number of digits changing direction in this frame
            n_c_digits = len(c_digits)
            np.put(y_rand, c_digits,
                   [self._rng.uniform(-1, 1) for _ in range(n_c_digits)])
            np.put(x_rand, c_digits,
                   [self._rng.uniform(-1, 1) for _ in range(n_c_digits)])
            # Take a step along velocity.
            y += v_y * y_rand  # + self.step_length_
            x += v_x * x_rand  # + self.step_length_
            # Bounce off edges.
            for j in range(self.num_digits):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                elif x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                elif y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[frame_id] = y
            start_x[frame_id] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def _get_sequence(self, verbose=False):
        start_y, start_x = self._get_random_trajectory()

        # minibatch data
        if self.background == 'zeros':
            out_sequence = np.zeros((self.seq_length+1, self.frame_size,
                                     self.frame_size, 1), dtype=np.float32)
        elif self.background == 'rand':
            out_sequence = self._rng.rand(self.seq_length+1, self.frame_size,
                                          self.frame_size, 1)

        for digit_id in range(self.num_digits):

            # get random digit from dataset
            ind = self.vids_indices[self.curr_data_idx]
            self.curr_data_idx += 1
            if self.curr_data_idx == self._MNIST_data.shape[0]:
                self._rng.shuffle(self.vids_indices)
                self.curr_data_idx = 0
            digit_image = self._MNIST_data[ind]
            zoom_factor = 1
            if self.digits_sizes != 28:
                zoom_factor = int(self.digits_sizes[digit_id]/28)
                digit_image = zoom(digit_image, zoom_factor)
            digit_size = digit_image.shape[0]

            # if self.mode_ == 'squares':
            #     digit_size = self._rng.randint(5, 20)
            #     digit_image = np.ones((digit_size, digit_size),
            #                           dtype=np.float32)

            # generate video
            digit_image = np.expand_dims(digit_image, -1)
            for i in range(self.seq_length+1):
                top = start_y[i, digit_id]
                left = start_x[i, digit_id]
                bottom = top + digit_size
                right = left + digit_size
                out_sequence[i, top:bottom, left:right, :] = np.maximum(
                    out_sequence[i, top:bottom, left:right, :], digit_image)
        return out_sequence

    def _reset(self, *args, **kwargs):
        """ This function is used to reset the rng and data index from
        where the digits are sampled to create the video sequence
        """
        if self.which_set != 'train':
            self._rng = np.random.RandomState(self.seed)
            self.curr_data_idx = 0
            self.vids_indices = range(self.nvids)

    def _fill_names_batches(self, *args, **kwargs):
        # Reset the digit generator at the end of the epoch
        self._reset(self, *args, **kwargs)
        super(MovingMNISTDataset, self)._fill_names_batches(*args, **kwargs)

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        sequence = self._get_sequence()
        X = sequence[:self.seq_length]
        # Return last frame as target
        Y = sequence[self.seq_length]
        F = self.vids_indices

        ret = {}
        ret['data'] = X
        ret['labels'] = np.asarray(Y[np.newaxis, ...])
        ret['subset'] = []
        ret['filenames'] = np.array(F)
        return ret
