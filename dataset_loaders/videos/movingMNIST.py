import numpy as np
import os
from scipy.ndimage.interpolation import zoom

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys


class MovingMNISTDataset(ThreadedDataset):
    # Adapted from
    # https://github.com/dbbert/dfn

    name = 'movingMNIST'
    # The number of *non void* classes
    non_void_nclasses = 10

    # A list of the ids of the void labels
    _void_labels = []

    data_shape = (64, 64, 1)

    # The dataset-wide statistics (either channel-wise or pixel-wise).
    # `extra/running_stats` contains utilities to compute them.
    # mean = []
    # std = []

    # A *dictionary* of the form 'class_id: (R, G, B)'. 'class_id' is
    # the class id in the original data.
    _cmap = {}

    # A *dictionary* of form 'class_id: label'. 'class_id' is the class
    # id in the original data
    _mask_labels = {}

    def __init__(self, which_set='train', nvids=50000,
                 image_size=64, num_digits=2, digits_sizes=[28, 28],
                 digits_speed=0.3, change_dir_prob=[0.5, 0.5],
                 rng=None, seed=1, *args, **kwargs):

        self.data_shape = (image_size,) * 2 + (self.data_shape[2],)

        self.which_set = 'validation' if which_set == 'valid' else which_set
        self.nvids = nvids
        self.image_size_ = image_size
        self.row_ = 0
        self.num_digits_ = num_digits
        self.num_channels_ = 1
        self.digits_sizes_ = digits_sizes
        self.num_digits_ = num_digits
        self.batch_size_ = 1
        self.set_has_GT = False
        self.seed = seed
        self.digits_speed = digits_speed
        self.change_dir_prob = change_dir_prob
        self.indices_ = range(nvids)
        self._rng = rng if rng else np.random.RandomState(seed)
        # self.path = self.shared_path
        import h5py
        try:
            f = h5py.File(os.path.join(self.path, 'mnist.h5'))
        except:
            raise RuntimeError('Failed to load dataset file')

        self.data_ = f[self.which_set].value[:nvids].reshape(-1, 28, 28)
        f.close()

        super(MovingMNISTDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        return {'default': range(self.nvids*self.seq_length)}

    def Overlap(self, a, b):
        """Put b on top of a."""
        return np.maximum(a, b)

    def get_random_trajectory(self, batch_size):
        length = self.seq_length
        canvas_size = self.image_size_ - np.max(self.digits_sizes_)

        # Initial position uniform random inside the box.
        y = self._rng.rand(batch_size)
        x = self._rng.rand(batch_size)
        # Choose speed.
        speed = self.digits_speed
        if isinstance(self.digits_speed, float):
            speed = [self.digits_speed for el in range(batch_size)]
        elif self.digits_speed == 'random':
            speed = self._rng.rand(batch_size)
        v_y = np.asarray(speed)  # np.sin(theta)
        v_x = np.asarray(speed)  # np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        # Random directions
        y_rand = np.asarray([self._rng.uniform(-1, 1) for _ in
                             range(batch_size)])
        x_rand = np.asarray([self._rng.uniform(-1, 1) for _ in
                             range(batch_size)])

        # For each digit for each frame compute the decision to change
        # direction following a binomial distribution
        change_dir = []
        for i in range(self.num_digits_):
            sampling = [self._rng.binomial(
                1, self.change_dir_prob[i]) for _ in range(length)]
            change_dir.append(sampling)
        change_dir = zip(*change_dir)

        for i in xrange(length):

            # Which digits change the direction in this frame
            c_digits = np.where(change_dir[i])
            # Number of digits changing direction in this frame
            n_digits = len(c_digits)
            np.put(y_rand, c_digits,
                   [self._rng.uniform(-1, 1) for _ in range(n_digits)])
            np.put(x_rand, c_digits,
                   [self._rng.uniform(-1, 1) for _ in range(n_digits)])
            # Take a step along velocity.
            y += v_y * y_rand  # + self.step_length_
            x += v_x * x_rand  # + self.step_length_
            # Bounce off edges.
            for j in xrange(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def get_batch(self, verbose=False):
        start_y, start_x = self.get_random_trajectory(
            self.batch_size_ * self.num_digits_)

        # minibatch data
        # if self.background_ == 'zeros':
        data = np.zeros((self.batch_size_, self.num_channels_,
                         self.image_size_, self.image_size_,
                         self.seq_length), dtype=np.float32)
        # elif self.background_ == 'rand':
        #     data = self._rng.rand(self.batch_size_, self.num_channels_,
        #                          self.image_size_, self.image_size_,
        #                          self.seq_length)

        for j in xrange(self.batch_size_):
            for n in xrange(self.num_digits_):

                # get random digit from dataset
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self._rng.shuffle(self.indices_)
                    self.row_ = 0
                digit_image = self.data_[ind, :, :]
                zoom_factor = 1
                if self.digits_sizes_ != 28:
                    zoom_factor = int(self.digits_sizes_[n]/28)
                    digit_image = zoom(digit_image, zoom_factor)
                digit_size = digit_image.shape[0]

                # if self.mode_ == 'squares':
                #     digit_size = self._rng.randint(5, 20)
                #     digit_image = np.ones((digit_size, digit_size),
                #                           dtype=np.float32)

                # generate video
                for i in xrange(self.seq_length):
                    top = start_y[i, j * self.num_digits_ + n]
                    left = start_x[i, j * self.num_digits_ + n]
                    bottom = top + digit_size
                    right = left + digit_size
                    data[j, :, top:bottom, left:right, i] = self.Overlap(
                        data[j, :, top:bottom, left:right, i], digit_image)

        data = np.transpose(data, (0, 4, 2, 3, 1))
        return data[0]

    def _reset(self, *args, **kwargs):
        if self.which_set != 'train':
            self._rng = np.random.RandomState(self.seed)
            self.row_ = 0
            self.indices_ = range(self.nvids)

    def _fill_names_batches(self, *args, **kwargs):
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
        batch = self.get_batch()
        X = batch
        # Return last frame as label
        Y = batch[self.seq_length-1]
        F = self.indices_

        ret = {}
        ret['data'] = X
        ret['labels'] = np.asarray(Y)
        ret['subset'] = []
        ret['filenames'] = np.array(F)
        return ret
