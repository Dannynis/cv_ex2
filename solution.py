"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""

        as_strided = np.lib.stride_tricks.as_strided

        def pad_image_one_side(img, amount, side):
            if side == 'l':
                amount_padding = (amount, 0)
            else:
                amount_padding = (0, amount)
            img = np.stack(
                [np.pad(img[:, :, c], ((0, 0), amount_padding), mode='constant', constant_values=0) for c in
                 range(img.shape[-1])],
                axis=2)
            return img, amount_padding

        def pad_image(img, pad, amount_padding):
            x_pad = np.array((pad, pad)) - np.array(amount_padding).clip(0, None)
            x_pad_cliped = np.clip(x_pad, a_min=0, a_max=None)
            img = np.pad(img, ((pad, pad), x_pad_cliped), mode='constant', constant_values=0)
            return img, x_pad

        def disperity_shift_diff(left_image, right_image, dsp):
            if dsp > 0:
                left_image_shifted, amount_padding = pad_image_one_side(left_image, dsp, 'l')
                right_image_shifted, _ = pad_image_one_side(right_image, dsp, 'r')
            elif dsp < 0:
                left_image_shifted, amount_padding = pad_image_one_side(left_image, -1 * dsp, 'r')
                right_image_shifted, _ = pad_image_one_side(right_image, -1 * dsp, 'l')
            else:
                left_image_shifted, right_image_shifted = left_image, right_image
                amount_padding = (0, 0)
            diff = (left_image_shifted - right_image_shifted) ** 2
            diff = diff.sum(axis=-1)
            return diff, amount_padding

        def strided_view(A, kernel_size, stride):
            output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                            (A.shape[1] - kernel_size) // stride + 1)

            shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
            strides_w = (stride * A.strides[0], stride * A.strides[1], A.strides[0], A.strides[1])

            A_w = as_strided(A, shape_w, strides_w)
            return A_w

        def calc_disperity(left_image, right_image, win_size, dsp):
            padding_size = int(np.floor(win_size / 2))
            diff, amount_padding = disperity_shift_diff(left_image, right_image, dsp)
            padded_diff, padding = pad_image(diff, padding_size, amount_padding)
            l, r = np.clip(padding, None, 0) * -1
            sum_pool = strided_view(padded_diff, win_size, 1).sum(axis=(2, 3))
            sum_pool = sum_pool[:, l:sum_pool.shape[1] - r]
            return sum_pool

        ssdd_tensor = np.stack([calc_disperity(left_image, right_image, win_size, d) for d in disparity_values], axis=-1)

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        """INSERT YOUR CODE HERE"""
        label_no_smooth = ssdd_tensor.argmin(axis=-1)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
        l_slice[:, 0] = c_slice[:, 0]
        b_indexs = np.arange(1, l_slice.shape[0] + 1)

        l_slice = np.pad(l_slice, ((1, 1), (0, 0)), mode='constant', constant_values=np.inf)

        def fill_L(k):
            a = l_slice[:, k - 1][b_indexs]
            b = np.min((l_slice[:, k - 1][b_indexs + 1], l_slice[:, k - 1][b_indexs - 1]), axis=0) + p1
            c_min_val = np.ones_like(b_indexs) * l_slice[:, k - 1].min()
            c = c_min_val + p2
            l_slice[1:-1, k] = np.min((a, b, c), axis=0) + c_slice[:, k] - c_min_val

        for k in range(1, l_slice.shape[1]):
            fill_L(k)

        l_slice = l_slice[1:-1, :]

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        for i in range(ssdd_tensor.shape[0]):
            c_slice = ssdd_tensor[i, :, :].T
            L = self.dp_grade_slice(c_slice, p1, p2)
            l[i] = L.T

        return self.naive_labeling(l)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""

        def flipper(x):
            return [np.fliplr(x) for x in x]

        def diagonolize(ssdd):
            longer_axis = max(ssdd.shape[1], ssdd.shape[0])
            diags = [ssdd.diagonal(i) for i in range(-longer_axis, longer_axis)]
            return diags

        def undiagonlize(orig_ssdd_shape, diags):
            xv, yv = np.meshgrid(range(orig_ssdd_shape[1]), range(orig_ssdd_shape[0]), sparse=False, indexing='ij')
            inds = np.stack([xv, yv]).T
            longer_axis = max(orig_ssdd_shape)
            diags_inds = np.concatenate([inds.diagonal(i) for i in range(-longer_axis, longer_axis)], axis=1)
            diags_c = np.concatenate(diags, axis=1)
            z = np.zeros(orig_ssdd_shape)
            z[diags_inds[1], diags_inds[0]] = diags_c.T
            return z

        def directions_slices(ssdd):
            d1 = [ssdd[i].T for i in range(ssdd.shape[0])]
            d2 = diagonolize(ssdd)
            d3 = [ssdd[:, i].T for i in range(ssdd.shape[1])]
            d4 = diagonolize(np.fliplr(ssdd))
            d5, d6, d7, d8 = map(flipper, [d1, d2, d3, d4])
            return [d1, d2, d3, d4, d5, d6, d7, d8]

        graded_direction = []
        for direction_slices in directions_slices(ssdd_tensor):
            graded_slices = []
            for c_slice in direction_slices:
                if c_slice.shape[1] > 0:
                    graded_slices.append(self.dp_grade_slice(c_slice, p1, p2))
            graded_direction.append(graded_slices)

        d1_l = np.stack(list(map(np.transpose, graded_direction[0])))
        d2_l = undiagonlize(ssdd_tensor.shape, graded_direction[1])
        d3_l = np.stack(list(map(np.transpose, graded_direction[2]))).transpose(1, 0, 2)
        d4_l = np.fliplr(undiagonlize(ssdd_tensor.shape, graded_direction[3]))
        d5_l = np.fliplr(list(map(np.transpose, graded_direction[4])))
        d6_l = undiagonlize(ssdd_tensor.shape, flipper(graded_direction[5]))
        d7_l = np.stack(list(map(np.transpose, flipper(graded_direction[6])))).transpose(1, 0, 2)
        d8_l = np.fliplr(undiagonlize(ssdd_tensor.shape, flipper(graded_direction[7])))

        l = np.mean([d1_l,d2_l,d3_l,d4_l,d5_l,d6_l,d7_l,d8_l],axis=0)

        return self.naive_labeling(l)

