# /* ===========================================================================
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# =========================================================================== */
"""Image related functions.
"""
import math
import tensorflow as tf

from tensorflow.python.layers import utils
from tensorflow.contrib.image.python.ops import distort_image_ops

def get_image_resize_method(resize_method, batch_position=0):
    """Get tensorflow resize method.

    If method is 'round_robin', return different methods based on batch position
    in a round-robin fashion. NOTE: If the batch size is not a multiple of the
    number of methods, then the distribution of methods will not be uniform.

    Args:
        resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
        batch_position: position of the image in a batch. NOTE: this argument can
            be an integer or a tensor
    Returns:
        one of resize type defined in tf.image.ResizeMethod.
    """
    resize_methods_map = {
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA
    }

    if resize_method != 'round_robin':
        return resize_methods_map[resize_method]

    # return a resize method based on batch position in a round-robin fashion.
    resize_methods = resize_methods_map.values()
    def lookup(index):
        return resize_methods[index]

    def resize_method_0():
        return utils.smart_cond(batch_position % len(resize_methods) == 0,
                                lambda: lookup(0), resize_method_1)

    def resize_method_1():
        return utils.smart_cond(batch_position % len(resize_methods) == 1,
                                lambda: lookup(1), resize_method_2)

    def resize_method_2():
        return utils.smart_cond(batch_position % len(resize_methods) == 2,
                                lambda: lookup(2), lambda: lookup(3))

    # NOTE(jsimsa): Unfortunately, we cannot use a single recursive function here
    # because TF would not be able to construct a finite graph.
    return resize_method_0()

def distort_color(image, batch_position=0, distort_color_in_yiq=False,
                  scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops based on the position of the image in a batch.

    Args:
        image: float32 Tensor containing single image. Tensor values should be in
            range [0, 1].
        batch_position: the position of the image in a batch. NOTE: this argument
            can be an integer or a tensor
        distort_color_in_yiq: distort color of input images in YIQ space.
        scope: Optional scope for op_scope.
    Returns:
        color-distorted image
    """
    with tf.name_scope(scope or 'distort_color'):

        def distort_fn_0(image=image):
            """Variant 0 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            if distort_color_in_yiq:
                image = distort_image_ops.random_hsv_in_yiq(
                    image, lower_saturation=0.5, upper_saturation=1.5,
                    max_delta_hue=0.2 * math.pi)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image

        def distort_fn_1(image=image):
            """Variant 1 of distort function."""
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            if distort_color_in_yiq:
                image = distort_image_ops.random_hsv_in_yiq(
                    image, lower_saturation=0.5, upper_saturation=1.5,
                    max_delta_hue=0.2 * math.pi)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            return image

        image = utils.smart_cond(batch_position % 2 == 0, distort_fn_0,
                                 distort_fn_1)
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
