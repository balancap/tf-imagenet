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
"""ImageNet pre-processing.
"""
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# from deploy import trainer_utils

# from tensorflow.contrib.data.python.ops import interleave_ops
# from tensorflow.contrib.data.python.ops import batching
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile

from .utils import parse_example_proto
from .image import get_image_resize_method, distort_color

try:
    from tensorflow.contrib.data.python.ops import interleave_ops
    from tensorflow.contrib.data.python.ops import batching
except ImportError:
    from tf_extended.legacy import interleave_ops
    from tf_extended.legacy import batching
    print('WARNING: TF without interleave_ops module.')


def eval_image(image,
               height,
               width,
               batch_position,
               resize_method,
               summary_verbosity=0):
    """Get the image for model evaluation.

    We preprocess the image simiarly to Slim, see
    https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
    Validation images do not have bounding boxes, so to crop the image, we first
    resize the image such that the aspect ratio is maintained and the resized
    height and width are both at least 1.15 times `height` and `width`
    respectively. Then, we do a central crop to size (`height`, `width`).

    TODO(b/64579165): Determine if we should use different evaluation
    prepossessing steps.

    Args:
        image: 3-D float Tensor representing the image.
        height: The height of the image that will be returned.
        width: The width of the image that will be returned.
        batch_position: position of the image in a batch, which affects how images
            are distorted and resized. NOTE: this argument can be an integer or a
            tensor
        resize_method: one of the strings 'round_robin', 'nearest', 'bilinear',
            'bicubic', or 'area'.
        summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
            summaries and checkpoints.
    Returns:
        An image of size (output_height, output_width, 3) that is resized and
        cropped as described above.
    """
    # TODO(reedwm): Currently we resize then crop. Investigate if it's faster to
    # crop then resize.
    with tf.name_scope('eval_image'):
        if summary_verbosity >= 3:
            tf.summary.image(
                'original_image', tf.expand_dims(image, 0))

        shape = tf.shape(image)
        image_height = shape[0]
        image_width = shape[1]
        image_height_float = tf.cast(image_height, tf.float32)
        image_width_float = tf.cast(image_width, tf.float32)

        scale_factor = 1.15

        # Compute resize_height and resize_width to be the minimum values such that
        #   1. The aspect ratio is maintained (i.e. resize_height / resize_width is
        #      image_height / image_width), and
        #   2. resize_height >= height * `scale_factor`, and
        #   3. resize_width >= width * `scale_factor`
        max_ratio = tf.maximum(height / image_height_float,
                               width / image_width_float)
        resize_height = tf.cast(image_height_float * max_ratio * scale_factor,
                                tf.int32)
        resize_width = tf.cast(image_width_float * max_ratio * scale_factor,
                               tf.int32)

        # Resize the image to shape (`resize_height`, `resize_width`)
        image_resize_method = get_image_resize_method(resize_method, batch_position)
        distorted_image = tf.image.resize_images(image,
                                                 [resize_height, resize_width],
                                                 image_resize_method,
                                                 align_corners=False)

        # Do a central crop of the image to size (height, width).
        total_crop_height = (resize_height - height)
        crop_top = total_crop_height // 2
        total_crop_width = (resize_width - width)
        crop_left = total_crop_width // 2
        distorted_image = tf.slice(distorted_image, [crop_top, crop_left, 0],
                                   [height, width, 3])

        distorted_image.set_shape([height, width, 3])
        if summary_verbosity >= 3:
            tf.summary.image(
                'cropped_resized_image', tf.expand_dims(distorted_image, 0))
        image = distorted_image
    return image


def train_image(image_buffer,
                height,
                width,
                bbox,
                batch_position,
                resize_method,
                distortions,
                scope=None,
                summary_verbosity=0,
                distort_color_in_yiq=False,
                fuse_decode_and_crop=False):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
        image_buffer: scalar string Tensor representing the raw JPEG image buffer.
        height: integer
        width: integer
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax].
        batch_position: position of the image in a batch, which affects how images
            are distorted and resized. NOTE: this argument can be an integer or a
            tensor
        resize_method: round_robin, nearest, bilinear, bicubic, or area.
        distortions: If true, apply full distortions for image colors.
        scope: Optional scope for op_scope.
        summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
            summaries and checkpoints.
        distort_color_in_yiq: distort color of input images in YIQ space.
        fuse_decode_and_crop: fuse the decode/crop operation.
    Returns:
        3-D float Tensor of distorted image used for training.
    """
    # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
    # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    with tf.name_scope(scope or 'distort_image'):
        # A large fraction of image datasets contain a human-annotated bounding box
        # delineating the region of the image containing the object of interest.  We
        # choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_buffer),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if summary_verbosity >= 4:
            image = tf.image.decode_jpeg(image_buffer, channels=3,
                                         dct_method='INTEGER_FAST')
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image(
                'images_with_distorted_bounding_box',
                image_with_distorted_box)

        # Crop the image to the specified bounding box.
        if fuse_decode_and_crop:
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
            image = tf.image.decode_and_crop_jpeg(
                image_buffer, crop_window, channels=3)
        else:
            image = tf.image.decode_jpeg(image_buffer, channels=3,
                                         dct_method='INTEGER_FAST')
            image = tf.slice(image, bbox_begin, bbox_size)

        if distortions:
            # After this point, all image pixels reside in [0,1]. Before, they were
            # uint8s in the range [0, 255].
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected.
        image_resize_method = get_image_resize_method(resize_method, batch_position)
        distorted_image = tf.image.resize_images(
            image, [height, width],
            image_resize_method,
            align_corners=False)

        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if summary_verbosity >= 4:
            tf.summary.image(
                'cropped_resized_image',
                tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        if distortions:
            # Randomly distort the colors.
            distorted_image = distort_color(distorted_image, batch_position,
                                            distort_color_in_yiq=distort_color_in_yiq)

            # Note: This ensures the scaling matches the output of eval_image
            distorted_image *= 255

        if summary_verbosity >= 4:
            tf.summary.image(
                'final_distorted_image',
                tf.expand_dims(distorted_image, 0))
        return distorted_image


class RecordInputImagePreprocessor(object):
    """Preprocessor for images with RecordInput format.
    """
    def __init__(self,
                 height,
                 width,
                 batch_size,
                 num_splits,
                 dtype,
                 train,
                 distortions,
                 resize_method,
                 shift_ratio,
                 summary_verbosity,
                 distort_color_in_yiq,
                 fuse_decode_and_crop):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.dtype = dtype
        self.train = train
        self.resize_method = resize_method
        self.shift_ratio = shift_ratio
        self.distortions = distortions
        self.distort_color_in_yiq = distort_color_in_yiq
        self.fuse_decode_and_crop = fuse_decode_and_crop
        if self.batch_size % self.num_splits != 0:
            raise ValueError(
                ('batch_size must be a multiple of num_splits: '
                 'batch_size %d, num_splits: %d') %
                (self.batch_size, self.num_splits))
        self.batch_size_per_split = self.batch_size // self.num_splits
        self.summary_verbosity = summary_verbosity

    def preprocess(self, image_buffer, bbox, batch_position):
        """Preprocessing image_buffer as a function of its batch position."""
        if self.train:
            image = train_image(image_buffer, self.height, self.width, bbox,
                                batch_position, self.resize_method, self.distortions,
                                None, summary_verbosity=self.summary_verbosity,
                                distort_color_in_yiq=self.distort_color_in_yiq,
                                fuse_decode_and_crop=self.fuse_decode_and_crop)
        else:
            image = tf.image.decode_jpeg(
                image_buffer, channels=3, dct_method='INTEGER_FAST')
            image = eval_image(image, self.height, self.width, batch_position,
                               self.resize_method,
                               summary_verbosity=self.summary_verbosity)
        # Note: image is now float32 [height,width,3] with range [0, 255]

        # image = tf.cast(image, tf.uint8) # HACK TESTING

        return image

    def parse_and_preprocess(self, value, batch_position):
        image_buffer, label_index, bbox, _ = parse_example_proto(value)
        image = self.preprocess(image_buffer, bbox, batch_position)
        return (label_index, image)

    def minibatch(self, dataset, subset, use_datasets, cache_data,
                  shift_ratio=-1):
        if shift_ratio < 0:
            shift_ratio = self.shift_ratio
        with tf.name_scope('batch_processing'):
            # Build final results per split.
            images = [[] for _ in range(self.num_splits)]
            labels = [[] for _ in range(self.num_splits)]
            if use_datasets:
                glob_pattern = dataset.tf_record_pattern(subset)
                file_names = gfile.Glob(glob_pattern)
                if not file_names:
                    raise ValueError('Found no files in --data_dir matching: {}'
                                     .format(glob_pattern))
                ds = tf.data.TFRecordDataset.list_files(file_names)
                ds = ds.apply(
                    interleave_ops.parallel_interleave(
                        tf.data.TFRecordDataset, cycle_length=10))
                if cache_data:
                    ds = ds.take(1).cache().repeat()
                counter = tf.data.Dataset.range(self.batch_size)
                counter = counter.repeat()
                ds = tf.data.Dataset.zip((ds, counter))
                ds = ds.prefetch(buffer_size=self.batch_size)
                ds = ds.shuffle(buffer_size=10000)
                ds = ds.repeat()
                ds = ds.apply(
                    batching.map_and_batch(
                        map_func=self.parse_and_preprocess,
                        batch_size=self.batch_size_per_split,
                        num_parallel_batches=self.num_splits))
                ds = ds.prefetch(buffer_size=self.num_splits)
                ds_iterator = ds.make_one_shot_iterator()
                for d in xrange(self.num_splits):
                    labels[d], images[d] = ds_iterator.get_next()

            else:
                record_input = data_flow_ops.RecordInput(
                    file_pattern=dataset.tf_record_pattern(subset),
                    seed=301,
                    parallelism=64,
                    buffer_size=10000,
                    batch_size=self.batch_size,
                    shift_ratio=shift_ratio,
                    name='record_input')
                records = record_input.get_yield_op()
                records = tf.split(records, self.batch_size, 0)
                records = [tf.reshape(record, []) for record in records]
                for idx in xrange(self.batch_size):
                    value = records[idx]
                    (label, image) = self.parse_and_preprocess(value, idx)
                    split_index = idx % self.num_splits
                    labels[split_index].append(label)
                    images[split_index].append(image)

            for split_index in xrange(self.num_splits):
                if not use_datasets:
                    images[split_index] = tf.parallel_stack(images[split_index])
                    labels[split_index] = tf.concat(labels[split_index], 0)
                images[split_index] = tf.cast(images[split_index], self.dtype)
                depth = 3
                images[split_index] = tf.reshape(
                    images[split_index],
                    shape=[self.batch_size_per_split, self.height, self.width, depth])
                labels[split_index] = tf.reshape(labels[split_index],
                                                 [self.batch_size_per_split])
            return images, labels
