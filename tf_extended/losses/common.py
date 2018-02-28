# =========================================================================== #
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
# =========================================================================== #
"""Misc. collection of losses.
"""
import tensorflow as tf
from tensorflow.python.framework import ops

# =========================================================================== #
# ACommon loss methods.
# =========================================================================== #
def sparse_softmax_cross_entropy(
        labels,
        logits,
        weights=1.0,
        label_smoothing=0.0,
        scope=None,
        loss_collection=ops.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.
    `weights` acts as a coefficient for the loss. If a scalar is provided,
    then the loss is simply scaled by the given value. If `weights` is a
    tensor of shape [`batch_size`], then the loss weights apply to each
    corresponding sample.
    Args:
        labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
        `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
        must be an index in `[0, num_classes)`. Other values will raise an
        exception when this op is run on CPU, and return `NaN` for corresponding
        loss and gradient rows on GPU.
        logits: Unscaled log probabilities of shape
        `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float32` or `float64`.
        weights: Coefficients for the loss. This must be scalar or broadcastable to
        `labels` (i.e. same rank and each dimension is either 1 or the same).
        scope: the scope for the operations performed in computing the loss.
        loss_collection: collection to which the loss will be added.
        reduction: Type of reduction to apply to loss.
    Returns:
        Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
        `NONE`, this has the same shape as `labels`; otherwise, it is scalar.
    Raises:
        ValueError: If the shapes of `logits`, `labels`, and `weights` are
        incompatible, or if any of them are None.
    """
    if labels is None:
        raise ValueError("labels must not be None.")
    if logits is None:
        raise ValueError("logits must not be None.")
    with tf.name_scope(scope, "sparse_softmax_cross_entropy_loss",
                       (logits, labels, weights)) as scope:
        # labels, logits, weights = _remove_squeezable_dimensions(
        #     labels, logits, weights, expected_rank_diff=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name="xentropy")
        loss = tf.losses.compute_weighted_loss(
            loss, weights, scope, loss_collection, reduction=reduction)

        # Label smoothing.
        smooth_loss = 0.
        if label_smoothing > 0:
            loss = tf.scalar_mul(1. - label_smoothing, loss)
            aux_log_softmax = -tf.nn.log_softmax(logits)
            # Label smoothing loss: sum of logits * weight.
            smooth_loss = tf.losses.compute_weighted_loss(
                aux_log_softmax, label_smoothing * weights,
                'label_smoothing', loss_collection, reduction=reduction)

        return loss + smooth_loss
