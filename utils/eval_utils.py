import numpy as np
import sklearn.metrics
import tensorflow as tf
# tf.enable_eager_execution()


def weighted_miou_sk(y_true, y_pred, num_classes, weights=None):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    iou = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        iou.append(tp / (tp + fp + fn))
    if weights is None:
        return np.mean(iou)
    else:
        return np.dot(weights, iou)


def _streaming_confusion_matrix(y_true, y_pred, num_classes):
    total_cm = tf.get_variable(
        name='total_confusion_matrix',
        shape=[num_classes, num_classes],
        dtype=tf.float64,
        initializer=tf.zeros_initializer(),
        collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.SUM)
    current_cm = tf.confusion_matrix(y_true, y_pred, num_classes, dtype=tf.float64)
    update_op = tf.assign_add(total_cm, current_cm)
    return total_cm, update_op


def weighted_miou(y_true, y_pred, num_classes, weights=None):
    # cm = tf.confusion_matrix(y_true, y_pred, num_classes)
    # iou = []
    # for c in range(num_classes):
    #     tp = cm[c, c]
    #     fn = tf.reduce_sum(cm[c, :]) - tp
    #     fp = tf.reduce_sum(cm[:, c]) - tp
    #     iou.append(tp / (tp + fp + fn))
    # if weights is None:
    #     return tf.reduce_mean(iou)
    # else:
    #     return tf.reduce_sum(tf.multiply(weights, iou))
    with tf.variable_scope('mean_iou'):
        total_cm, update_op = _streaming_confusion_matrix(y_true, y_pred, num_classes)
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.linalg.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(tf.cast(tf.not_equal(denominator, 0), dtype=tf.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(tf.greater(denominator, 0), denominator, tf.ones_like(denominator))
        iou = tf.div(cm_diag, denominator)

        # If the number of valid entries is 0 (no classes) we return 0.
        if weights is None:
            result = tf.where(tf.greater(num_valid_entries, 0), tf.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
        else:
            result = tf.where(tf.greater(num_valid_entries, 0), tf.reduce_sum(tf.multiply(weights, iou), name='mean_iou'), 0)
        return result, update_op


if __name__ == '__main__':
    y_true_ = [2, 0, 2, 2, 0, 1]
    y_pred_ = [0, 0, 2, 2, 0, 2]
    weights = [0.2, 0.5, 0.1]
    num_classes = 3

    # sk
    miou_sk = weighted_miou_sk(y_true_, y_pred_, num_classes, weights)

    # tf orig
    # with tf.Graph().as_default():
    #     y_true = tf.constant(y_true_, dtype=tf.float32)
    #     y_pred = tf.constant(y_pred_, dtype=tf.float32)
    #     m, u = tf.metrics.mean_iou(y_true, y_pred, num_classes)
    #     with tf.device('cpu:0'):
    #         with tf.Session() as sess:
    #             sess.run(tf.global_variables_initializer())
    #             sess.run(tf.local_variables_initializer())
    #             sess.run(u)
    #             miou_tf_orig = sess.run(m)

    # tf me
    with tf.Graph().as_default():
        y_true = tf.constant(y_true_, dtype=tf.float32)
        y_pred = tf.constant(y_pred_, dtype=tf.float32)
        m, update_op = weighted_miou(y_true, y_pred, num_classes, weights)
        with tf.device('cpu:0'):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                sess.run(update_op)
                miou_tf_me = sess.run(m)

    # res
    print('sklearn:', miou_sk)
    # print('tf orig:', miou_tf_orig)
    print('tf me:  ', miou_tf_me)
