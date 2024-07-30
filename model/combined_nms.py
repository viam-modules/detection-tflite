import tensorflow as tf
from keras_cv import bounding_box


class CombinedNMS(tf.keras.layers.Layer):
    def __init__(
        self,
        from_logits,
        num_classes,
        src_bounding_box_format,
        iou_threshold=0.35,
        confidence_threshold=0,
        max_detections_per_class=32,
        max_total_detections=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.num_classes = num_classes
        self.src_bounding_box_format = src_bounding_box_format
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_total_detections = max_total_detections
        # must be named bounding_box_format in order to be a valid prediction decoder
        # rel_yxyx is the expected bounding box format in RDK
        self.bounding_box_format = "rel_yxyx"

    def call(self, box_prediction, class_predictions, image_shape):
        """Accepts box predictions and class predictions, and returns filtered bounding box
        predictions.

        Args:
            box_prediction: tensor of shape [batch, boxes, 4]
            class_predictions: tensor of shape [batch, boxes, num_classes] with per class scores from base model
        """
        # Reshape predictions we can use the NMS function
        box_prediction = tf.reshape(box_prediction, [-1, 4])
        class_predictions = tf.reshape(class_predictions, [-1, self.num_classes])

        def nms_filtering_indices(scores):
            # Based on the top scores, filter scores below the confidence threshold,
            # Then, perform non-max suppression
            indices = tf.where(
                tf.keras.backend.greater(scores, self.confidence_threshold)
            )

            filtered_boxes = tf.gather_nd(box_prediction, indices)
            filtered_scores = tf.gather(scores, indices)[:, 0]

            # Perform NMS
            nms_indices = tf.image.non_max_suppression(
                filtered_boxes,
                filtered_scores,
                max_output_size=self.max_detections_per_class,
                iou_threshold=self.iou_threshold,
            )

            # Filter indices based on NMS
            indices = tf.gather(indices, nms_indices)
            return indices

        # Always convert to bounding_box_format for RDK
        box_prediction = bounding_box.convert_format(
            box_prediction,
            image_shape=image_shape,
            source=self.src_bounding_box_format,
            target=self.bounding_box_format,
        )

        # Apply logits to all class predictions, if specified
        if self.from_logits:
            class_predictions = tf.math.sigmoid(class_predictions)

        scores = tf.keras.backend.max(class_predictions, axis=1)
        labels = tf.keras.backend.argmax(class_predictions, axis=1)
        indices = nms_filtering_indices(scores)

        # Filter labels and then class predictions on indices and labels
        labels = tf.gather_nd(labels, indices)
        scores = tf.gather_nd(
            class_predictions,
            indices=tf.keras.backend.stack([indices[:, 0], labels], axis=1),
        )

        scores, top_indices = tf.nn.top_k(
            scores,
            k=tf.keras.backend.minimum(
                self.max_total_detections, tf.keras.backend.shape(scores)[0]
            ),
        )

        # Filter input using the final set of indices
        indices = tf.gather(indices, top_indices)
        boxes = tf.gather(box_prediction, indices)
        labels = tf.gather(labels, top_indices)

        # Outputs should either be named or be in order of location, category, score to comply with RDK
        bounding_boxes = {
            "boxes": boxes,
            # Since the prediction decoder is expected batched input,
            # we expand the dims such that the batch size is 1.
            "classes": tf.expand_dims(tf.cast(labels, dtype=tf.float32), axis=1),
            "confidence": tf.expand_dims(scores, axis=1),
            "num_detections": tf.cast(
                [tf.keras.backend.shape(boxes)[1]], dtype=tf.float32
            ),
        }

        return bounding_boxes
