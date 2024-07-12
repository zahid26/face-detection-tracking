from functools import partial
import cv2
import tensorflow as tf
import numpy as np
import cv2
import time
from loguru import logger

from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=100, n_init=10)


class UltraLightFaceDetecion():
    def __init__(self, filepath, input_size=(320, 240), conf_threshold=0.6,
                 center_variance=0.1, size_variance=0.2,
                 nms_max_output_size=200, nms_iou_threshold=0.3) -> None:

        self._feature_maps = np.array([[40, 30], [20, 15], [10, 8], [5, 4]])
        self._min_boxes = [[10, 16, 24], [32, 48],
                                    [64, 96], [128, 192, 256]]

        self._resize = partial(cv2.resize, dsize=input_size)
        self._input_size = np.array(input_size)[:, None]

        self._anchors_xy, self._anchors_wh = self._generate_anchors()
        self._conf_threshold = conf_threshold
        self._center_variance = center_variance
        self._size_variance = size_variance
        self._nms = partial(tf.image.non_max_suppression,
                            max_output_size=nms_max_output_size,
                            iou_threshold=nms_iou_threshold)

        # tflite model init
        self._interpreter = tf.lite.Interpreter(model_path=filepath)
        self._interpreter.allocate_tensors()

        # model details
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        # inference helper
        self._set_input_tensor = partial(self._interpreter.set_tensor,
                                         input_details[0]["index"])
        self._get_boxes_tensor = partial(self._interpreter.get_tensor,
                                         output_details[0]["index"])
        self._get_scores_tensor = partial(self._interpreter.get_tensor,
                                          output_details[1]["index"])

    def _generate_anchors(self):
        anchors = []
        for feature_map_w_h, min_box in zip(self._feature_maps, self._min_boxes):

            wh_grid = min_box / self._input_size
            wh_grid = np.tile(wh_grid.T, (np.prod(feature_map_w_h), 1))

            xy_grid = np.meshgrid(range(feature_map_w_h[0]),
                                  range(feature_map_w_h[1]))
            xy_grid = np.add(xy_grid, 0.5)

            xy_grid /= feature_map_w_h[..., None, None]

            xy_grid = np.stack(xy_grid, axis=-1)
            xy_grid = np.tile(xy_grid, [1, 1, len(min_box)])
            xy_grid = xy_grid.reshape(-1, 2)

            prior = np.concatenate((xy_grid, wh_grid), axis=-1)
            anchors.append(prior)

        anchors = np.concatenate(anchors, axis=0)
        anchors = np.clip(anchors, 0.0, 1.0)

        return anchors[:, :2], anchors[:, 2:]

    def _pre_processing(self, img):
        resized = self._resize(img)
        image_rgb = resized[..., ::-1]
        image_norm = image_rgb.astype(np.float32)
        cv2.normalize(image_norm, image_norm,
                      alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
        return image_norm[None, ...]

    def inference(self, img):
        # BGR image to tensor
        input_tensor = self._pre_processing(img)

        # set tensor and invoke
        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

        # get results
        boxes = self._get_boxes_tensor()[0]
        scores = self._get_scores_tensor()[0]

        # decode boxes to corner format
        boxes, scores = self._post_processing(boxes, scores)
        boxes *= np.tile(img.shape[1::-1], 2)

        return boxes, scores

    def _post_processing(self, boxes, scores):
        # bounding box regression
        boxes = self._decode_regression(boxes)
        scores = scores[:, 1]

        # confidence threshold filter
        conf_mask = self._conf_threshold < scores
        boxes, scores = boxes[conf_mask], scores[conf_mask]

        # non-maximum suppression
        nms_mask = self._nms(boxes=boxes, scores=scores)
        boxes = np.take(boxes, nms_mask, axis=0)

        return boxes, scores

    def _decode_regression(self, reg):
        # bounding box regression
        center_xy = reg[:, :2] * self._center_variance * \
            self._anchors_wh + self._anchors_xy
        center_wh = np.exp(
            reg[:, 2:] * self._size_variance) * self._anchors_wh / 2

        # center to corner
        start_xy = center_xy - center_wh
        end_xy = center_xy + center_wh

        boxes = np.concatenate((start_xy, end_xy), axis=-1)
        boxes = np.clip(boxes, 0.0, 1.0)

        return boxes
    

def image_inference(image_path, model_path, color=(125, 255, 0)):

    fd = UltraLightFaceDetecion(model_path,
                                conf_threshold=0.6)

    img = cv2.imread(image_path)

    boxes, scores = fd.inference(img)

    for result in boxes.astype(int):
        cv2.rectangle(img, (result[0], result[1]),
                      (result[2], result[3]), color, 2)

    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_inference(video, model_path, color=(125, 255, 0)):

    fd = UltraLightFaceDetecion(model_path, conf_threshold=0.88)
    cap = cv2.VideoCapture(video)
    # prev_bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        boxes, scores = fd.inference(frame)
        print("Inference time: ", time.perf_counter() - start_time)

        detections = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], float(score), 0))

        if detections:
            # tracked_objects = tracker.update_tracks(detections, frame=frame)
            # object_chips = tracker.chipper(frame, [x1,y1,w,h])
            # cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
            embeds = tracker.generate_embeds(frame, detections)
            tracked_objects = tracker.update_tracks(detections, embeds=embeds)

            logger.warning(np.array(embeds).shape)

            for track in tracked_objects:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()  # Get the bounding box coordinates in (x1, y1, x2, y2) format

                # if prev_bbox is not None:
                #     bbox = [int((prev + curr) / 2) for prev, curr in zip(prev_bbox, bbox)]
                # prev_bbox = bbox

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            logger.info("NO detections...")

        cv2.imshow('res', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filepath = "" # model path here
    camera_url = ""# camera url here or camera index
    video_inference(camera_url, filepath)

