# Source references

Checked on 14 November 2026. These are links and project summaries of official documentation, not archived copies. Recheck APIs and licensing before deployment.

- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/): tracking returns object IDs alongside detection boxes; ByteTrack can be explicitly selected with `tracker="bytetrack.yaml"`. This project uses persistent tracking across sequential frames.
- [Ultralytics training](https://docs.ultralytics.com/modes/train/): custom detection training can start from pretrained weights and a dataset configuration. This project requires custom training for the target logo.
- [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect/): dataset YAML identifies split directories and class names. Detection labels use class IDs and normalized box coordinates.
- [YOLO11 models](https://docs.ultralytics.com/models/yolo11/): `yolo11n.pt` is a supported detection checkpoint with training, prediction, and validation support. It is the initial small baseline here.
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html): reads frames from files, cameras, and IP video streams; capture resources must be released.
- [CVAT dataset formats](https://docs.cvat.ai/docs/dataset_management/formats/): annotation formats include Ultralytics YOLO detection exports.
- [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk): candidate toolkit for larger streaming video analytics deployments on NVIDIA hardware; not needed for the initial local pipeline.
- [OpenCV template matching](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html): the evaluation prototype uses normalized correlation against rotated and scaled reference-tag crops as an additional shape check. Correlation scores are not probabilities.

The architecture, alert rules, crop strategy, and acceptance criteria in `approach.md` are project design decisions built on these tools, not guarantees made by their documentation.
