import type { CaseStudy } from "../types";

export const aiReverie: CaseStudy = {
  slug: "ai-reverie",
  kind: "work",
  name: "AI.Reverie",
  org: "AI.Reverie",
  role: "Machine Learning Engineer",
  team: "Computer Vision Team",
  period: "2020",
  tagline: "On-device computer vision",
  summary:
    "Real-time object detection on iPhone with PyTorch: a YOLO model trained on a customer's dataset, traced to TorchScript, and run on device through an Objective-C++ bridge and an AVFoundation capture pipeline, with device motion logged to plot the phone's path in 3D.",
  accent: "violet",
  highlights: [
    "Lead engineer on a deep-learning + iOS app for real-time object detection and instance segmentation.",
    "Trained YOLOv5 on a customer dataset, converted the annotations to YOLO format, added an output layer that returns usable detections directly, and traced the model to TorchScript for PyTorch Mobile.",
    "Wrote the iOS side: Objective-C++ bridge between Swift and the LibTorch C++ API, AVFoundation capture, hand-rolled non-maximum suppression, bounding boxes over the live feed, and on-device logging of every detection.",
    "Logged accelerometer and device motion with CoreMotion and plotted the phone's 3D trajectory, the first step toward estimating a drone's position without GPS.",
    "Improved model accuracy with dropout, batch normalisation, and training on custom synthetic data; productionised the library and ETL pipeline with custom PyTorch DataLoaders and dataset mappers.",
  ],
  sections: [
    {
      overline: "Context",
      heading: "Fast, accurate detection on a phone, with no cloud in the loop.",
      paragraphs: [
        "AI.Reverie generated synthetic training data for computer vision, and the computer vision team also built the models and tooling around it. My project was a real-time object detection app for iOS, built on PyTorch rather than CoreML so the trained model stayed bit-exact on device. The longer-term idea was a phone mounted on a drone: detect objects from the sky, and use the device's own orientation and acceleration data alongside the detections to estimate where the drone is without GPS or WiFi, plot its path, and feed that back into the detections themselves.",
        "Alongside the app, I worked on the team's detection and segmentation library: making the early-stage pipeline production-ready, fixing RCNN and DETR training with custom PyTorch DataLoaders and model configuration changes, and standardising data processing with custom dataset mappers and pre- and post-processing scripts.",
      ],
    },
    {
      overline: "Model",
      heading: "YOLOv5 on a customer dataset, traced for mobile.",
      paragraphs: [
        "The customer's dataset came as images with XML annotations. I iterated over every file to find the class set, converted each annotation to the YOLO format (one text file per image, one line per object with class and normalised bounding box), wrote the dataset configuration, and fine-tuned yolov5s from the pretrained checkpoint rather than from scratch so the weights started in a sensible place and mAP came up faster.",
        "The trained model's raw output is not what an app wants, so I loaded it back and appended a layer that returns x, y, width, height, confidence, and class for every detection above a threshold. Then I traced it with a dummy input tensor through torch.jit.trace into a TorchScript file that PyTorch Mobile can load on iOS.",
        "Getting there took a few dead ends worth recording. The first plan was CoreML via ONNX. Detectron2 would not export because its inputs are lists of dicts rather than tensors, and the export that did succeed failed to decode in the CoreML converter. YOLOv5 exported to ONNX but the CoreML converter choked on a Slice op from the upsampling layers. On the TorchScript path, DETR and Faster R-CNN traced but would not load on iOS because torchvision's NMS op is not available in mobile TorchScript, which is also why NMS ended up implemented by hand in the app rather than inside the model.",
      ],
    },
    {
      overline: "On device",
      heading: "Swift to Objective-C++ to LibTorch, and back with boxes.",
      paragraphs: [
        "TorchScript modules are driven from C++, so the app uses Objective-C++ as the bridge between Swift and the LibTorch API. The app loads the traced model and the class list, starts an AVFoundation capture session, and calls predict on every frame. The frame goes from the sample buffer through a pixel buffer and a rotation-corrected CIImage to a 224 by 224 normalised tensor, the module runs the forward pass, and the results come back to Swift. I implemented non-maximum suppression with IoU in Swift, drew the surviving boxes over the live video, and saved the detection stream to a timestamped file when the app quits.",
        "CoreMotion's device-motion API gave the acceleration caused by the user in three axes. I logged it to the device, and on a button press integrated it into a 3D path rendered with an open-source plotting framework. In the demo, a circular spiral motion of the phone comes back as a recognisable spiral.",
      ],
    },
    {
      overline: "Results",
      heading: "Speed versus accuracy on an iPhone 8.",
      paragraphs: [
        "Measured on an iPhone 8 with 2 GB of memory, iOS 13.5. mAP is proportional to image resolution, so the smaller input is faster but misses small objects. I chose 224 for frame rate in the demo and documented the trade-off.",
      ],
      bullets: [
        "YOLOv5 L at 480: 0.77 FPS, 0.241 mAP at 0.5 IoU.",
        "YOLOv5 L at 224: 1.75 FPS, 0.150 mAP at 0.5 IoU.",
        "YOLOv5 S at 480: 2.63 FPS, 0.233 mAP at 0.5 IoU.",
        "YOLOv5 S at 224: 8.33 FPS, 0.149 mAP at 0.5 IoU.",
      ],
    },
    {
      overline: "Experiments",
      heading: "Dropout did not help, and what I would do next.",
      paragraphs: [
        "I tried adding dropout to yolov5s while still loading the pretrained checkpoint, which meant patching the checkpoint loader to skip layers that did not exist in the saved weights. Four dropout layers after the convolutions were slightly worse than the baseline; moving them after the activations inside the bottleneck blocks was still not better. I wrote it up as a negative result rather than ship it.",
        "The next steps I documented: soft-NMS in place of greedy suppression, quantization for speed, and handling the phone camera's aspect ratio properly, since YOLO takes square inputs and cropping to 1:1 would likely beat squashing the frame.",
      ],
    },
  ],
  stack: [
    {
      label: "Deep learning",
      items: [
        "PyTorch, YOLOv5, Detectron2, DETR",
        "Dropout, batch normalisation, hyperparameter tuning",
        "Training on custom synthetic data",
        "Custom PyTorch DataLoaders and dataset mappers",
      ],
    },
    {
      label: "On-device mobile ML",
      items: [
        "TorchScript tracing, PyTorch Mobile, LibTorch",
        "Swift, AVFoundation capture pipeline",
        "Objective-C++ bridge to the LibTorch C++ API",
        "CoreMotion device-motion logging, 3D trajectory plotting",
      ],
    },
  ],
  links: [],
};
