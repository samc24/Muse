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
    "Real-time object detection and instance segmentation on iOS: PyTorch models converted to LibTorch and run on-device through a Swift + AVFoundation capture pipeline.",
  accent: "violet",
  highlights: [
    "Lead engineer on a deep-learning + iOS app for real-time object detection and instance segmentation.",
    "Improved accuracy with dropout, batch normalisation, and training on custom synthetic data.",
    "Converted models to run on-device via LibTorch; wrote the Objective-C++ bridge, hand-rolled NMS, and the full AVFoundation capture pipeline. CoreMotion integration plotted phone-path trajectory.",
    "Productionised the library and ETL pipeline with custom PyTorch DataLoaders and dataset mappers.",
  ],
  sections: [],
  stack: [
    {
      label: "Deep learning",
      items: [
        "Dropout, batch normalisation, hyperparameter tuning",
        "Training on custom synthetic data",
        "Custom PyTorch DataLoaders and dataset mappers",
        "TorchScript model conversion for mobile",
      ],
    },
    {
      label: "On-device mobile ML",
      items: [
        "iOS object detection + instance segmentation",
        "Swift + AVFoundation capture pipeline",
        "LibTorch via Objective-C++ bridge",
        "Device motion logging for trajectory analysis",
      ],
    },
  ],
  links: [],
  placeholder: true,
};
