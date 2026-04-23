type TimelineItem = {
  company: string;
  role: string;
  team?: string;
  period: string;
  points: string[];
};

const ITEMS: TimelineItem[] = [
  {
    company: "Amazon",
    role: "Machine Learning Engineer II",
    team: "Prime Video -- Browse and Discovery",
    period: "2025 -- current",
    points: [
      "Mixture-of-Retrieval ML models that proportionally promote content by genre and propensity.",
      "Custom logistic regression with stratified sampling + softmax tuning fixed flat predictions caused by 90%+ channel over-representation in training data; improved Share-of-Voice across genre carousels.",
      "Designed the end-to-end pipeline for FreshPicks, a carousel for new third-party content, driving +134K annualised subscriptions.",
      "Architected a cross-service content-filtering pipeline for subscription-bundle carousels with realtime entitlement parsing, token decoding, and multi-filter orchestration.",
    ],
  },
  {
    company: "MIT",
    role: "Lead Software Engineer",
    team: "Plasma Science and Fusion Center",
    period: "2025 -- current",
    points: [
      "Architecting DisruptionPy: a framework for analysing plasma-fusion data across five Tokamak reactors to predict and avoid disruptions using anomaly models.",
      "Revamping the backend to decouple data access from specific implementations; pluggable backends (MDSplus, Xarray/Zarr) through abstract interfaces for testability and multi-machine scalability.",
    ],
  },
  {
    company: "Cerebro Sports",
    role: "Lead Machine Learning Engineer",
    team: "Live Gameplay Team",
    period: "2025",
    points: [
      "Full-stack build: frontend, API, LLM layer, and data pipeline all shipped by a small team.",
      "App lets coaches, scouts, and analysts interrogate live basketball-tournament stats via LLM chat, realtime graphs, and dashboards.",
      "Designed the pipeline that converts semantic text to SQL, queries the stats DB, processes data, and handles edge cases.",
    ],
  },
  {
    company: "Microsoft",
    role: "Software Engineer II",
    team: "M365 Security Detections",
    period: "2021 -- 2025",
    points: [
      "Petabyte-scale big-data platform using ML to detect cyber-attacks in near real time across all of M365.",
      "Lead engineer for Microsoft&apos;s first mail-anomaly detection (response to the nation-state incident where hackers accessed executive and federal emails).",
      "Lead engineer on a GPT-based security-detection assistant that analyses machine and identity activity for attack patterns.",
      "Designed the production workspace that enables data-science and LLM-backed detections in an agile PySpark environment with CI/CD and realtime job monitoring.",
      "Expedited detection programming by over 75% on distributed systems; auto-failover geo-replicated BCDR architecture.",
    ],
  },
  {
    company: "AI.Reverie",
    role: "Machine Learning Engineer",
    team: "Computer Vision Team",
    period: "2020",
    points: [
      "Lead engineer on a deep-learning + iOS app for real-time object detection and instance segmentation.",
      "Improved accuracy with dropout, batch normalisation, and training on custom synthetic data.",
      "Converted models to run on-device via LibTorch; wrote the Objective-C++ bridge, hand-rolled NMS, and the full AVFoundation capture pipeline. CoreMotion integration plotted phone-path trajectory.",
      "Productionised the library and ETL pipeline with custom PyTorch DataLoaders and dataset mappers.",
    ],
  },
  {
    company: "Red Hat",
    role: "Software Engineer",
    team: "OpenShift -- Container Runtimes",
    period: "2019",
    points: [
      "Upstream contributions to CRI-O (Kubernetes container runtime interface) and Libpod (container and pod manager).",
      "Unblocked engineers on image-volume write issues inside containers, fixed container-hook monitoring and execution, and added CLI env-var flag passthrough.",
    ],
  },
];

export function Timeline() {
  return (
    <ol className="relative">
      {ITEMS.map((item, i) => (
        <li
          key={`${item.company}-${i}`}
          className="relative pl-8 pb-10 last:pb-0 border-l hairline-strong ml-3"
        >
          <span className="absolute -left-[7px] top-1 w-3 h-3 rounded-full bg-accent ring-4 ring-background" />
          <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
            <div>
              <span className="text-lg font-semibold tracking-tight text-foreground">
                {item.role}
              </span>
              <span className="text-muted"> at </span>
              <span className="text-lg font-semibold tracking-tight text-foreground">
                {item.company}
              </span>
            </div>
            <span className="text-xs font-mono uppercase tracking-[0.15em] text-muted whitespace-nowrap">
              {item.period}
            </span>
          </div>
          {item.team && (
            <p className="mt-1 text-sm text-muted">{item.team}</p>
          )}
          <ul className="mt-4 space-y-2">
            {item.points.map((p, j) => (
              <li
                key={j}
                className="text-muted-strong leading-relaxed text-[0.975rem] pl-4 relative before:content-['-'] before:absolute before:left-0 before:text-muted"
                dangerouslySetInnerHTML={{ __html: p }}
              />
            ))}
          </ul>
        </li>
      ))}
    </ol>
  );
}
