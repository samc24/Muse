import type { CaseStudy } from "../types";

export const microsoft: CaseStudy = {
  slug: "microsoft",
  kind: "work",
  name: "Microsoft",
  org: "Microsoft",
  role: "Software Engineer II",
  team: "M365 Security Detections",
  period: "2021 -- 2025",
  tagline: "Security ML at petabyte scale",
  summary:
    "Four years on the petabyte-scale platform that detects cyber-attacks across all of M365 in near real time. Lead engineer for Microsoft's first mail anomaly detection and for the GPT-based detection assistant.",
  accent: "blue",
  highlights: [
    "Petabyte-scale big-data platform using ML to detect cyber-attacks in near real time across all of M365.",
    "Lead engineer for Microsoft's first mail-anomaly detection (response to the nation-state incident where hackers accessed executive and federal emails).",
    "Lead engineer on a GPT-based security-detection assistant that analyses machine and identity activity for attack patterns.",
    "Designed the production workspace that enables data-science and LLM-backed detections in an agile PySpark environment with CI/CD and realtime job monitoring.",
    "Expedited detection programming by over 75% on distributed systems; auto-failover geo-replicated BCDR architecture.",
  ],
  sections: [],
  stack: [
    {
      label: "Security + anomaly ML",
      items: [
        "Mail-anomaly detection for M365",
        "GPT-based security-detection assistant",
        "Random-forest entity detections",
        "Petabyte-scale event processing",
      ],
    },
    {
      label: "Distributed systems",
      items: [
        "Spark, PySpark, Azure Functions",
        "CI/CD, real-time job monitoring",
        "Auto-failover geo-replicated BCDR",
      ],
    },
  ],
  links: [],
  placeholder: true,
};
