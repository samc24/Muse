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
    "Lead engineer for Microsoft's first mail anomaly detection, built to catch malicious mailbox access in response to the global incident in which Russian state hackers accessed the emails of executives and federal agencies.",
    "Lead engineer on a GPT-based security-detection assistant that analyses machine and identity activity for attack patterns.",
    "Designed and built the production workspace that enables data-science and LLM-backed detections in an agile PySpark environment, integrated with AI libraries, CI/CD, and real-time job monitoring. Detection authoring got over 75% faster.",
    "Developed an event-processing system on Spark and Azure Functions with real-time alerting that scales with load, and the auto-failover geo-replicated architecture behind the platform's business continuity and disaster recovery.",
    "Primary on-call for pipeline maintenance; analysed host logs, atomic detections, and random-forest entity detections to investigate suspicious activity.",
  ],
  sections: [
    {
      overline: "Context",
      heading: "Protecting the M365 datacenters in real time.",
      paragraphs: [
        "The team monitors Microsoft 365's datacenters, detects threats, and acts to prevent compromise. The work sits at the intersection of data science, statistical inference, machine learning, distributed computing, and cyber security, and the thing we built was the investigation and remediation pipeline: millions of events per second in, real-time security detection and alerting out.",
        "The platform ingests compressed host and Azure telemetry from Event Hubs and processes it downstream with Spark streaming jobs. Detections feed alerting, hunting, and ML model training. The core runs on HDInsight Spark with jobs in Scala; the data ingestion pipeline is a .NET Core service on Service Fabric clusters; the data stores range across Cosmos DB, Parquet, Azure Data Explorer, and Azure SQL. I produced architecture, design, and code for many of the core components, working across Spark, Scala, C#, Python, SQL, Kusto, and Azure.",
      ],
    },
    {
      overline: "Detections",
      heading: "The first mail anomaly detection, and an LLM that reads the logs.",
      paragraphs: [
        "When a nation-state actor was found to have accessed the mailboxes of executives and federal agencies, the response needed a detection that did not yet exist anywhere at Microsoft: anomalous mailbox access. I was the lead engineer on it, and it runs on the streaming platform so it catches malicious access in near real time.",
        "I was also lead engineer on a GPT-based security-detection assistant. Given a machine's or an identity's activity, it analyses that activity for attacks.",
      ],
    },
    {
      overline: "Platform",
      heading: "A workspace where detections ship in days, not months.",
      paragraphs: [
        "I designed and built a production workspace that lets data scientists and engineers author data-science and LLM-backed detections in an agile PySpark environment, integrated with AI libraries, CI/CD, and real-time job monitoring. Detection programming on the distributed platform got over 75% faster, which is what made iterative progress toward catching attackers possible.",
        "Underneath it, I developed an event-processing system on Spark and Azure Functions with real-time alerting that scales horizontally with load.",
      ],
    },
    {
      overline: "Reliability",
      heading: "Replicating the whole platform to a second region.",
      paragraphs: [
        "For business continuity and disaster recovery I designed and built an auto-failover geo-replicated architecture, which meant replicating the entire platform to a secondary region: Spark batch and streaming jobs, SQL tables, host and Azure telemetry from Event Hubs, the detection triage UI, Service Fabric clusters and Traffic Managers, and the assorted data stores and features around them.",
        "I served as primary on-call for pipeline maintenance, tuned Spark jobs, and remediated platform health. On the analyst side, I triaged security alerts by analysing processes and applying judgment on atomic detections and entity metadetections, including random-forest-trained entity detections, to separate benign from malicious activity and investigate further.",
      ],
    },
  ],
  stack: [
    {
      label: "Security + anomaly ML",
      items: [
        "Mail anomaly detection for M365",
        "GPT-based security-detection assistant",
        "Random-forest entity detections",
        "Alert triage: atomic detections, entity metadetections",
      ],
    },
    {
      label: "Distributed systems",
      items: [
        "Spark streaming on HDInsight, Scala, PySpark",
        "Azure Functions, Event Hubs, Service Fabric, .NET Core, C#",
        "Cosmos DB, Parquet, Azure Data Explorer (Kusto), Azure SQL",
        "CI/CD, real-time job monitoring, auto-failover geo-replication",
      ],
    },
  ],
  links: [],
};
