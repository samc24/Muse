import type { CaseStudy } from "../types";

export const amazonPrimeVideo: CaseStudy = {
  slug: "amazon-prime-video",
  kind: "work",
  name: "Amazon",
  org: "Amazon Prime Video",
  role: "Machine Learning Engineer II",
  team: "Prime Video Personalization and Discovery, New York",
  period: "Jul 2025 -- current",
  tagline: "Recommendation ML + agentic AI",
  summary:
    "Prime Video's personalization service assembles the recommendation rows on the home page and storefronts worldwide. I work across the ML models and data pipelines that rank and mix those rows, and I'm a core developer of the agentic AI tooling the team uses to operate them.",
  accent: "sky",
  current: true,
  highlights: [
    "Core developer of Son of Anton, the team's agentic AI oncall platform: 7 Rust tools and 17 agent skills that give LLM agents live access to production systems, run in Claude Code and Kiro. It became the team's standard way to investigate production issues.",
    "Shipped one of the first AWS Bedrock AgentCore agents with per-user OAuth on-behalf-of delegation: platform GA to a production runtime in 8 days, now live in Slack answering as each requester.",
    "Worked across two generations of a worldwide subscription-channel recommender: a contextual multi-armed bandit learning in 6 regions, and its replacement, a sequence-aware collaborative-filtering model served through approximate-nearest-neighbor retrieval, whose 8-market A/B launch I owned.",
    "Root-caused months of silent recommendation degradation in one region to a stale upstream input, shipped six iterations of data-freshness safeguards, recovered the fleet across six regions, and built an automated health monitor (PySpark, Airflow on EMR on EKS, 80 tests).",
    "Argued for a final A/B test before a planned carousel retirement. It showed significant subscription lift (+134K annualized), reversed the decision, and led to worldwide expansion.",
    "Delivered a cross-org Fire TV home-screen filtering cap, caught four silent-failure modes before ship, and built and ran the partner's production load tests, cutting the required load about 3x.",
    "Owned annual infrastructure cost planning for two rounds (about 7.5% year-over-year reduction against a 5% target) and tier-1 oncall: 11 primary weeks and about 20 Sev2 incidents.",
    "Won the 2026 Prime Video hackathon with an automated NBA highlight row: live stats and play-by-play, PyTorch neural style transfer, OpenCV, ComfyUI, and stat-driven title cards.",
  ],
  sections: [
    {
      overline: "Context",
      heading: "The rows on the home page, and the models behind them.",
      paragraphs: [
        "The team owns the service that assembles personalized content rows for Prime Video's home page and storefront pages, the ML models and data pipelines that rank and mix those rows, and the AI tooling the team uses to operate them. It is a tier-1 customer-facing service deployed across six regions, with A/B testing at production scale and a primary oncall rotation I have been on since my third month.",
        "The recommender I have spent the most time on is a worldwide row that recommends titles from subscription channels a customer might sign up for. Generation one is a contextual multi-armed bandit: daily Spark learning jobs in six regions learn per-segment policies over channels, which a serving layer turns into slot quotas blended with a click-through-rate ranker. Generation two, the migration I later owned, is a sequence-aware collaborative-filtering model over the customer's viewing sequence, served through an ANN retrieval service with sponsored placements injected downstream.",
        "I joined during an organizational transition: my original manager and both onboarding buddies left the team within the first months. I delivered two features end to end in that window, fixed the onboarding guide so local builds and remote debugging actually worked for the next three engineers, and completed four primary oncall weeks before the year was out.",
      ],
    },
    {
      overline: "Agentic AI",
      heading: "Son of Anton: LLM agents with live access to production.",
      paragraphs: [
        "Son of Anton is the team's agentic oncall platform. I built it in the office with a teammate from the first tool onward, and it grew into the team's default way to investigate production issues. Seven Rust CLIs expose live systems as agent tools: log analytics with region and time macros, metrics graphs with deployment overlays, deployment history, content-configuration inspection and diffs, service request replay, catalog lookup, and search-index queries. They are packaged as 17 agent skills plus one agent definition that runs in both Claude Code and Kiro, with a one-command installer that wires the CLIs, skills, MCP servers, and a least-privilege permission allowlist.",
        "I wrote the end-to-end oncall investigation skill: a seven-step checklist (metrics and deployment correlation, log queries, request replay with debug info, ground-truth offer lookup, configuration diff, one worklog per stage) with parallel subagent fan-out, a local-only mode, trigger evals, and reference playbooks. Two overlapping agents became one agent plus skills, on the principle that the agent is the runtime envelope and the skill is the knowledge. A shared lessons library and a team context file load into every session.",
        "Then I took it multi-user. Before AWS Bedrock AgentCore shipped, I proved from source that the alternatives were single-owner or shared-identity, and picked per-user on-behalf-of delegation as the only correct model: every action runs as the person asking. From platform GA to a running production agent took eight calendar days, solo. That covered a formal application security review (a threat model with 24 mitigated threats, manual Rust review, static scan, privacy review, MCP-server scan, zero open critical or high findings), a standalone CDK package with its own pipeline, an execution role with mutation-tested security unit tests, a customer-managed KMS key on the token vault, and Bedrock Guardrails activated only after checking the false-positive rate against real tool output.",
        "The first end-to-end validation found three production IAM defects the platform's paved path never surfaced. I fixed all three live and backported them within 24 hours. The remaining failure, root-caused with the platform team, was a missing platform capability, and our use case became a driver of that roadmap item. Once it landed I rebuilt the Slack ingress as its own Lambda package with a CDK construct, got it through security approval, and shipped it in a day. The success measure: a teammate asked the agent a configuration question in the team channel and got a correct answer under their own identity after one consent click.",
      ],
      bullets: [
        "7 Rust CLIs, 17 agent skills, 2 agent harnesses (Claude Code, Kiro), one hosted AgentCore runtime, one Slack surface.",
        "Demoed end to end to a partner team, which filed its first ticket and change through it; presented at an org AI-native engineering brown bag.",
      ],
    },
    {
      overline: "Recommender ML",
      heading: "Owning a bandit in production, then launching its successor.",
      paragraphs: [
        "The first model I inherited was a logistic regression producing near-uniform predictions, about 20% for every strategy. The fix came in three steps: hashed interaction features, then stratified sampling per benefit with a minimum-share floor after finding that one benefit made up over 90% of positives, then a lower softmax temperature at inference, since ranking quality mattered more than calibration. It launched through a staged A/B dial-up from 1% to 100%, gated on punt rate, retrieval failures, latency SLAs, CPU, and availability.",
        "For the bandit I owned operations, data quality, monitoring, and experiments. Months of silently degraded recommendations in one region traced to a stale upstream input consumed with no freshness check. The systemic fix shipped as six iterating releases, from fail-fast on stale input through a formal succeed-on-empty-dataset contract, patched live across all six daily production jobs. When the upstream feed then died region by region, I ran a coordinated six-region policy reset, found separately descheduled jobs that would otherwise have silently frozen the reset, and verified the loop overnight: five regions fully recovered, the sixth proven dead upstream and deliberately silenced. Reading the publishing code also turned up a dated future outage nobody had seen, where one dead region would fail publishing for every healthy one. Three pipeline changes averted it before the cliff date.",
        "The plan of record was to retire the row. I argued for one last on/off experiment as due diligence. It showed significant subscription lift, the retirement was cancelled, and leadership expanded the row to every territory. For the dying upstream signal I mapped replacement options across three organizations, revived a dormant model at the right grain, and built a backtest harness over hundreds of millions of scored pairs, using histogram-based AUC to keep it tractable.",
        "When the owner of the generation-two migration left the stream mid-escalation, I took it over. I unblocked a stalled market by diagnosing a configuration-validator conflict, achieved ads parity across seven treatment configurations, found an eighth live configuration outside the experiment design, and rewrote the change-management plan into a stepped ramp with measured downstream capacity and numeric rollback triggers. Then I ran the dial myself: 10% to 50% to 100% in the first market with holds, then seven more markets to 100%. When one market showed no sponsored tiles, I found the allowlist cause and wrote the fix into the owning team's repository the same day.",
      ],
      bullets: [
        "Weekly fleet health monitor: 4 PySpark jobs and an Airflow DAG on EMR on EKS, 80 unit tests, an operational-versus-quality severity split, and an experiment-readiness go/no-go block. Validating it against live data found a real scheduled outage before it ever ran in production.",
        "One of my own changes caused a one-day production regression: a distinct() over a date column defeated partition pruning and ran out of memory. Reverted the same day, corrected, and written down as a rule.",
      ],
    },
    {
      overline: "Cross-org delivery",
      heading: "Fire TV filtering: the obvious fix was a no-op.",
      paragraphs: [
        "A partner needed our service to cap each row's fetch from 1,000 titles to 200 to cut their uncacheable filtering cost. The obvious one-line change did nothing: the size parameter was a floor, not a ceiling, and truncation happened after the expensive call. Before ship I caught four silent-failure modes (a device group mistaken for a device type, two row types bypassing the shared factory, typo'd ids failing silently) and added the counter metric that made the feature verifiable in production. Rollout was gated on explicit treatment arms rather than anything-but-control, and I mutation-tested both wrong versions.",
        "When the partner could not run their own production load tests, I built and ran them. Filtering the replay pool by page and row cut the required load about 3x, which brought two regions within pre-production capacity. A partner-reported leak traced to two mechanisms on their side, and a class of token-less downstream calls traced to a five-year-old bug in our own code; after the fix that traffic class went to zero.",
      ],
    },
    {
      overline: "Operations",
      heading: "Tier-1 oncall, and planning the fleet.",
      paragraphs: [
        "Eleven weeks as primary oncall plus cover days, around twenty Sev2 incidents, and 151 tickets resolved. The ones worth telling: a regional search-dependency outage that failed a large share of rows for over an hour, which I traced to another team's game-day load hitting a shared dependency directly, correcting five tickets that had been misattributed to a bad deployment. Recurring deployment-time availability dips, root-caused to cold-host warmup rather than the memory theory everyone held, with a factorial experiment packaged as a reusable playbook. A duplicate-titles page, resolved by replaying the customer's own request to show fifty distinct titles were served and tracing the identical cards to catalog entities with no parent.",
        "I owned the team's annual infrastructure cost plan for two rounds. The first was a bottom-up fleet model that landed about a 7.5% year-over-year reduction against a 5% target and held through two leadership reviews. The second re-planned the next year quarter by quarter from traffic forecasts divided by measured per-host throughput, found that the current fleet size reflected held event scaling rather than organic growth, and was verified to the dollar by re-export. I also cut one service's regional log ingestion to zero, fixed a load generator that logged every response body, and handed phase two to a junior engineer as their first project, reviewing rather than writing it.",
      ],
      bullets: [
        "Four two-week scrum-lead rotations; 149 code reviews in the first seven months of 2026, including a latent search-query defect caught in someone else's review and patched within 24 hours with zero customer exposure.",
        "A 34-slide system walkthrough for a newly merged team, every runtime fact re-verified against code and live systems.",
        "Tore down a declined experiment across four regions: service code, scheduled jobs, infrastructure stacks, 25 DynamoDB tables, and 8 versioned S3 buckets.",
      ],
    },
    {
      overline: "Hackathon",
      heading: "An NBA hype row, built in a weekend.",
      paragraphs: [
        "January 2026, Prime Video hackathon, first place. An automated highlight row for upcoming NBA matchups: live stats and play-by-play pulled in, highlight reels assembled with PyTorch neural style transfer, OpenCV, and ComfyUI workflows, and stat-driven title cards generated for each game. The one week at work where the basketball and the computer vision were the job.",
      ],
    },
  ],
  stack: [
    {
      label: "Languages",
      items: [
        "Java, Kotlin (service and inference)",
        "Python, PySpark (ML pipelines)",
        "Rust (agent CLI tools)",
        "TypeScript (AWS CDK)",
      ],
    },
    {
      label: "Recommender ML",
      items: [
        "Contextual multi-armed bandits",
        "Sequence-aware collaborative filtering",
        "Approximate-nearest-neighbor (ANN) retrieval",
        "A/B testing at production scale, staged dial-ups with guardrails",
      ],
    },
    {
      label: "AWS",
      items: [
        "Bedrock AgentCore, Bedrock Guardrails",
        "Lambda, API Gateway, DynamoDB, S3, KMS",
        "EMR on EKS, SageMaker, Airflow, Spark, OpenSearch",
        "CloudWatch metrics, alarms, and logs; CloudFormation and CDK",
      ],
    },
    {
      label: "Agents and identity",
      items: [
        "Claude Code, Kiro, MCP servers, agent skills",
        "OAuth 2.0: token exchange, PKCE, on-behalf-of delegation, JWT authorizers",
        "Threat modelling, mutation-tested IAM policies",
      ],
    },
  ],
  links: [],
};
