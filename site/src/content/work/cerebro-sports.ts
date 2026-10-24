import type { CaseStudy } from "../types";

export const cerebroSports: CaseStudy = {
  slug: "cerebro-sports",
  kind: "work",
  name: "Cerebro Sports",
  org: "Cerebro Sports",
  role: "Machine Learning Engineer, contract",
  team: "Live Gameplay Team",
  period: "Summer 2025",
  tagline: "Short contract, live stats demo",
  summary:
    "A brief contract around the NBA Summer League in Las Vegas. I built a demo app for exploring live basketball tournament stats conversationally: ask a question in plain English, get the numbers, realtime graphs, and dashboards back.",
  accent: "green",
  highlights: [
    "Short engagement in summer 2025, scoped to one demo for a tournament.",
    "LLM chat over live tournament stats, with realtime graphs and dashboards.",
    "Designed the pipeline that converts semantic text to SQL, queries the stats DB, processes data, and handles edge cases.",
  ],
  sections: [],
  stack: [
    {
      label: "Live sports + LLM apps",
      items: [
        "Semantic-text to SQL pipelines",
        "Basketball stats DB + edge-case handling",
        "LLM chat + realtime graphs + dashboards",
      ],
    },
  ],
  links: [],
};
