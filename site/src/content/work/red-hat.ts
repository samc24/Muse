import type { CaseStudy } from "../types";

export const redHat: CaseStudy = {
  slug: "red-hat",
  kind: "work",
  name: "Red Hat",
  org: "Red Hat OpenShift",
  role: "Software Engineer",
  team: "OpenShift -- Container Runtimes",
  period: "2019",
  tagline: "Kubernetes container runtimes",
  summary:
    "Upstream contributions to CRI-O and Libpod, the container runtime and pod manager underneath OpenShift and Kubernetes.",
  accent: "red",
  highlights: [
    "Upstream contributions to CRI-O (Kubernetes container runtime interface) and Libpod (container and pod manager).",
    "Unblocked engineers on image-volume write issues inside containers, fixed container-hook monitoring and execution, and added CLI env-var flag passthrough.",
  ],
  sections: [],
  stack: [
    {
      label: "Container runtimes",
      items: ["CRI-O", "Libpod", "Kubernetes"],
    },
  ],
  links: [],
  placeholder: true,
};
