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
    "Upstream contributions to CRI-O and Libpod, the container runtime and pod manager underneath OpenShift and Kubernetes. All of it is public: three merged pull requests in Go, each with its own tests.",
  accent: "red",
  highlights: [
    "Made image volumes writeable by the container user in CRI-O, closing a bug that blocked engineers from writing to image volumes from inside the container. Merged upstream with a test.",
    "Added an environment-variable alias to every CRI-O CLI flag, so the daemon can be configured from the environment instead of a long flag list at start-up. 11 files, merged upstream.",
    "Rewrote hooks-directory monitoring in Libpod to re-read the directories on any watcher event, fixing four edge cases that could drop valid hooks or run a hook twice. Merged upstream with a monitoring test.",
    "Updated the crio.conf man page and the build-from-source tutorial along the way, and proposed an integration test for annotation passthrough to the runtime.",
  ],
  sections: [
    {
      overline: "Context",
      heading: "The layer under Kubernetes.",
      paragraphs: [
        "In 2019 I was on the OpenShift container runtimes team at Red Hat, working upstream in Go on CRI-O, the container runtime interface that Kubernetes uses to run pods, and Libpod, the library behind Podman that manages containers and pods. Everything I did there is on GitHub under my own name, which is still the part of my resume I can point to most directly.",
      ],
    },
    {
      overline: "Image volumes",
      heading: "Writeable from inside the container.",
      paragraphs: [
        "Image volumes in CRI-O were created with permissions that the container's own user could not write to, which engineers kept running into. The fix made image volumes always writeable from inside the container: when a run-as-user is specified for the container, the volume gets the matching ownership. It shipped with a test that exercises the case.",
      ],
    },
    {
      overline: "Configuration",
      heading: "Every flag, also an environment variable.",
      paragraphs: [
        "Starting CRI-O meant passing every setting as a command-line flag. I added an environment-variable alias to each flag, named after the flag in upper case with underscores for hyphens, so the same values can be set in the environment and the daemon picks them up without a flag list. Small change in concept, 11 files in practice, because every flag definition had to carry it.",
      ],
    },
    {
      overline: "Hooks",
      heading: "Monitoring hook directories without missing events.",
      paragraphs: [
        "OCI hooks are small programs a runtime executes at points in a container's life, and Libpod watches the hook directories so new hooks take effect without a restart. The existing monitor mishandled several real edge cases: reading a directory stopped at the first invalid hook instead of registering the error and continuing to the valid ones, rename and chmod events were ignored, a hook file that was moved rather than deleted was still treated as present, and a renamed hook was registered as a second hook alongside the original, so it executed twice.",
        "Rather than patch each case, I simplified the monitor to re-read the hook directories whenever the watchers detect any change, and added a test that drops a different invalid hook into the primary directory. The change landed in Libpod with fewer lines than it removed. I then opened the follow-up to vendor it into CRI-O.",
      ],
    },
  ],
  stack: [
    {
      label: "Container runtimes",
      items: ["CRI-O", "Libpod / Podman", "Kubernetes", "OCI hooks"],
    },
    {
      label: "Tooling",
      items: ["Go", "fsnotify directory watchers", "Integration tests"],
    },
  ],
  links: [
    {
      label: "CRI-O: writeable image volumes",
      href: "https://github.com/cri-o/cri-o/pull/2652",
    },
    {
      label: "CRI-O: env vars for CLI flags",
      href: "https://github.com/cri-o/cri-o/pull/2423",
    },
    {
      label: "Libpod: hooks monitoring",
      href: "https://github.com/containers/podman/pull/3580",
    },
  ],
};
