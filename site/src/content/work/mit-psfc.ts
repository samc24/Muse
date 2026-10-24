import type { CaseStudy } from "../types";

const GH = "https://github.com/MIT-PSFC/disruption-py";

export const mitPsfc: CaseStudy = {
  slug: "mit-psfc",
  kind: "work",
  name: "MIT PSFC",
  org: "MIT Plasma Science and Fusion Center",
  role: "Lead Software Engineer",
  team: "Disruption Studies Group -- DisruptionPy",
  period: "2025 -- current",
  tagline: "Scientific software for fusion",
  summary:
    "DisruptionPy is the group's open-source framework for turning raw tokamak experiment data into analysis-ready and ML-ready datasets across five machines. I own its software architecture: the data-access layer, configuration, multiprocessing, and developer tooling.",
  accent: "red",
  current: true,
  highlights: [
    "Architecting DisruptionPy: a framework for analysing plasma-fusion data across five tokamak reactors to predict and avoid disruptions using anomaly models.",
    "Revamped the backend to decouple data access from specific implementations: pluggable backends (MDSplus, Xarray/Zarr) behind abstract interfaces for testability and multi-machine scalability. Validated bit-identical output against the reference dataset before merge.",
    "Root-caused and fixed a family of multiprocessing start-method bugs (fork vs spawn vs forkserver) that silently dropped every MAST physics variable on macOS and Python 3.14.",
    "Moved framework defaults into layered configuration (TOML, environment, explicit arguments) so each run records the settings it actually used.",
    "First author on an APS Division of Plasma Physics 2026 poster on maintainable, extensible architecture for multi-machine disruption studies.",
  ],
  sections: [
    {
      overline: "Context",
      heading: "What the framework does, and why the data is hard.",
      paragraphs: [
        "A tokamak confines a plasma at tens of millions of degrees in a donut-shaped magnetic field. Each experiment is a shot lasting a few seconds, recorded by thousands of diagnostics at kHz to MHz and stored per shot in hierarchical MDSplus trees. A disruption is a sudden loss of the plasma's stored energy over milliseconds. It can damage the machine, and next-generation devices have to operate close to the limits that trigger it, so predicting and avoiding disruptions is a leading engineering problem. There is no first-principles predictor, but models trained on past shots have worked, which is why the group builds cross-machine datasets.",
        "DisruptionPy runs a standard pipeline of physics routines over shots, interpolates every signal onto one requested timebase, labels time-until-disruption from a metadata database, and writes a dataset indexed by shot and time with roughly sixty variables per machine. It supports Alcator C-Mod, DIII-D, EAST, HBT-EP, and MAST, is MIT-licensed, and ships on PyPI with a documentation site. It is an offline data pipeline, not a realtime system.",
      ],
    },
    {
      overline: "Backend abstraction",
      heading: "Adding a tokamak should be physics work, not data-access plumbing.",
      paragraphs: [
        "The framework grew up MDSplus-centric. MAST's data lives in Zarr stores on an S3-compatible object store with no MDSplus at all, and the first MAST integration bolted an Xarray connection on with duck typing and a type cast repeated in every MAST physics method. The fix was to separate two concerns the code had been mixing: per-shot data access and per-process connection lifecycle.",
        "I introduced DataConnection and ProcessConnection abstract base classes, implemented them for MDSplus and for Xarray/Zarr, and rewired the retrieval manager, settings, and the machine modules on top of them. Path formats deliberately stay backend-specific, because normalising them would have meant rewriting every physics method for no user-visible gain. The design went through five design documents and one rejected abstraction that turned out to be indirection for its own sake.",
        "Before merging I verified the output was bit-identical to the reference dataset on a set of C-Mod shots, serial versus parallel, and that known-corrupt shots produced the same errors as before. Later work on the error hierarchy and on data-access forwarders built on the new interfaces.",
      ],
      bullets: [
        "PR #523: introduce the DataConnection and ProcessConnection ABCs.",
        "PR #534: migrate every consumer from the MDSplus-specific connection to the abstract one; add the data-connection reference page to the docs.",
      ],
    },
    {
      overline: "Multiprocessing",
      heading: "One root cause, three bugs: state that does not survive spawn.",
      paragraphs: [
        "The workflow fans shots out to a multiprocessing pool. On Linux the pool forks, so workers inherit everything the main process set up. On macOS the default is spawn, and Python 3.14 moved Linux to forkserver, and in both cases workers start from a fresh interpreter. Anything registered only in the main process is simply gone. The first symptom was that every MAST physics method failed inside workers with a missing custom log level, so zero of thirty-three parameters had data. A pytest hang and an xarray merge error on concatenation turned out to be the same problem wearing different clothes.",
        "The fixes moved all logger state to import time, gave the pool an initializer that resets handlers, made the affected test spawn-safe so the macOS CI matrix could land, and stamped workflow metadata in the main process instead of in each worker.",
      ],
      bullets: [
        "PR #546: logging that works under both fork and spawn.",
        "PR #560: make the tokamak parameter test spawn-safe; unblocked the macOS CI matrix.",
        "PR #565: stamp workflow metadata in the main process.",
      ],
    },
    {
      overline: "Configuration",
      heading: "Defaults belong in config, and each run should record what it used.",
      paragraphs: [
        "Defaults were scattered through the codebase as hardcoded literals. I scoped the migration into tiers, from trivial scalars through small design calls to architectural registries, and posted the analysis on the tracking issue before writing code. The first slice moved logging defaults into layered configuration via Dynaconf with an explicit precedence: explicit argument, then environment variable, then the user's TOML, then the repository TOML. The effective settings are written back into the configuration inside the main entry point, so the per-run config dump records what actually ran, for API callers as well as the CLI.",
        "The same PR fixed a log-file corruption that had been present since the spawn work: the main process opened the file in write mode and its offset overwrote lines the workers had appended. I reproduced it byte for byte in a standalone script and replaced it with truncate-once-then-append.",
      ],
      bullets: ["PR #569: resolve logging defaults from configuration."],
    },
    {
      overline: "In review",
      heading: "Explicit equilibrium selection, and workers that stop printing over the progress bar.",
      paragraphs: [
        "Most derived quantities come from EFIT, an equilibrium-reconstruction code, and a shot can have several EFIT runs at different time resolutions. Selecting one used to go through an implicit SQL-backed fallback. I replaced it with an explicit, user-specified cascade of tree names that falls through on missing trees and logs each step, backed by twenty-eight unit tests against a fake connection and a regression test on real C-Mod shots. Approved, pending merge.",
        "Worker processes also printed over the main process's tqdm progress bar, and lost their console output entirely when stdout was not a terminal. I compared candidate fixes empirically by capturing runs through a pseudo-terminal, traced the non-terminal loss to the logger's queue not being drained at worker exit, and shipped a queue-based console relay plus two spawn-safe tests. In review.",
      ],
      bullets: [
        "PR #559: explicit EFIT nickname cascade.",
        "PR #602: relay worker console lines through the main process.",
        "PR #576: portable lint targets. BSD find on macOS had been erroring silently and the lint target exited green having checked nothing.",
      ],
    },
    {
      overline: "Publication",
      heading: "APS Division of Plasma Physics, November 2026.",
      paragraphs: [
        "Poster: Maintainable and Extensible Software Architecture for Multi-Machine Disruption Studies with DisruptionPy. First author, with the MIT PSFC Disruption Studies Group. 68th Annual Meeting of the APS Division of Plasma Physics, Chicago, 2 to 6 November 2026.",
      ],
    },
  ],
  stack: [
    {
      label: "Data",
      items: [
        "MDSplus (thin-client fallback)",
        "xarray, zarr, netCDF4, fsspec (S3)",
        "pandas, numpy, scipy",
        "SQLAlchemy + pyodbc for metadata databases",
      ],
    },
    {
      label: "Framework",
      items: [
        "Python 3.12 / 3.13",
        "Dynaconf layered TOML configuration",
        "multiprocessing pools, fork / spawn / forkserver",
        "loguru, tqdm",
      ],
    },
    {
      label: "Tooling",
      items: [
        "uv and Poetry, PEP 621 packaging",
        "pytest against live lab servers over SSH tunnels in CI",
        "black, isort, ruff, pylint, shellcheck, yamllint, toml-sort",
        "MkDocs Material + mkdocstrings, GitHub Actions, Dependabot",
      ],
    },
  ],
  links: [
    { label: "DisruptionPy on GitHub", href: GH },
    { label: "Documentation", href: "https://mit-psfc.github.io/disruption-py/" },
    { label: "JOSS paper (2026)", href: "https://doi.org/10.21105/joss.09364" },
    { label: "My pull requests", href: `${GH}/pulls?q=is%3Apr+author%3Asamc24` },
  ],
};
