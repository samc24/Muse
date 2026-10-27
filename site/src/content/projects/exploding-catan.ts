import type { CaseStudy } from "../types";

export const explodingCatan: CaseStudy = {
  slug: "exploding-catan",
  kind: "project",
  name: "Exploding Catan",
  org: "Exploding Catan -- a browser board game with bots and online play",
  period: "Oct 2026 -- ongoing",
  tagline: "Rules engine, bots, multiplayer",
  summary:
    "Catan with the development deck torn out and 54 kitten cards put in its place. A friend's single-file browser game, rebuilt as a deterministic rules engine, bots that play on their own redacted view, and an authoritative multiplayer server, so game night works with a room code and nobody's laptop has to host.",
  accent: "violet",
  current: true,
  highlights: [
    "Rebuilt a 10,000-line single-file browser game as a TypeScript monorepo: a pure rules engine, bots, a wire protocol, a plain DOM and SVG web app, and a WebSocket server. 77 commits and 23 pull requests in the first week, every one reviewed into master.",
    "The engine is a pure, deterministic, clockless reducer over plain JSON state. Same seed and same actions give the same game byte for byte, so replays, bug reports, and self-play tuning all work, and a whole game runs in tests in milliseconds.",
    "Bots are pure functions that decide from the same redacted view a player gets. Fairness is a tested property: at every decision the test permutes the deck and swaps every other hand and requires the identical choice.",
    "Online play runs on an authoritative server: rooms own the real state, each seat receives only what it may see, bots and clocks run server-side so a table keeps playing with nobody's tab open, and a session token hands a reloading player their seat back.",
    "Engine 73, bots 8, server 12 unit tests, plus headless-Chrome tests that play through setup, every card flow, a table of bots, a phone, and two browsers at one server table. CI gates the deploy to GitHub Pages.",
    "A non-commercial fan fork of the original game, made with its author's blessing. The Catan and Exploding Kittens names and the card art belong to their owners; see the repository notice.",
  ],
  sections: [
    {
      overline: "Why",
      heading: "Game night needed a server.",
      paragraphs: [
        "A friend built Exploding Catan as one HTML file: Catan, but the development deck is replaced with 54 kitten cards that let you Nope a whole turn, defuse an attack back onto its sender, or swap two numbers on the board. We had been playing it on laptops at game night. The catch was that the whole game ran in the host's browser and mirrored state through a database, so the host could read every hand, the game died when their tab closed, and anyone with a room code could write anything.",
        "I forked it, with his blessing, to rebuild it as something that could be developed, tested, and run as a public service without losing the rules that make it the game it is. The approach is a strangler fig: the rules first, then the bots, then the browser app, then the server, each a package that only imports downward.",
      ],
    },
    {
      overline: "Engine",
      heading: "Every decision is state, not a dialog.",
      paragraphs: [
        "The engine is a reducer: create a game from a config and a seed, apply an action with the current time, get back a new state and a list of events, or a refusal with a reason. It never mutates its input. All randomness comes from a seeded generator whose state lives on the game state, and nothing reads the clock; the host passes the time with every action and sends a tick to drive the turn timer. A rule that says no throws a typed error that the reducer returns as a refusal; anything else is a bug and is thrown, so a server never plays on past a crash it mistook for a refusal.",
        "Everything the original asked for in a modal, a discard, a victim, three cards for a Favor, is a field on the state that the UI renders and the bots answer. Nothing can be lost by closing a window because there are no windows. The part I kept whole is the journal with snapshots that makes Nope and Defuse work: every answerable action records the state before it, a Nope rewinds and re-charges the cards that were spent, and a cancelled win costs every Feral Kitten. Each of those rules has a test, mostly transcribed from the bug post-mortems in the original's README.",
        "The view a seat gets is itself a game state: its own cards, face-down placeholders for everyone else's, a deck of the right length with nothing in it, and no snapshots. The web app renders and queries it with the same code it uses for a local game.",
      ],
    },
    {
      overline: "Bots and server",
      heading: "Bots see what a player sees. The server sees everything.",
      paragraphs: [
        "The bots are the original strategy ported to pure functions. A bot is asked for the single action it wants next, from its own redacted view, and the host applies it through the engine like any other player's. Each bot carries its own seeded generator, so a table of bots is a pure function of its seeds: the self-play test plays whole games across seeds, player counts, modes, and boards, checks that every action it chose was legal, checks the bank balances, and replays a seed to prove determinism.",
        "Online, a room owns the real state. Clients send actions, the server stamps the sender's seat on them, applies them, and sends each seat its own view. A dropped connection starts a twelve-second clock rather than a takeover; past it, mid-game, a bot plays the seat and the chair stays theirs. Rooms and sessions persist through a small store interface so a deploy costs players a reconnect, not the game. The server runs on a free tier that sleeps when idle, so the join dialog knocks on a health endpoint to wake it; the always-on config is written and waiting for when the game is worth a few dollars a month.",
      ],
      bullets: [
        "One Docker image serves the web app and the WebSocket on one port.",
        "First deploy died on start: a multi-source COPY in the Dockerfile flattened every src directory one level up. Docker was not installed locally, so I replicated the runtime layout in a scratch directory to reproduce it.",
        "A redeploy mid-game lost every room (memory store). Auto-deploy is now off and the client treats a fresh welcome while at a table as table lost, with a way back to the lobby.",
      ],
    },
    {
      overline: "Web app",
      heading: "No framework. One render, from the state.",
      paragraphs: [
        "The browser app is plain DOM and SVG with a single render function that rebuilds each region of the page from the state. The source is in layers, and a lint script fails the build if an import points upward. Regions compare before writing, so a tick that moved nothing but the clocks repaints the clocks in place instead of throwing away every animation on the page. Gestures are not rules: arming a spot on the first click, what a drop on the pile answers, when the chooser opens, all live in an input layer that asks the engine what a card could answer right now.",
        "Most of the design work since parity has been the phone. The original was very tough to trade and use the bank on mobile. Portrait hangs the board from a furniture strip with the dice, clock, and piles; landscape puts the seats and decisions in a side column so the board stays in view; a hold lifts a card and a shorter touch scrolls the hand. Everything that moves is decoration layered over the state and never read back by the rules, and a Motion setting turns it down or off.",
      ],
    },
    {
      overline: "Next",
      heading: "Make it mine, then make it stranger.",
      paragraphs: [
        "Two things on the list. First, make the game legally clean of the Catan and Exploding Kittens names, art, card text, and music, which is why the code is under a non-commercial license rather than MIT. Second, a fourth mode where each player picks a character with a superpower. The three existing modes are checks scattered through the reducer; gathering them into one ruleset object the reducer consults is the step that makes a power a registered hook instead of another branch.",
      ],
    },
  ],
  stack: [
    {
      label: "Language and build",
      items: ["TypeScript end to end", "npm workspaces monorepo", "Vite", "ESLint + a custom import-layer check"],
    },
    {
      label: "Runtime",
      items: ["Plain DOM and SVG, no framework", "Node 22 WebSocket server", "Docker, one image for web + socket"],
    },
    {
      label: "Testing",
      items: ["Vitest: engine, bots, protocol, server", "playwright-core headless Chrome e2e", "Self-play and fairness tests over seeded games", "GitHub Actions CI gating GitHub Pages"],
    },
  ],
  links: [
    { label: "Play it", href: "https://samc24.github.io/Exploding-Catan/" },
    { label: "Source on GitHub", href: "https://github.com/samc24/Exploding-Catan" },
    { label: "Architecture notes", href: "https://github.com/samc24/Exploding-Catan/blob/master/docs/ARCHITECTURE.md" },
  ],
  posterImage: "/projects/exploding-catan.jpg",
};
