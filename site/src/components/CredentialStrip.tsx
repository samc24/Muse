type Credential = {
  company: string;
  role: string;
  period: string;
  note?: string;
};

const CREDENTIALS: Credential[] = [
  { company: "Amazon", role: "ML Engineer II", period: "2025 -- now", note: "Prime Video" },
  { company: "MIT PSFC", role: "Lead Software Engineer", period: "2025 -- now", note: "DisruptionPy" },
  { company: "Cerebro Sports", role: "Lead ML Engineer", period: "2025", note: "Live gameplay" },
  { company: "Microsoft", role: "Software Engineer II", period: "2021 -- 2025", note: "M365 Security" },
  { company: "AI.Reverie", role: "ML Engineer", period: "2020", note: "Computer vision" },
  { company: "Red Hat", role: "Software Engineer", period: "2019", note: "OpenShift" },
];

export function CredentialStrip() {
  return (
    <div className="relative">
      <p className="overline">Worked at</p>
      <div className="mt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-x-8 gap-y-6">
        {CREDENTIALS.map((c) => (
          <div
            key={c.company}
            className="flex flex-col border-l hairline-strong pl-4"
          >
            <span className="text-sm font-semibold tracking-tight text-foreground">
              {c.company}
            </span>
            <span className="mt-1 text-xs text-muted leading-snug">
              {c.role}
            </span>
            <span className="mt-2 text-[10px] uppercase tracking-[0.15em] text-muted/70 font-mono">
              {c.period}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
