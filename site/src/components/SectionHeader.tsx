type SectionHeaderProps = {
  overline?: string;
  title: string;
  lede?: string;
  accentColor?: string;
};

export function SectionHeader({
  overline,
  title,
  lede,
  accentColor,
}: SectionHeaderProps) {
  return (
    <div className="max-w-3xl">
      {overline && (
        <p
          className="text-xs uppercase tracking-[0.2em] font-semibold"
          style={{ color: accentColor ?? "var(--color-accent)" }}
        >
          {overline}
        </p>
      )}
      <h1 className="mt-3 text-4xl sm:text-5xl font-semibold tracking-tight text-foreground">
        {title}
      </h1>
      {lede && (
        <p className="mt-5 text-lg text-muted-strong leading-relaxed">{lede}</p>
      )}
    </div>
  );
}
