import Link from "next/link";
import Image from "next/image";

type ProjectCardProps = {
  href: string;
  logo: string;
  logoAlt: string;
  name: string;
  tagline: string;
  summary: string;
  posterVideo: string;
  accent: "orange" | "blue";
  invertLogo?: boolean;
};

const accentMap: Record<
  ProjectCardProps["accent"],
  { border: string; label: string; glow: string }
> = {
  orange: {
    border: "hover:border-accent/70",
    label: "text-accent",
    glow: "group-hover:shadow-[0_0_0_1px_rgba(249,115,22,0.25),0_20px_50px_-15px_rgba(249,115,22,0.25)]",
  },
  blue: {
    border: "hover:border-ft-blue/70",
    label: "text-ft-blue",
    glow: "group-hover:shadow-[0_0_0_1px_rgba(59,130,246,0.25),0_20px_50px_-15px_rgba(59,130,246,0.25)]",
  },
};

export function ProjectCard({
  href,
  logo,
  logoAlt,
  name,
  tagline,
  summary,
  posterVideo,
  accent,
  invertLogo = false,
}: ProjectCardProps) {
  const style = accentMap[accent];
  return (
    <Link
      href={href}
      className={`group relative flex flex-col rounded-2xl border hairline overflow-hidden bg-surface-raised transition-all duration-300 ${style.border} ${style.glow}`}
    >
      <div className="relative aspect-[16/9] w-full overflow-hidden bg-black">
        <video
          src={posterVideo}
          muted
          loop
          playsInline
          preload="metadata"
          autoPlay
          className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-surface-raised via-transparent to-transparent" />
      </div>
      <div className="p-7">
        <div className="flex items-center gap-4 mb-5">
          <Image
            src={logo}
            alt={logoAlt}
            width={56}
            height={56}
            className={`rounded-lg ${invertLogo ? "invert" : ""}`}
          />
          <div>
            <h3 className="text-2xl font-semibold tracking-tight text-foreground leading-none">
              {name}
            </h3>
            <p
              className={`mt-2 text-xs uppercase tracking-[0.2em] font-semibold ${style.label}`}
            >
              {tagline}
            </p>
          </div>
        </div>
        <p className="mt-3 text-muted-strong leading-relaxed">{summary}</p>
        <span className="mt-6 inline-flex items-center gap-2 text-sm text-muted group-hover:text-foreground transition-colors">
          Explore
          <span
            aria-hidden
            className="transition-transform group-hover:translate-x-1"
          >
            -&gt;
          </span>
        </span>
      </div>
    </Link>
  );
}
