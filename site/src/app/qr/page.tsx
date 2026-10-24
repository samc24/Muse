import Image from "next/image";

export const metadata = {
  title: "QR codes",
  description: "Scannable links to sameerc.com for cards, posters, and slides.",
  robots: { index: false, follow: false },
};

type Code = {
  label: string;
  url: string;
  display: string;
  svg: string;
  png: string;
};

const CODES: Code[] = [
  {
    label: "Everything",
    url: "https://sameerc.com/q",
    display: "sameerc.com/q",
    svg: "/qr/sameerc.svg",
    png: "/qr/sameerc.png",
  },
  {
    label: "Basketball",
    url: "https://sameerc.com/q/muse",
    display: "sameerc.com/q/muse",
    svg: "/qr/sameerc-muse.svg",
    png: "/qr/sameerc-muse.png",
  },
];

export default function QrPage() {
  return (
    <section className="relative overflow-hidden">
      <div className="hero-aura print:hidden" aria-hidden />
      <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-24 sm:pt-24">
        <div className="print:hidden">
          <p className="overline">QR codes</p>
          <h1 className="mt-5 h-display max-w-4xl">
            Scan to
            <br />
            <span className="text-muted">get here.</span>
          </h1>
          <p className="mt-8 text-lg sm:text-xl text-muted-strong max-w-3xl leading-relaxed">
            For business cards, posters, and slides. The first code lands on
            the home page; the second lands on Muse for a basketball
            audience. Both go through a short redirect that tags the visit
            as a scan in analytics and can be repointed later without
            reprinting. SVG scales to any size for print. PNG is 1024px for
            slides and messages.
          </p>
        </div>

        <div className="mt-12 print:mt-0 grid sm:grid-cols-2 gap-8 print:gap-16">
          {CODES.map((c) => (
            <div key={c.url} className="flex flex-col items-start">
              <a
                href={c.url}
                className="block rounded-2xl bg-white p-6 sm:p-8 w-full max-w-sm print:max-w-none"
                aria-label={`QR code for ${c.display}`}
              >
                <Image
                  src={c.svg}
                  alt={`QR code linking to ${c.display}`}
                  width={512}
                  height={512}
                  className="w-full h-auto"
                  unoptimized
                  priority
                />
              </a>
              <p className="mt-5 overline print:text-black">{c.label}</p>
              <p className="mt-2 text-xl font-semibold tracking-tight text-foreground print:text-black">
                {c.display}
              </p>
              <div className="mt-5 flex flex-wrap gap-3 print:hidden">
                <a href={c.svg} download className="btn btn-primary">
                  Download SVG
                </a>
                <a href={c.png} download className="btn btn-secondary">
                  Download PNG
                </a>
              </div>
            </div>
          ))}
        </div>

        <p className="mt-12 text-sm text-muted print:hidden">
          Error correction is set to the highest level, so a logo over the
          centre or a scuffed print still scans.
        </p>
      </div>
    </section>
  );
}
