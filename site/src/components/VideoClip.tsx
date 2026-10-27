import { posterFor } from "@/lib/media";

type VideoClipProps = {
  src: string;
  poster?: string;
  className?: string;
};

export function VideoClip({ src, poster, className = "" }: VideoClipProps) {
  return (
    <div className={`overflow-hidden rounded-xl border hairline bg-black ${className}`}>
      <video
        controls
        muted
        playsInline
        preload="none"
        poster={poster ?? posterFor(src)}
        src={src}
      />
    </div>
  );
}
