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
        preload="metadata"
        poster={poster}
        src={src}
      />
    </div>
  );
}
