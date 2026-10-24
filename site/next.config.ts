import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    // Short, printable URLs for the QR codes on /qr. Temporary (307) on
    // purpose: the targets can change later without reprinting anything,
    // and browsers will not cache the hop. UTM tags surface in Vercel Web
    // Analytics so scans can be told apart from organic visits.
    return [
      {
        source: "/q",
        destination: "/?utm_source=qr&utm_medium=card",
        permanent: false,
      },
      {
        source: "/q/muse",
        destination: "/muse?utm_source=qr&utm_medium=card",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
