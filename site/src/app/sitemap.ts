import type { MetadataRoute } from "next";

const BASE = "https://sameerc.com";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  return [
    { url: `${BASE}/`, lastModified: now, changeFrequency: "monthly", priority: 1.0 },
    { url: `${BASE}/follow-through`, lastModified: now, changeFrequency: "monthly", priority: 0.8 },
    { url: `${BASE}/eyeball`, lastModified: now, changeFrequency: "monthly", priority: 0.8 },
    { url: `${BASE}/about`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
  ];
}
