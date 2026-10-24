import type { MetadataRoute } from "next";
import { WORK, PROJECTS, pagedEntries, hrefFor } from "@/content";

const BASE = "https://sameerc.com";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const fixed: MetadataRoute.Sitemap = [
    { url: `${BASE}/`, lastModified: now, changeFrequency: "monthly", priority: 1.0 },
    { url: `${BASE}/work`, lastModified: now, changeFrequency: "monthly", priority: 0.9 },
    { url: `${BASE}/projects`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE}/muse`, lastModified: now, changeFrequency: "monthly", priority: 0.8 },
    { url: `${BASE}/follow-through`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE}/eyeball`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE}/about`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
  ];
  const entries: MetadataRoute.Sitemap = [
    ...pagedEntries(WORK),
    ...pagedEntries(PROJECTS),
  ].map((e) => ({
    url: `${BASE}${hrefFor(e)}`,
    lastModified: now,
    changeFrequency: "monthly",
    priority: 0.6,
  }));
  return [...fixed, ...entries];
}
