import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { WORK, findWork, pagedEntries } from "@/content";
import { CaseStudyPage } from "@/components/CaseStudyPage";

type Params = { slug: string };

export const dynamicParams = false;

export function generateStaticParams(): Params[] {
  return pagedEntries(WORK).map((w) => ({ slug: w.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<Params>;
}): Promise<Metadata> {
  const { slug } = await params;
  const entry = findWork(slug);
  if (!entry) return {};
  return {
    title: entry.org,
    description: entry.summary,
    openGraph: {
      title: `${entry.org} -- Sameer Chaturvedi`,
      description: entry.summary,
    },
  };
}

export default async function WorkEntryPage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { slug } = await params;
  const entry = findWork(slug);
  if (!entry || entry.href) notFound();
  return <CaseStudyPage entry={entry} siblings={pagedEntries(WORK)} />;
}
