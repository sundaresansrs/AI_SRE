import type { ReactNode } from "react";
import type { Verdict } from "../../lib/types";

export type Tone = "accept" | "review" | "reject" | "info" | "neutral";

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`badge badge--${tone}`}>{children}</span>;
}

const VERDICT_TONE: Record<Verdict, Tone> = { ACCEPT: "accept", REVIEW: "review", REJECT: "reject" };

export function VerdictBadge({ verdict }: { verdict: Verdict | null }) {
  if (!verdict) {
    return <Badge>Unscored</Badge>;
  }
  return <Badge tone={VERDICT_TONE[verdict]}>{verdict}</Badge>;
}
