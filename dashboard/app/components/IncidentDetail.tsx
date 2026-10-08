import { useState } from "react";
import type { ReactNode } from "react";
import { formatDateTime, summarizeAction } from "../lib/format";
import type { Incident } from "../lib/types";
import { ExecutionPanel } from "./ExecutionPanel";
import { Badge, VerdictBadge } from "./ui/Badge";
import { Icon } from "./ui/Icon";
import { Markdown } from "./ui/Markdown";
import { Tabs, panelId, tabId } from "./ui/Tabs";
import type { TabItem } from "./ui/Tabs";
import { TrustMeter } from "./ui/TrustMeter";

type DetailTab = "overview" | "diagnosis" | "fix" | "plan" | "execution";
export type RowAction = "approve" | "reject" | "execute";

function Reading({ text, empty }: { text: string | null; empty: string }) {
  return text ? (
    <div className="reading">
      <Markdown>{text}</Markdown>
    </div>
  ) : (
    <p className="panel__sub">{empty}</p>
  );
}

function Overview({ incident }: { incident: Incident }) {
  const action = summarizeAction(incident);
  return (
    <div className="detail__panel">
      <p className="alert-text">{incident.alert}</p>

      <div className="detail__grid">
        <div className="card">
          <span className="card__title">Recommended action</span>
          <dl className="kv">
            <dt>Action</dt>
            <dd>{action.label}</dd>
            {action.target ? (
              <>
                <dt>Target</dt>
                <dd>
                  <code>{action.target}</code>
                </dd>
              </>
            ) : null}
            {incident.recommended_action === "scale_deployment" ? (
              <>
                <dt>Replicas</dt>
                <dd>{incident.action_replicas ?? "-"}</dd>
              </>
            ) : null}
          </dl>
          {!action.executable ? (
            <span className="panel__sub">Nothing here can be executed from the dashboard; follow the proposed fix manually.</span>
          ) : null}
        </div>

        <div className="card">
          <span className="card__title">Verdict</span>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <VerdictBadge verdict={incident.classification} />
            <TrustMeter score={incident.trust_score} large />
          </div>
          {incident.verifier_reasoning ? <Markdown>{incident.verifier_reasoning}</Markdown> : null}
        </div>

        <div className="card">
          <span className="card__title">Timeline</span>
          <ul className="timeline">
            <li>
              Analyzed
              <span className="timeline__time">{formatDateTime(incident.created_at)}</span>
            </li>
            {incident.approval_updated_at ? (
              <li>
                {incident.approval_status === "approved" ? "Approved" : "Rejected"}
                <span className="timeline__time">{formatDateTime(incident.approval_updated_at)}</span>
              </li>
            ) : null}
            {incident.executed_at ? (
              <li>
                Executed ({incident.execution_trigger ?? "manual"})
                <span className="timeline__time">{formatDateTime(incident.executed_at)}</span>
              </li>
            ) : null}
          </ul>
        </div>
      </div>
    </div>
  );
}

/** Why the action buttons are in their current state, in one sentence. */
function actionHint(incident: Incident, executable: boolean, running: boolean): string {
  if (incident.executed) return "This incident has been executed. Approval can no longer change.";
  if (running) return "Execution is in progress. The executor will verify the rollout before finishing.";
  if (incident.classification === "REJECT") return "The verifier rejected this incident, so it cannot be approved.";
  if (incident.approval_status === "rejected") return "Rejected. You can still approve it if you change your mind.";
  if (!executable) return "No executable action. Approving only records that a human reviewed it.";
  if (incident.approval_status === "approved") return "Approved. Execute applies the action to the live cluster and verifies the result.";
  return "Nothing changes in the cluster until you approve and then click Execute.";
}

export function IncidentDetail({
  incident,
  busyAction,
  onApprove,
  onReject,
  onExecute,
}: {
  incident: Incident;
  busyAction: RowAction | undefined;
  onApprove: () => void;
  onReject: () => void;
  onExecute: () => void;
}) {
  const [tab, setTab] = useState<DetailTab>("overview");
  const prefix = `incident-${incident.id}`;

  const items: TabItem<DetailTab>[] = [
    { id: "overview", label: "Overview" },
    { id: "diagnosis", label: "Diagnosis" },
    { id: "fix", label: "Proposed fix" },
    { id: "plan", label: "Plan" },
    ...(incident.execution_result ? [{ id: "execution" as const, label: "Execution" }] : []),
  ];
  const active = items.some((item) => item.id === tab) ? tab : "overview";

  const action = summarizeAction(incident);
  const running = incident.execution_status === "running" || busyAction === "execute";
  const locked = incident.executed || running || busyAction !== undefined;
  const canApprove = !locked && incident.classification !== "REJECT" && incident.approval_status !== "approved";
  const canReject = !locked && incident.approval_status !== "rejected";
  const canExecute =
    !locked && incident.approval_status === "approved" && action.executable && !incident.executed;

  let content: ReactNode;
  switch (active) {
    case "diagnosis":
      content = <Reading text={incident.diagnosis} empty="The agent did not produce a diagnosis." />;
      break;
    case "fix":
      content = <Reading text={incident.proposed_fix} empty="The agent did not propose a fix." />;
      break;
    case "plan":
      content = <Reading text={incident.plan} empty="No investigation plan was recorded." />;
      break;
    case "execution":
      content = <ExecutionPanel incident={incident} />;
      break;
    default:
      content = <Overview incident={incident} />;
  }

  return (
    <div className="detail" id={`${prefix}-detail`}>
      <Tabs items={items} value={active} onChange={setTab} label="Incident sections" idPrefix={prefix} variant="underline" />
      <div role="tabpanel" id={panelId(prefix, active)} aria-labelledby={tabId(prefix, active)}>
        {content}
      </div>

      <div className="actions">
        <button type="button" className="btn btn--accept" onClick={onApprove} disabled={!canApprove}>
          {busyAction === "approve" ? <span className="spinner" aria-hidden="true" /> : <Icon name="check" />}
          Approve
        </button>
        <button type="button" className="btn btn--danger" onClick={onReject} disabled={!canReject}>
          {busyAction === "reject" ? <span className="spinner" aria-hidden="true" /> : <Icon name="x" />}
          Reject
        </button>
        <span className="actions__spacer" />
        {incident.executed ? <Badge tone="accept">Executed</Badge> : null}
        <button type="button" className="btn btn--primary" onClick={onExecute} disabled={!canExecute}>
          {running ? <span className="spinner" aria-hidden="true" /> : <Icon name="play" />}
          {running ? "Executing…" : "Execute"}
        </button>
        <p className="actions__hint">
          <Icon name="info" size={14} />
          {actionHint(incident, action.executable, running)}
        </p>
      </div>
    </div>
  );
}
