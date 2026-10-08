import { formatDateTime, summarizeAction, timeAgo } from "../lib/format";
import type { Incident } from "../lib/types";
import { IncidentDetail } from "./IncidentDetail";
import type { RowAction } from "./IncidentDetail";
import { Badge, VerdictBadge } from "./ui/Badge";
import { Icon } from "./ui/Icon";
import { TrustMeter } from "./ui/TrustMeter";

function ApprovalBadge({ status }: { status: Incident["approval_status"] }) {
  if (status === "approved") return <Badge tone="info">Approved</Badge>;
  if (status === "rejected") return <Badge tone="reject">Rejected</Badge>;
  return <Badge>Pending</Badge>;
}

function ExecutionBadge({ incident }: { incident: Incident }) {
  if (incident.execution_status === "running") {
    return (
      <Badge tone="info">
        <span className="spinner" aria-hidden="true" style={{ width: 10, height: 10, borderWidth: 2 }} />
        Executing
      </Badge>
    );
  }
  if (incident.executed) return <Badge tone="accept">Executed</Badge>;
  if (incident.execution_status === "failed") return <Badge tone="reject">Execution failed</Badge>;
  return null;
}

export function IncidentRow({
  incident,
  now,
  expanded,
  busyAction,
  onToggle,
  onApprove,
  onReject,
  onExecute,
}: {
  incident: Incident;
  now: number;
  expanded: boolean;
  busyAction: RowAction | undefined;
  onToggle: () => void;
  onApprove: () => void;
  onReject: () => void;
  onExecute: () => void;
}) {
  const action = summarizeAction(incident);

  return (
    <article
      id={`incident-${incident.id}`}
      className={expanded ? "incident incident--open" : "incident"}
      data-testid={`incident-${incident.id}`}
    >
      <button
        type="button"
        className="incident__summary"
        aria-expanded={expanded}
        aria-controls={`incident-${incident.id}-detail`}
        onClick={onToggle}
      >
        <span className="incident__chevron">
          <Icon name="chevron" />
        </span>
        <span className="incident__id">#{incident.id}</span>
        <span className="incident__main">
          <span className="incident__alert">{incident.alert}</span>
          <span className="incident__meta">
            <VerdictBadge verdict={incident.classification} />
            <TrustMeter score={incident.trust_score} />
            <span className={action.executable ? "chip" : "chip chip--muted"}>
              {action.executable ? <Icon name="bolt" size={12} /> : null}
              <span>{action.short}</span>
              {action.target ? <code>{action.target}</code> : null}
            </span>
            <ApprovalBadge status={incident.approval_status} />
            <ExecutionBadge incident={incident} />
          </span>
        </span>
        <span className="incident__side">
          <span className="time" title={formatDateTime(incident.created_at)}>
            {timeAgo(incident.created_at, now)}
          </span>
        </span>
      </button>

      {expanded ? (
        <IncidentDetail
          incident={incident}
          busyAction={busyAction}
          onApprove={onApprove}
          onReject={onReject}
          onExecute={onExecute}
        />
      ) : null}
    </article>
  );
}
