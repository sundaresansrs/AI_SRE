import type { Filters, Tab } from "../lib/incidents";
import type { ActionFilter, SortKey, VerdictFilter } from "../lib/incidents";
import { Icon } from "./ui/Icon";
import { Select } from "./ui/Select";
import type { SelectOption } from "./ui/Select";
import { Tabs } from "./ui/Tabs";

const VERDICTS: SelectOption<VerdictFilter>[] = [
  { value: "all", label: "All verdicts" },
  { value: "ACCEPT", label: "Accept" },
  { value: "REVIEW", label: "Review" },
  { value: "REJECT", label: "Reject" },
  { value: "unscored", label: "Unscored" },
];

const ACTIONS: SelectOption<ActionFilter>[] = [
  { value: "all", label: "All actions" },
  { value: "restart_deployment", label: "Restart deployment" },
  { value: "scale_deployment", label: "Scale deployment" },
  { value: "none", label: "No action" },
];

const SORTS: SelectOption<SortKey>[] = [
  { value: "newest", label: "Newest first" },
  { value: "oldest", label: "Oldest first" },
  { value: "trust-desc", label: "Trust: high to low" },
  { value: "trust-asc", label: "Trust: low to high" },
];

export function Toolbar({
  filters,
  counts,
  updatedAt,
  refreshing,
  onChange,
  onRefresh,
}: {
  filters: Filters;
  counts: Record<Tab, number>;
  updatedAt: string;
  refreshing: boolean;
  onChange: (patch: Partial<Filters>) => void;
  onRefresh: () => void;
}) {
  return (
    <div className="toolbar">
      <div className="toolbar__top">
        <Tabs<Tab>
          label="Incident status"
          idPrefix="main"
          value={filters.tab}
          onChange={(tab) => onChange({ tab })}
          items={[
            { id: "review", label: "Needs review", count: counts.review },
            { id: "approved", label: "Approved", count: counts.approved },
            { id: "rejected", label: "Rejected", count: counts.rejected },
            { id: "all", label: "All", count: counts.all },
          ]}
        />
        <div className="updated">
          {updatedAt ? <span>Updated {updatedAt}</span> : null}
          <button type="button" className="btn btn--sm" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? <span className="spinner" aria-hidden="true" /> : <Icon name="refresh" size={14} />}
            Refresh
          </button>
        </div>
      </div>

      <div className="toolbar__filters">
        <label className="field search">
          <span className="field__label">Search</span>
          <span style={{ position: "relative", display: "block" }}>
            <span className="search__icon">
              <Icon name="search" size={15} />
            </span>
            <input
              className="input"
              type="search"
              placeholder="Alert text, deployment, #id…"
              value={filters.query}
              onChange={(event) => onChange({ query: event.target.value })}
              style={{ paddingLeft: 36 }}
            />
          </span>
        </label>
        <Select label="Verdict" value={filters.verdict} options={VERDICTS} onChange={(verdict) => onChange({ verdict })} />
        <Select label="Action" value={filters.action} options={ACTIONS} onChange={(action) => onChange({ action })} />
        <Select label="Sort by" value={filters.sort} options={SORTS} onChange={(sort) => onChange({ sort })} />
      </div>
    </div>
  );
}
