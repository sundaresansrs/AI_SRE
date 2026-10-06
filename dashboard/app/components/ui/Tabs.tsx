import type { KeyboardEvent } from "react";

export type TabItem<T extends string> = { id: T; label: string; count?: number };

export const tabId = (prefix: string, id: string) => `${prefix}-tab-${id}`;
export const panelId = (prefix: string, id: string) => `${prefix}-panel-${id}`;

/** Accessible tab strip: roving tabindex and arrow-key navigation. Render a role="tabpanel" with panelId(). */
export function Tabs<T extends string>({
  items,
  value,
  onChange,
  label,
  idPrefix,
  variant = "pill",
}: {
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  idPrefix: string;
  variant?: "pill" | "underline";
}) {
  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const index = items.findIndex((item) => item.id === value);
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % items.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + items.length) % items.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = items.length - 1;
    else return;
    event.preventDefault();
    onChange(items[next].id);
    requestAnimationFrame(() => document.getElementById(tabId(idPrefix, items[next].id))?.focus());
  }

  return (
    <div
      className={variant === "underline" ? "tabs tabs--underline" : "tabs"}
      role="tablist"
      aria-label={label}
      onKeyDown={onKeyDown}
    >
      {items.map((item) => {
        const selected = item.id === value;
        return (
          <button
            key={item.id}
            id={tabId(idPrefix, item.id)}
            type="button"
            role="tab"
            className="tab"
            aria-selected={selected}
            aria-controls={panelId(idPrefix, item.id)}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.id)}
          >
            {item.label}
            {item.count !== undefined ? <span className="tab__count">{item.count}</span> : null}
          </button>
        );
      })}
    </div>
  );
}
