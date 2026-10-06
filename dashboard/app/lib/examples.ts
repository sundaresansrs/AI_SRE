export type ExampleAlert = { label: string; alert: string };
export type ExampleGroup = { group: string; items: ExampleAlert[] };

const TAIL = "Investigate the real cluster state.";

/** Alerts that match the broken workloads in infra/scenarios and the Online Boutique shop. */
export const EXAMPLE_ALERTS: ExampleGroup[] = [
  {
    group: "Lab scenarios (namespace sre-lab)",
    items: [
      {
        label: "Out of memory (lab-oom-killed)",
        alert: `Deployment lab-oom-killed in namespace sre-lab keeps restarting and its pod is not staying up. ${TAIL}`,
      },
      {
        label: "Crash loop (lab-crashloop)",
        alert: `Deployment lab-crashloop in namespace sre-lab is crash-looping and never becomes ready. ${TAIL}`,
      },
      {
        label: "Image cannot be pulled (lab-image-pull)",
        alert: `Deployment lab-image-pull in namespace sre-lab has no running pods and never becomes available. ${TAIL}`,
      },
      {
        label: "Pod stuck pending (lab-pending)",
        alert: `Deployment lab-pending in namespace sre-lab has a pod that never starts. ${TAIL}`,
      },
      {
        label: "Readiness probe failing (lab-readiness-fail)",
        alert: `Deployment lab-readiness-fail in namespace sre-lab is running but never becomes ready. ${TAIL}`,
      },
      {
        label: "Scaled to zero (lab-scaled-to-zero)",
        alert: `Deployment lab-scaled-to-zero in namespace sre-lab has no running pods and the service it backs is returning errors. ${TAIL}`,
      },
      {
        label: "Missing configuration (lab-missing-config)",
        alert: `Deployment lab-missing-config in namespace sre-lab cannot start its container. ${TAIL}`,
      },
      {
        label: "Healthy control (lab-healthy)",
        alert: `Monitoring flagged deployment lab-healthy in namespace sre-lab for a possible problem. ${TAIL}`,
      },
    ],
  },
  {
    group: "Online Boutique (namespace online-boutique)",
    items: [
      {
        label: "cartservice errors",
        alert: `Deployment cartservice in namespace online-boutique is returning errors and its pods keep restarting. ${TAIL}`,
      },
      {
        label: "Slow checkout",
        alert: `Deployment checkoutservice in namespace online-boutique has high latency and failing requests. ${TAIL}`,
      },
    ],
  },
];
