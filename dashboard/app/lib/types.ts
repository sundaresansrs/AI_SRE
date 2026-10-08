export type Verdict = "ACCEPT" | "REVIEW" | "REJECT";
export type ApprovalStatus = "pending" | "approved" | "rejected";
export type ExecutionStatus = "running" | "succeeded" | "failed";
export type ActionName = "restart_deployment" | "scale_deployment" | "none";

export type DeploymentSnapshot = {
  name?: string;
  desired_replicas?: number;
  available_replicas?: number;
  ready_replicas?: number;
  updated_replicas?: number;
  unavailable_replicas?: number;
  generation?: number;
  observed_generation?: number;
  error?: string;
};

export type ExecutionResult = {
  executed?: boolean;
  action?: string;
  reason?: string;
  verification_status?: string;
  execution_trigger?: string;
  before_state?: DeploymentSnapshot | null;
  after_state?: DeploymentSnapshot | null;
  [key: string]: unknown;
};

export type Incident = {
  id: number;
  alert: string;
  plan: string | null;
  diagnosis: string | null;
  proposed_fix: string | null;
  verifier_reasoning: string | null;
  trust_score: number | null;
  classification: Verdict | null;
  approval_status: ApprovalStatus;
  approval_updated_at: string | null;
  created_at: string;
  recommended_action: ActionName | null;
  action_namespace: string | null;
  action_deployment_name: string | null;
  action_replicas: number | null;
  executed: boolean;
  executed_at: string | null;
  execution_trigger: string | null;
  execution_status: ExecutionStatus | null;
  execution_result: ExecutionResult | null;
};
