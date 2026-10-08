import type { Incident } from "./types";

// All calls go through the server-side relay in app/api/sre, which adds the backend API key.
const BASE = "/api/sre";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function describeError(body: unknown, response: Response): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  if (typeof body === "string" && body) {
    return body.slice(0, 300);
  }
  return `HTTP ${response.status} ${response.statusText}`.trim();
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { ...init, cache: "no-store" });
  } catch (error) {
    throw new ApiError(error instanceof Error ? error.message : "Network error", 0);
  }

  const text = await response.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!response.ok) {
    throw new ApiError(describeError(body, response), response.status);
  }
  return body as T;
}

export const api = {
  list: () => request<Incident[]>("/incidents"),
  approve: (id: number) => request<Incident>(`/incidents/${id}/approve`, { method: "POST" }),
  reject: (id: number) => request<Incident>(`/incidents/${id}/reject`, { method: "POST" }),
  execute: (id: number) => request<Incident>(`/incidents/${id}/execute`, { method: "POST" }),
  analyze: (alert: string) =>
    request<Incident>("/incidents/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alert }),
    }),
};
