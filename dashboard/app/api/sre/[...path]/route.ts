// Server-side relay to the FastAPI backend. The browser only ever talks to this route, so the
// backend API key (AI_SRE_API_KEY) stays on the server and is never shipped in the client bundle.

const BACKEND_URL = (
  process.env.AI_SRE_API_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

// Analysis runs the whole agent graph (MCP tools, RAG, several LLM calls).
export const maxDuration = 300;

type RouteParams = { params: Promise<{ path: string[] }> };

async function relay(request: Request, { params }: RouteParams): Promise<Response> {
  const { path } = await params;
  if (path[0] !== "incidents") {
    return Response.json({ detail: "not found" }, { status: 404 });
  }

  const target = `${BACKEND_URL}/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;
  const headers = new Headers({ Accept: "application/json" });
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("Content-Type", contentType);
  }
  if (process.env.AI_SRE_API_KEY) {
    headers.set("X-API-Key", process.env.AI_SRE_API_KEY);
  }

  try {
    const response = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store",
    });
    return new Response(await response.text(), {
      status: response.status,
      headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ detail: `backend unreachable at ${BACKEND_URL}: ${message}` }, { status: 502 });
  }
}

export const GET = relay;
export const POST = relay;
