import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Renders the agent's LLM output (tables, lists, code blocks). react-markdown does not render raw HTML
 * and sanitizes link URLs, so model output cannot inject markup into the page.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}
