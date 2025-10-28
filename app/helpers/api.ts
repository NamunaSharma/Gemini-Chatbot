import { FileItem, Message } from "./types";

export async function fetchFiles(): Promise<FileItem[]> {
  const res = await fetch("http://127.0.0.1:8000/api/prompts/");
  if (!res.ok) throw new Error("Failed to fetch files");
  return res.json();
}

export async function sendMessage(
  prompt: string,
  filename: string,
  retrievalMethod: string
): Promise<Message> {
  const res = await fetch("http://127.0.0.1:8000/api/chat/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      filename,
      retrieval_method: retrievalMethod,
    }),
  });
  const data = await res.json();
  return {
    role: "assistant",
    content: data.response,
    sources: [filename],
    retrievalTime: data.retrieval_time_seconds,
    expandedQuery: data.expanded_query,
  };
}
