export interface FileItem {
  id: string;
  name: string;
  description: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  retrievalTime?: number;
  expandedQuery?: string;
}
