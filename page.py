"use client";

import { useEffect, useState } from "react";

interface FileItem {
  id: string;
  name: string;
  description: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  retrievalTime?: number;
  expandedQuery?: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [selectedFile, setSelectedFile] = useState("");
  const [retrievalMethod, setRetrievalMethod] = useState("dense");

  // Fetch available files from backend
  useEffect(() => {
    async function fetchFiles() {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/prompts/");
        if (!res.ok) throw new Error("Failed to fetch files");
        const data = await res.json();
        setFiles(data);
        if (data.length > 0) setSelectedFile(data[0].name);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    }
    fetchFiles();
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedFile(event.target.value);
    setMessages([]);
  };

  const handleRetrievalChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setRetrievalMethod(event.target.value);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const response = await fetch("http://127.0.0.1:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: input,
          filename: selectedFile,
          retrieval_method: retrievalMethod,
        }),
      });
      const data = await response.json();

      const botMessage: Message = {
        role: "assistant",
        content: data.response,
        sources: [selectedFile],
        retrievalTime: data.retrieval_time_seconds,
        expandedQuery: data.expanded_query,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("Error sending message:", err);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendMessage();
    }
  };

  if (loading) return <div>Loading files...</div>;
  if (error) return <div>Error: {String(error)}</div>;

  return (
    <div className="min-h-screen w-auto p-10 bg-white">
      <h1 className="text-gray-500 text-2xl mb-2">Chatbot</h1>

      {/* file and retrieval selectors */}
      <div className="flex gap-4 mb-4">
        <select
          value={selectedFile}
          onChange={handleFileChange}
          className="p-2 border rounded bg-amber-700"
        >
          {files.map((file) => (
            <option key={file.id} value={file.name}>
              {file.name}
            </option>
          ))}
        </select>

        <select
          value={retrievalMethod}
          onChange={handleRetrievalChange}
          className="p-2 border rounded bg-amber-700"
        >
          <option value="dense">Dense</option>
          <option value="sparse">Sparse</option>
          <option value="hybrid">Hybrid</option>
          <option value="expanded">Expanded</option>
        </select>
      </div>

      {/* chat box */}
      <div className="w-150 h-96 outline rounded-sm p-6 bg-gray-600 overflow-y-scroll text-white">
        {messages.map((msg, id) => (
          <div
            key={id}
            className={
              msg.role === "user"
                ? "text-right mb-2 mt-4"
                : "text-left mt-4 w-[80%]"
            }
          >
            <span className="text-white text-md rounded-sm p-3 leading-8 inline-block bg-gray-800">
              {msg.content}
            </span>
            {msg.role === "assistant" && (
              <div className="text-xs text-gray-300 mt-1">
                {msg.sources && (
                  <>
                    Source: {msg.sources.join(", ")}
                    <br />
                  </>
                )}
                {msg.retrievalTime && (
                  <>
                    Retrieval Time: {msg.retrievalTime.toFixed(3)} s<br />
                  </>
                )}
                {msg.expandedQuery && <>Expanded Query: {msg.expandedQuery}</>}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* input bar */}
      <div className="flex mt-4 space-x-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter your prompt"
          className="w-120 text-black p-3 border rounded"
        />
        <button
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-5 rounded"
          onClick={sendMessage}
        >
          Send
        </button>
      </div>
    </div>
  );
}