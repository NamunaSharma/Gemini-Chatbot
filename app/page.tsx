"use client";

import { useEffect, useState, useRef } from "react";

export default function HomePage() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    []
  );
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch instruction files
  useEffect(() => {
    fetch("http://localhost:8000/api/files")
      .then((res) => res.json())
      .then(setFiles)
      .catch(console.error);
  }, []);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input, filename: selectedFile }),
      });
      const data = await res.json();
      const botMessage = { role: "assistant", content: data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-4">
        <h1 className="text-2xl font-bold text-center mb-4">AI Chat</h1>

        {/* Instruction Selector */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">
            Select AI Personality:
          </label>
          <select
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            className="w-full p-2 border rounded-md"
          >
            <option value="">Default Assistant</option>
            {files.map((file) => (
              <option key={file} value={file}>
                {file.replace(".txt", "")}
              </option>
            ))}
          </select>
        </div>

        {/* Active Instruction Indicator */}
        {selectedFile && (
          <p className="text-sm text-gray-500 mb-2">
            Using instruction:{" "}
            <strong>{selectedFile.replace(".txt", "")}</strong>
          </p>
        )}

        {/* Chat Messages */}
        <div className="h-96 overflow-y-auto border rounded-lg p-4 mb-4 bg-gray-50">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              Start a conversation by typing a message below
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`mb-3 ${
                  msg.role === "user" ? "text-right" : "text-left"
                }`}
              >
                <span
                  className={`inline-block px-4 py-2 rounded-lg ${
                    msg.role === "user"
                      ? "bg-blue-500 text-white"
                      : "bg-gray-200 text-gray-800"
                  }`}
                >
                  {msg.content}
                </span>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Type your message..."
            className="flex-1 p-2 border rounded-md"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="bg-blue-500 text-white px-4 py-2 rounded-md disabled:bg-gray-300"
          >
            Send
          </button>
        </div>

        {isLoading && (
          <div className="text-center text-gray-500 mt-2">
            AI is thinking...
          </div>
        )}
      </div>
    </div>
  );
}
