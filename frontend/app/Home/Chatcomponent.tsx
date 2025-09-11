"use client";

import { useEffect, useState, useRef } from "react";

const ChatComponent = () => {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    []
  );
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch files on component mount
  useEffect(() => {
    fetch("http://localhost:8000/api/files")
      .then((res) => res.json())
      .then((data) => {
        setFiles(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching files:", err);
        setIsLoading(false);
      });
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input, filename: selectedFile }),
      });

      if (!res.ok) {
        throw new Error(`Server responded with status ${res.status}`);
      }

      const data = await res.json();
      const botMessage = { role: "bot", content: data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("Error sending message:", err);
      const errorMessage = {
        role: "bot",
        content: "Sorry, I encountered an error. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-20">
      <div className="max-w-4xl mx-auto h-full bg-gray-800 p-4 rounded-lg shadow-md flex flex-col">
        <h1 className="text-2xl font-bold mb-4 text-center">
          AI Chat with Custom Instructions
        </h1>

        {/* Dropdown */}
        <div className="mb-4">
          <label
            htmlFor="instruction-select"
            className="block mb-2 font-medium"
          >
            Select system instruction:
          </label>
          <select
            id="instruction-select"
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            className="w-full p-2 rounded-md text-black"
            disabled={isLoading}
          >
            <option value="">Default behavior</option>
            {files.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
          {isLoading && (
            <p className="text-sm text-gray-400 mt-1">
              Loading instructions...
            </p>
          )}
        </div>

        {/* Chat messages */}
        <div className="space-y-4 flex-1 overflow-y-auto border border-gray-700 p-4 rounded-lg bg-gray-900 mb-4 h-96">
          {messages.length === 0 ? (
            <div className="text-center text-gray-400 py-8">
              Start a conversation by typing a message below
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <span
                  className={`px-4 py-2 rounded-md max-w-xs md:max-w-md ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700 text-white"
                  }`}
                >
                  {msg.content}
                </span>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input box */}
        <div className="flex flex-col sm:flex-row gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Type your message here..."
            className="flex-1 p-3 border border-gray-600 rounded-md text-black"
            disabled={isLoading}
          />
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
          >
            Send
          </button>
        </div>

        {selectedFile && (
          <p className="text-sm text-gray-400 mt-2">
            Using instruction file: <strong>{selectedFile}</strong>
          </p>
        )}
      </div>
    </div>
  );
};

export default ChatComponent;
