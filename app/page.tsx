"use client";

import { useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    []
  );
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);

    setInput("");

    const res = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt: input }),
    });

    const data = await res.json();
    const botMessage = { role: "bot", content: data.response };

    setMessages((prev) => [...prev, botMessage]);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-2xl mx-auto space-y-4">
        <h1 className="text-3xl font-bold mb-4">Gemini Chatbot</h1>
        <div className="space-y-2 max-h-[60vh] overflow-y-auto border p-4 rounded-lg bg-gray-800">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`${msg.role === "user" ? "text-right" : "text-left"}`}
            >
              <span className="block px-4 py-2 rounded-md bg-gray-700 inline-block max-w-xs">
                {msg.content}
              </span>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask something..."
            className="flex-grow px-4 py-2 rounded-md text-black"
          />
          {/* <button
            onClick={sendMessage}
            className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-md"
          >
            Send
          </button> */}
        </div>
      </div>
    </div>
  );
}
