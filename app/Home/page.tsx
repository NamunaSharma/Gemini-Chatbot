"use client";
import { useState } from "react";
import React from "react";
export default function HomePage() {
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
    <div className="min-h-screen bg-gray-900 text-white p-20 ">
      <div className="w-96 h-full bg-gray-600 p-4 rounded-lg shadow-md overflow-y-auto">
        <div className="space-y-2 max-h-[60vh] overflow-y-auto border p-4 rounded-lg bg-gray-800">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`${msg.role === "user" ? "text-right" : "text-left"}`}
            >
              {/* <span className="block px-4 py-2 rounded-md bg-gray-700 inline-block max-w-xs">
                {msg.content}
              </span> */}
            </div>
          ))}
        </div>
        <div className="w-full flex justify-start mb-2"></div>
        <div className="w-full">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Type your message here..."
            className="w-full p-2 border rounded-md"
          />
          <button
            className="bg-blue-500 text-white px-4 py-2 rounded-md mt-2"
            onClick={sendMessage}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
