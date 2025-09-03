"use client";
import { useEffect, useState } from "react";

// ✅ Interfaces should be declared outside the component
interface FileItem {
  id: string;
  name: string;
  description: string;
}

export default function Home() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    []
  );
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [selectedFiles, setSelectedFiles] = useState("IT_professional.txt");

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/prompts/");
        if (!res.ok) throw new Error("Failed to fetch data");
        const data = await res.json();
        setFiles(data);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const handleChange = (event: any) => {
    setSelectedFiles(event.target.value);
    setMessages([]);
  };

  const sendMessages = async () => {
    if (!input.trim()) return;
    const usermessage = { role: "user", content: input };
    setMessages((prev) => [...prev, usermessage]);
    setInput(""); // ✅ Clear input properly

    const response = await fetch("http://127.0.0.1:8000/api/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: input, filename: selectedFiles }),
    });

    const data = await response.json();
    const botmessage = { role: "assistant", content: data.message }; // ✅ lowercase
    setMessages((prev) => [...prev, botmessage]);
  };

  const handleKeyDown = (event: any) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendMessages();
    }
  };

  return (
    <div className="min-h-screen w-auto p-10 bg-white">
      <h1 className="text-gray-500 text-2xl mb-2">Chatbot</h1>
      <div className="w-150 h-96 outline-solid rounded-sm p-6 bg-gray-600 overflow-y-scroll text-white">
        <select value={selectedFiles} onChange={handleChange}>
          {files.map((file) => (
            <option key={file.id} value={file.name}>
              {file.name}
            </option>
          ))}
        </select>

        {messages.map((msg, id) => (
          <div
            key={id}
            className={
              msg.role === "user"
                ? "text-right mb-2 mt-8 leading-relaxed"
                : "text-left mt-8 leading-relaxed w-[80%]"
            }
          >
            <span className="text-white text-md rounded-sm p-3 leading-8 w-[50%]">
              {msg.content}
            </span>
          </div>
        ))}
      </div>

      <div className="flex mt-4 space-x-8">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="enter the prompt"
          className="w-115 h-auto text-black p-3 outline outline-gray-500 rounded-sm overflow-y-scroll"
        />
        <button
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-5 rounded"
          onClick={sendMessages}
        >
          Click Me
        </button>
      </div>
    </div>
  );
}
