"use client";

import { useEffect, useState } from "react";
import { FileItem, Message } from "./helpers/types";
import { fetchFiles, sendMessage } from "./helpers/api";
import ChatBox from "./components/ChatBox";
import InputBar from "./components/InputBar";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [selectedFile, setSelectedFile] = useState("");
  const [retrievalMethod, setRetrievalMethod] = useState("dense");

  useEffect(() => {
    fetchFiles()
      .then((data) => {
        setFiles(data);
        if (data.length > 0) setSelectedFile(data[0].name);
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const botMessage = await sendMessage(
        input,
        selectedFile,
        retrievalMethod
      );
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) return <div>Loading files...</div>;
  if (error) return <div>Error: {String(error)}</div>;

  return (
    <div className="min-h-screen w-auto p-10 bg-white">
      <h1 className="text-gray-500 text-2xl mb-2">Chatbot</h1>

      <div className="flex gap-4 mb-4">
        <select
          value={selectedFile}
          onChange={(e) => {
            setSelectedFile(e.target.value);
            setMessages([]);
          }}
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
          // onChange={(e) => {
          //   setRetrievalMethod(e.target.value);
          //   window.location.reload(); // full page refresh
          // }}
          onChange={(e) => setRetrievalMethod(e.target.value)}
          className="p-2 border rounded bg-amber-700"
        >
          <option value="dense">Dense</option>
          <option value="sparse">Sparse</option>
          <option value="hybrid">Hybrid</option>
          <option value="expanded">Expanded</option>
        </select>
      </div>

      <ChatBox messages={messages} />
      <InputBar input={input} setInput={setInput} onSend={handleSend} />
    </div>
  );
}
