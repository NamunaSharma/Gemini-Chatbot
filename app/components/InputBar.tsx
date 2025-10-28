"use client";
import React from "react";

interface Props {
  input: string;
  setInput: (val: string) => void;
  onSend: () => void;
}

export default function InputBar({ input, setInput, onSend }: Props) {
  return (
    <div className="flex mt-4 space-x-4">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSend()}
        placeholder="Enter your prompt"
        className="w-[40%] text-black p-3 border rounded"
      />
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-10 rounded"
        onClick={onSend}
      >
        Send
      </button>
    </div>
  );
}
