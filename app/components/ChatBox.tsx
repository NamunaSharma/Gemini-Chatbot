"use client";
import React from "react";
import { Message } from "../helpers/types";

interface Props {
  messages: Message[];
}

export default function ChatBox({ messages }: Props) {
  return (
    <div className="w-1/2 h-96 outline rounded-sm p-6 bg-gray-600 overflow-y-scroll text-white">
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
  );
}
