// // // // "use client";

// // // // import { useEffect, useState } from "react";

// // // // interface FileItem {
// // // //   id: string;
// // // //   name: string;
// // // //   description: string;
// // // // }

// // // // interface Message {
// // // //   role: "user" | "assistant";
// // // //   content: string;
// // // //   sources?: string[]; // optional: store source files
// // // // }

// // // // export default function Home() {
// // // //   const [messages, setMessages] = useState<Message[]>([]);
// // // //   const [input, setInput] = useState("");
// // // //   const [files, setFiles] = useState<FileItem[]>([]);
// // // //   const [loading, setLoading] = useState(true);
// // // //   const [error, setError] = useState<unknown>(null);
// // // //   const [selectedFiles, setSelectedFiles] = useState("IT_professional.txt");

// // // //   useEffect(() => {
// // // //     async function fetchData() {
// // // //       try {
// // // //         const res = await fetch("http://127.0.0.1:8000/api/prompts/");
// // // //         if (!res.ok) throw new Error("Failed to fetch data");
// // // //         const data = await res.json();
// // // //         setFiles(data);
// // // //       } catch (err) {
// // // //         setError(err);
// // // //       } finally {
// // // //         setLoading(false);
// // // //       }
// // // //     }
// // // //     fetchData();
// // // //   }, []);

// // // //   const handleChange = (event: any) => {
// // // //     setSelectedFiles(event.target.value);
// // // //     setMessages([]); // clear conversation when file changes
// // // //   };

// // // //   const sendMessages = async () => {
// // // //     if (!input.trim()) return;
// // // //     const userMessage: Message = { role: "user", content: input };
// // // //     setMessages((prev) => [...prev, userMessage]);
// // // //     setInput("");

// // // //     try {
// // // //       const response = await fetch("http://127.0.0.1:8000/api/chat/", {
// // // //         method: "POST",
// // // //         headers: { "Content-Type": "application/json" },
// // // //         body: JSON.stringify({ prompt: input, filename: selectedFiles }),
// // // //       });
// // // //       const data = await response.json();
// // // //       const botMessage: Message = {
// // // //         role: "assistant",
// // // //         content: data.message,
// // // //         sources: [selectedFiles], // you can expand this later for multiple sources
// // // //       };
// // // //       setMessages((prev) => [...prev, botMessage]);
// // // //     } catch (err) {
// // // //       console.error("Error sending message:", err);
// // // //     }
// // // //   };

// // // //   const handleKeyDown = (event: any) => {
// // // //     if (event.key === "Enter") {
// // // //       event.preventDefault();
// // // //       sendMessages();
// // // //     }
// // // //   };

// // // //   return (
// // // //     <div className="min-h-screen w-auto p-10 bg-white">
// // // //       <h1 className="text-gray-500 text-2xl mb-2">Chatbot</h1>

// // // //       <div className="w-150 h-96 outline-solid rounded-sm p-6 bg-gray-600 overflow-y-scroll text-white">
// // // //         <select value={selectedFiles} onChange={handleChange}>
// // // //           {files.map((file) => (
// // // //             <option key={file.id} value={file.name}>
// // // //               {file.name}
// // // //             </option>
// // // //           ))}
// // // //         </select>

// // // //         {messages.map((msg, id) => (
// // // //           <div
// // // //             key={id}
// // // //             className={
// // // //               msg.role === "user"
// // // //                 ? "text-right mb-2 mt-8 leading-relaxed"
// // // //                 : "text-left mt-8 leading-relaxed w-[80%]"
// // // //             }
// // // //           >
// // // //             <span className="text-white text-md rounded-sm p-3 leading-8 w-[50%]">
// // // //               {msg.content}
// // // //             </span>
// // // //             {msg.role === "assistant" && msg.sources && (
// // // //               <div className="text-xs text-gray-300 mt-1">
// // // //                 Source: {msg.sources.join(", ")}
// // // //               </div>
// // // //             )}
// // // //           </div>
// // // //         ))}
// // // //       </div>

// // // //       <div className="flex mt-4 space-x-8">
// // // //         <input
// // // //           type="text"
// // // //           value={input}
// // // //           onChange={(e) => setInput(e.target.value)}
// // // //           onKeyDown={handleKeyDown}
// // // //           placeholder="Enter your prompt"
// // // //           className="w-115 h-auto text-black p-3 outline outline-gray-500 rounded-sm overflow-y-scroll"
// // // //         />
// // // //         <button
// // // //           className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-5 rounded"
// // // //           onClick={sendMessages}
// // // //         >
// // // //           Send
// // // //         </button>
// // // //       </div>
// // // //     </div>
// // // //   );
// // // // }
// "use client";

// import { useEffect, useState, useRef } from "react";

// const ChatComponent = () => {
//   const [messages, setMessages] = useState<{ role: string; content: string }[]>(
//     []
//   );
//   const [input, setInput] = useState("");
//   const [files, setFiles] = useState<string[]>([]);
//   const [selectedFile, setSelectedFile] = useState<string>("");
//   const [isLoading, setIsLoading] = useState(true);
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   // Fetch files on component mount
//   //   useEffect(() => {
//   //     fetch("http://localhost:8000/api/files")
//   //       .then((res) => res.json())
//   //       .then((data) => {
//   //         setFiles(data);
//   //         setIsLoading(false);
//   //       })
//   //       .catch((err) => {
//   //         console.error("Error fetching files:", err);
//   //         setIsLoading(false);
//   //       });
//   //   }, []);

//   // Fetch files on component mount
//   useEffect(() => {
//     fetch("http://localhost:8000/api/prompts/")
//       .then((res) => {
//         if (!res.ok) throw new Error("Failed to fetch");
//         return res.json();
//       })
//       .then((data) => {
//         // data is array of objects with {id, name, description}
//         setFiles(data.map((f: any) => f.name));
//         setIsLoading(false);
//       })
//       .catch((err) => {
//         console.error("Error fetching files:", err);
//         setIsLoading(false);
//       });
//   }, []);

//   // Scroll to bottom on new messages
//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   }, [messages]);

//   const sendMessage = async () => {
//     if (!input.trim()) return;
//     const userMessage = { role: "user", content: input };
//     setMessages((prev) => [...prev, userMessage]);
//     setInput("");

//     try {
//       const res = await fetch("http://localhost:8000/api/chat", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ prompt: input, filename: selectedFile }),
//       });

//       if (!res.ok) {
//         throw new Error(`Server responded with status ${res.status}`);
//       }

//       const data = await res.json();
//       const botMessage = { role: "bot", content: data.response };
//       setMessages((prev) => [...prev, botMessage]);
//     } catch (err) {
//       console.error("Error sending message:", err);
//       const errorMessage = {
//         role: "bot",
//         content: "Sorry, I encountered an error. Please try again.",
//       };
//       setMessages((prev) => [...prev, errorMessage]);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gray-900 text-white p-4 md:p-20">
//       <div className="max-w-4xl mx-auto h-full bg-gray-800 p-4 rounded-lg shadow-md flex flex-col">
//         <h1 className="text-2xl font-bold mb-4 text-center">
//           AI Chat with Custom Instructions
//         </h1>

//         {/* Dropdown */}
//         <div className="mb-4">
//           <label
//             htmlFor="instruction-select"
//             className="block mb-2 font-medium"
//           >
//             Select system instruction:
//           </label>
//           <select
//             id="instruction-select"
//             value={selectedFile}
//             onChange={(e) => setSelectedFile(e.target.value)}
//             className="w-full p-2 rounded-md text-black"
//             disabled={isLoading}
//           >
//             <option value="">Default behavior</option>
//             {files.map((f) => (
//               <option key={f} value={f}>
//                 {f}
//               </option>
//             ))}
//           </select>
//           {isLoading && (
//             <p className="text-sm text-gray-400 mt-1">
//               Loading instructions...
//             </p>
//           )}
//         </div>

//         {/* Chat messages */}
//         <div className="space-y-4 flex-1 overflow-y-auto border border-gray-700 p-4 rounded-lg bg-gray-900 mb-4 h-96">
//           {messages.length === 0 ? (
//             <div className="text-center text-gray-400 py-8">
//               Start a conversation by typing a message below
//             </div>
//           ) : (
//             messages.map((msg, idx) => (
//               <div
//                 key={idx}
//                 className={`flex ${
//                   msg.role === "user" ? "justify-end" : "justify-start"
//                 }`}
//               >
//                 <span
//                   className={`px-4 py-2 rounded-md max-w-xs md:max-w-md ${
//                     msg.role === "user"
//                       ? "bg-blue-600 text-white"
//                       : "bg-gray-700 text-white"
//                   }`}
//                 >
//                   {msg.content}
//                 </span>
//               </div>
//             ))
//           )}
//           <div ref={messagesEndRef} />
//         </div>

//         {/* Input box */}
//         <div className="flex flex-col sm:flex-row gap-2">
//           <input
//             type="text"
//             value={input}
//             onChange={(e) => setInput(e.target.value)}
//             onKeyDown={(e) => e.key === "Enter" && sendMessage()}
//             placeholder="Type your message here..."
//             className="flex-1 p-3 border border-gray-600 rounded-md text-black"
//             disabled={isLoading}
//           />
//           <button
//             className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
//             onClick={sendMessage}
//             disabled={isLoading || !input.trim()}
//           >
//             Send
//           </button>
//         </div>

//         {selectedFile && (
//           <p className="text-sm text-gray-400 mt-2">
//             Using instruction file: <strong>{selectedFile}</strong>
//           </p>
//         )}
//       </div>
//     </div>
//   );
// };

// export default ChatComponent;
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
  // Fetch files on component mount
  useEffect(() => {
    fetch("http://localhost:8000/api/prompts/")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch");
        return res.json();
      })
      .then((data) => {
        // data is array of objects with {id, name, description}
        setFiles(data.map((f: any) => f.name));
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
