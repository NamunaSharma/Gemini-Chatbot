import dynamic from "next/dynamic";
const ChatComponent = dynamic(() => import("./Chatcomponent"), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-20 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
        <p>Loading chat interface...</p>
      </div>
    </div>
  ),
});

export default function HomePage() {
  return <ChatComponent />;
}
