import Message from "@/components/chat/Message";
import { MessageType } from "@/types/chat";


type MessageListProps = {
  messages: MessageType[];
};

function MessageList({ messages }: MessageListProps) {
  return (
    <div className="flex-1 p-4 overflow-y-auto">
      {messages.map((msg, i) => (
        <Message key={i} role={msg.role} text={msg.text} />
      ))}
    </div>
  );
}

export default MessageList;
