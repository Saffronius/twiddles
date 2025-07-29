import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { channelApi } from '../api';
import { Message } from '../types';

interface ChannelViewProps {
  channelId: string;
}

const MessageComponent: React.FC<{ message: Message }> = ({ message }) => {
  return (
    <div className={`message ${message.role}`}>
      <div className="message-meta">
        {message.role} • {new Date(message.created_at).toLocaleString()} • {message.memory_layer}
      </div>
      <div>{message.content}</div>
    </div>
  );
};

const MessageComposer: React.FC<{ channelId: string }> = ({ channelId }) => {
  const [content, setContent] = useState('');
  const [useRag, setUseRag] = useState(false);
  const queryClient = useQueryClient();

  const sendMutation = useMutation({
    mutationFn: (data: { content: string; rag: boolean }) =>
      channelApi.sendMessage(channelId, data.content, data.rag),
    onSuccess: () => {
      setContent('');
      queryClient.invalidateQueries({ queryKey: ['messages', channelId] });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (content.trim()) {
      sendMutation.mutate({ content: content.trim(), rag: useRag });
    }
  };

  return (
    <div className="composer">
      <form onSubmit={handleSubmit}>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Type your message..."
          disabled={sendMutation.isPending}
        />
        <div className="composer-controls">
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={useRag}
              onChange={(e) => setUseRag(e.target.checked)}
            />
            Use RAG (include summaries)
          </label>
          <button type="submit" className="primary" disabled={sendMutation.isPending}>
            {sendMutation.isPending ? 'Sending...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
};

export const ChannelView: React.FC<ChannelViewProps> = ({ channelId }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: messages = [], isLoading } = useQuery({
    queryKey: ['messages', channelId],
    queryFn: () => channelApi.getMessages(channelId).then(res => res.data),
    enabled: !!channelId,
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!channelId) {
    return (
      <div className="main-content">
        <div className="content-area" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p>Select a channel to start chatting</p>
        </div>
      </div>
    );
  }

  return (
    <div className="main-content">
      <div className="content-area">
        {isLoading ? (
          <div>Loading messages...</div>
        ) : (
          <div className="messages-container">
            {messages.length === 0 ? (
              <p>No messages yet. Start the conversation!</p>
            ) : (
              messages.map((message) => (
                <MessageComponent key={message.id} message={message} />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      <MessageComposer channelId={channelId} />
    </div>
  );
}; 