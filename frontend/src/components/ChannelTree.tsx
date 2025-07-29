import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { channelApi } from '../api';
import { Channel } from '../types';

interface ChannelTreeProps {
  selectedChannelId: string | null;
  onChannelSelect: (channelId: string) => void;
}

interface ChannelItemProps {
  channel: Channel;
  level: number;
  selectedChannelId: string | null;
  onChannelSelect: (channelId: string) => void;
}

const ChannelItem: React.FC<ChannelItemProps> = ({ channel, level, selectedChannelId, onChannelSelect }) => {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="channel-item">
      <div style={{ marginLeft: `${level * 20}px`, display: 'flex', alignItems: 'center' }}>
        {channel.children.length > 0 && (
          <button
            style={{ marginRight: '8px', padding: '2px 6px', fontSize: '12px' }}
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? '▼' : '▶'}
          </button>
        )}
        <button
          className={`channel-button ${selectedChannelId === channel.id ? 'active' : ''}`}
          onClick={() => onChannelSelect(channel.id)}
          style={{ flex: 1 }}
        >
          {channel.name}
        </button>
      </div>
      {expanded && channel.children.map((child) => (
        <ChannelItem
          key={child.id}
          channel={child}
          level={level + 1}
          selectedChannelId={selectedChannelId}
          onChannelSelect={onChannelSelect}
        />
      ))}
    </div>
  );
};

const CreateChannelForm: React.FC<{ parentId?: string; onClose: () => void }> = ({ parentId, onClose }) => {
  const [name, setName] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const queryClient = useQueryClient();

  const createMutation = useMutation({
    mutationFn: (data: { name: string; parent_id?: string; system_prompt?: string }) =>
      channelApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['channels'] });
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      createMutation.mutate({
        name: name.trim(),
        parent_id: parentId,
        system_prompt: systemPrompt.trim() || undefined,
      });
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h2>Create New Channel</h2>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '1rem' }}>
            <label>Name:</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              style={{ width: '100%', marginTop: '0.5rem' }}
              required
            />
          </div>
          <div style={{ marginBottom: '1rem' }}>
            <label>System Prompt (optional):</label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              style={{ width: '100%', marginTop: '0.5rem', minHeight: '100px' }}
            />
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button type="submit" className="primary" disabled={createMutation.isPending}>
              {createMutation.isPending ? 'Creating...' : 'Create'}
            </button>
            <button type="button" onClick={onClose}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export const ChannelTree: React.FC<ChannelTreeProps> = ({ selectedChannelId, onChannelSelect }) => {
  const [showCreateForm, setShowCreateForm] = useState(false);

  const { data: channels = [], isLoading } = useQuery({
    queryKey: ['channels'],
    queryFn: () => channelApi.getTree().then(res => res.data),
  });

  if (isLoading) {
    return <div className="sidebar">Loading channels...</div>;
  }

  return (
    <div className="sidebar">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2>Channels</h2>
        <button onClick={() => setShowCreateForm(true)}>+ New</button>
      </div>
      
      <div className="channel-tree">
        {channels.map((channel) => (
          <ChannelItem
            key={channel.id}
            channel={channel}
            level={0}
            selectedChannelId={selectedChannelId}
            onChannelSelect={onChannelSelect}
          />
        ))}
      </div>

      {showCreateForm && (
        <CreateChannelForm onClose={() => setShowCreateForm(false)} />
      )}
    </div>
  );
}; 