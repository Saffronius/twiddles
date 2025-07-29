import React, { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import EditorJS from '@editorjs/editorjs';
import Header from '@editorjs/header';
import List from '@editorjs/list';
import Paragraph from '@editorjs/paragraph';
import { scratchpadApi, routingApi, channelApi } from '../api';
import { RoutingResponse, Channel } from '../types';

interface SpitballModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface RoutingDialogProps {
  entryId: string;
  suggestion: RoutingResponse;
  channels: Channel[];
  onClose: () => void;
}

const RoutingDialog: React.FC<RoutingDialogProps> = ({ entryId, suggestion, channels, onClose }) => {
  const [newChannelName, setNewChannelName] = useState('');
  const [seedDescription, setSeedDescription] = useState('');
  const queryClient = useQueryClient();

  const acceptMutation = useMutation({
    mutationFn: (channelId: string) => routingApi.accept(entryId, channelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scratchpad'] });
      queryClient.invalidateQueries({ queryKey: ['channels'] });
      onClose();
    },
  });

  const createMutation = useMutation({
    mutationFn: (data: { name: string; seedDescription?: string }) =>
      routingApi.createChannel(entryId, data.name, data.seedDescription),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scratchpad'] });
      queryClient.invalidateQueries({ queryKey: ['channels'] });
      onClose();
    },
  });

  const getBestChannel = () => {
    if (suggestion.best_channel_id) {
      const findChannel = (channels: Channel[]): Channel | null => {
        for (const channel of channels) {
          if (channel.id === suggestion.best_channel_id) return channel;
          const found = findChannel(channel.children);
          if (found) return found;
        }
        return null;
      };
      return findChannel(channels);
    }
    return null;
  };

  const bestChannel = getBestChannel();

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h2>Route Scratchpad</h2>
        
        {suggestion.suggest_new ? (
          <div>
            <p>No good channel match found. Create a new channel?</p>
            <div style={{ marginBottom: '1rem' }}>
              <label>Channel Name:</label>
              <input
                type="text"
                value={newChannelName}
                onChange={(e) => setNewChannelName(e.target.value)}
                style={{ width: '100%', marginTop: '0.5rem' }}
                placeholder="Enter channel name"
              />
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <label>Seed Description (optional):</label>
              <textarea
                value={seedDescription}
                onChange={(e) => setSeedDescription(e.target.value)}
                style={{ width: '100%', marginTop: '0.5rem', minHeight: '80px' }}
                placeholder="Brief description of this channel's purpose"
              />
            </div>
          </div>
        ) : (
          <div>
            <p>
              Best match: <strong>{bestChannel?.name}</strong> 
              (confidence: {(suggestion.confidence! * 100).toFixed(1)}%)
            </p>
            <p style={{ fontSize: '0.9em', color: '#666' }}>
              {bestChannel?.description}
            </p>
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
          {suggestion.suggest_new ? (
            <button
              className="primary"
              onClick={() => createMutation.mutate({ name: newChannelName, seedDescription })}
              disabled={!newChannelName.trim() || createMutation.isPending}
            >
              {createMutation.isPending ? 'Creating...' : 'Create Channel'}
            </button>
          ) : (
            <button
              className="primary"
              onClick={() => acceptMutation.mutate(suggestion.best_channel_id!)}
              disabled={acceptMutation.isPending}
            >
              {acceptMutation.isPending ? 'Routing...' : 'Accept'}
            </button>
          )}
          <button onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
};

export const SpitballModal: React.FC<SpitballModalProps> = ({ isOpen, onClose }) => {
  const editorRef = useRef<EditorJS | null>(null);
  const editorContainerRef = useRef<HTMLDivElement>(null);
  const [routingSuggestion, setRoutingSuggestion] = useState<{ entryId: string; suggestion: RoutingResponse } | null>(null);

  const { data: channels = [] } = useQuery({
    queryKey: ['channels'],
    queryFn: () => channelApi.getTree().then(res => res.data),
  });

  const createScratchpadMutation = useMutation({
    mutationFn: (content_json: any) => scratchpadApi.create(content_json),
  });

  const routeMutation = useMutation({
    mutationFn: (entryId: string) => scratchpadApi.route(entryId),
    onSuccess: (data, entryId) => {
      setRoutingSuggestion({ entryId, suggestion: data.data });
    },
  });

  useEffect(() => {
    if (isOpen && editorContainerRef.current && !editorRef.current) {
      editorRef.current = new EditorJS({
        holder: editorContainerRef.current,
        tools: {
          header: Header,
          list: List,
          paragraph: Paragraph,
        },
        placeholder: 'Start typing your thoughts...',
      });
    }

    return () => {
      if (editorRef.current) {
        editorRef.current.destroy();
        editorRef.current = null;
      }
    };
  }, [isOpen]);

  const handleSubmit = async () => {
    if (!editorRef.current) return;

    try {
      const outputData = await editorRef.current.save();
      
      // Create scratchpad entry
      const scratchpadResponse = await createScratchpadMutation.mutateAsync(outputData);
      
      // Get routing suggestion
      routeMutation.mutate(scratchpadResponse.data.id);
    } catch (error) {
      console.error('Failed to save or route:', error);
    }
  };

  const handleClose = () => {
    setRoutingSuggestion(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <>
      <div className="modal-overlay">
        <div className="modal">
          <h2>Spitball Ideas</h2>
          <div ref={editorContainerRef} className="editor-container" />
          
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
            <button
              className="primary"
              onClick={handleSubmit}
              disabled={createScratchpadMutation.isPending || routeMutation.isPending}
            >
              {createScratchpadMutation.isPending || routeMutation.isPending ? 'Processing...' : 'Route to Channel'}
            </button>
            <button onClick={handleClose}>Cancel</button>
          </div>
        </div>
      </div>

      {routingSuggestion && (
        <RoutingDialog
          entryId={routingSuggestion.entryId}
          suggestion={routingSuggestion.suggestion}
          channels={channels}
          onClose={() => {
            setRoutingSuggestion(null);
            onClose();
          }}
        />
      )}
    </>
  );
}; 