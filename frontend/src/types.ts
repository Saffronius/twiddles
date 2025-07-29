export interface Channel {
  id: string;
  parent_id: string | null;
  name: string;
  description: string;
  system_prompt: string;
  inherit_prompt: boolean;
  created_at: string;
  children: Channel[];
}

export interface Message {
  id: string;
  channel_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  memory_layer: 'recent' | 'compressed';
}

export interface ScratchpadEntry {
  id: string;
  content_json: any;
  content_text: string;
  state: 'staged' | 'routed';
  created_at: string;
}

export interface RoutingResponse {
  best_channel_id?: string;
  confidence?: number;
  suggest_new: boolean;
}

export interface RoutingResult {
  message_id: string;
  channel_id: string;
  description_updated: boolean;
}

export interface KnowledgeCard {
  id: string;
  channel_id: string;
  title: string;
  body: string;
  source_message_ids: string[];
  created_at: string;
} 