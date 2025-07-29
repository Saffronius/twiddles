import axios from 'axios';
import { Channel, Message, ScratchpadEntry, RoutingResponse, RoutingResult, KnowledgeCard } from './types';

const api = axios.create({
  baseURL: '/api',
});

export const channelApi = {
  getTree: () => api.get<Channel[]>('/channels/tree'),
  create: (data: { name: string; parent_id?: string; system_prompt?: string }) =>
    api.post<Channel>('/channels', data),
  update: (id: string, data: { name?: string; system_prompt?: string; inherit_prompt?: boolean }) =>
    api.patch<Channel>(`/channels/${id}`, data),
  summarizeOld: (id: string, retain: number = 50) =>
    api.post(`/channels/${id}/summarize_old?retain=${retain}`),
  getMessages: (id: string, after?: string) =>
    api.get<Message[]>(`/channels/${id}/messages${after ? `?after=${after}` : ''}`),
  sendMessage: (id: string, content: string, rag: boolean = false) =>
    api.post<{ user_message_id: string; assistant_message_id: string; content: string }>(`/channels/${id}/messages`, { content, rag }),
};

export const scratchpadApi = {
  create: (content_json: any) =>
    api.post<ScratchpadEntry>('/scratchpad', { content_json }),
  getStaged: () =>
    api.get<ScratchpadEntry[]>('/scratchpad/staged'),
  route: (id: string) =>
    api.post<RoutingResponse>(`/scratchpad/${id}/route`),
};

export const routingApi = {
  accept: (entryId: string, channelId: string) =>
    api.post<RoutingResult>(`/routing/${entryId}/accept`, { channel_id: channelId }),
  createChannel: (entryId: string, name: string, seedDescription?: string) =>
    api.post<RoutingResult>(`/routing/${entryId}/create_channel`, { name, seed_description: seedDescription }),
  undo: (entryId: string) =>
    api.post(`/routing/${entryId}/undo`),
};

export const knowledgeCardApi = {
  create: (data: { channel_id: string; title: string; body: string; source_message_ids: string[] }) =>
    api.post<KnowledgeCard>('/knowledge_cards', data),
  getByChannel: (channelId?: string) =>
    api.get<KnowledgeCard[]>(`/knowledge_cards${channelId ? `?channel_id=${channelId}` : ''}`),
}; 