import axios from 'axios';
import { Note, SearchResult, ChatResponse } from './types';

const api = axios.create({
  baseURL: '/api',
});

export const notesApi = {
  getAll: (): Promise<Note[]> => 
    api.get<Note[]>('/notes').then(res => res.data),
  
  getById: (id: string): Promise<Note> => 
    api.get<Note>(`/notes/${id}`).then(res => res.data),
  
  create: (note: { title: string; content: string }): Promise<Note> => 
    api.post<Note>('/notes', note).then(res => res.data),
  
  update: (id: string, note: { title: string; content: string }): Promise<Note> => 
    api.put<Note>(`/notes/${id}`, note).then(res => res.data),
  
  delete: (id: string): Promise<void> => 
    api.delete(`/notes/${id}`).then(() => {}),
  
  search: (query: string, limit = 5): Promise<SearchResult[]> => 
    api.post<SearchResult[]>('/search', { query, limit }).then(res => res.data),
  
  chat: (query: string): Promise<ChatResponse> => 
    api.post<ChatResponse>('/chat', { query }).then(res => res.data),
};