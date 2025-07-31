export interface Note {
  id: string;
  title: string;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface SearchResult {
  id: string;
  title: string;
  content: string;
  similarity: number;
}

export interface ChatResponse {
  response: string;
  relevant_notes: SearchResult[];
}