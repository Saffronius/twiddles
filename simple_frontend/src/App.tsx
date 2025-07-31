import React, { useState, useEffect, useCallback } from 'react';
import { Plus, Search, MessageCircle, Save, Trash2, X } from 'lucide-react';
import { notesApi } from './api';
import { Note, ChatResponse } from './types';

function App() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [selectedNote, setSelectedNote] = useState<Note | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{type: 'user' | 'assistant', content: string, relevant_notes?: any[]}>>([]);
  const [chatInput, setChatInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const loadNotes = async () => {
    try {
      const data = await notesApi.getAll();
      setNotes(data);
    } catch (error) {
      console.error('Failed to load notes:', error);
    }
  };

  useEffect(() => {
    loadNotes();
  }, []);

  useEffect(() => {
    if (selectedNote) {
      setTitle(selectedNote.title);
      setContent(selectedNote.content);
      setHasUnsavedChanges(false);
    }
  }, [selectedNote]);

  const handleTitleChange = (value: string) => {
    setTitle(value);
    setHasUnsavedChanges(true);
  };

  const handleContentChange = (value: string) => {
    setContent(value);
    setHasUnsavedChanges(true);
  };

  const createNewNote = async () => {
    try {
      const newNote = await notesApi.create({
        title: 'Untitled Note',
        content: ''
      });
      setNotes(prev => [newNote, ...prev]);
      setSelectedNote(newNote);
    } catch (error) {
      console.error('Failed to create note:', error);
    }
  };

  const saveNote = async () => {
    if (!selectedNote || !hasUnsavedChanges) return;
    
    try {
      const updatedNote = await notesApi.update(selectedNote.id, {
        title: title || 'Untitled Note',
        content
      });
      
      setNotes(prev => prev.map(note => 
        note.id === selectedNote.id ? updatedNote : note
      ));
      setSelectedNote(updatedNote);
      setHasUnsavedChanges(false);
    } catch (error) {
      console.error('Failed to save note:', error);
    }
  };

  const deleteNote = async () => {
    if (!selectedNote) return;
    
    if (confirm('Are you sure you want to delete this note?')) {
      try {
        await notesApi.delete(selectedNote.id);
        setNotes(prev => prev.filter(note => note.id !== selectedNote.id));
        setSelectedNote(null);
        setTitle('');
        setContent('');
        setHasUnsavedChanges(false);
      } catch (error) {
        console.error('Failed to delete note:', error);
      }
    }
  };

  const filteredNotes = notes.filter(note => {
    // Hide empty notes (untitled with no content)
    const isEmpty = (!note.title || note.title === 'Untitled Note') && !note.content.trim();
    if (isEmpty) return false;
    
    // Apply search filter
    return note.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
           note.content.toLowerCase().includes(searchQuery.toLowerCase());
  });

  const sendChatMessage = async () => {
    if (!chatInput.trim() || isLoading) return;

    const userMessage = chatInput;
    setChatInput('');
    setChatMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response: ChatResponse = await notesApi.chat(userMessage);
      setChatMessages(prev => [...prev, { 
        type: 'assistant', 
        content: response.response,
        relevant_notes: response.relevant_notes 
      }]);
    } catch (error) {
      console.error('Chat error:', error);
      setChatMessages(prev => [...prev, { 
        type: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) {
        return 'Just now';
      }
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (error) {
      return 'Just now';
    }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-title">Notes</div>
          <button className="new-note-btn" onClick={createNewNote}>
            <Plus size={16} />
            New Note
          </button>
        </div>

        <div className="search-bar">
          <Search className="search-icon" size={16} />
          <input
            type="text"
            className="search-input"
            placeholder="Search notes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="rag-toggle">
          <button className="rag-button" onClick={() => setIsChatOpen(true)}>
            <MessageCircle size={16} />
            Chat with Notes
          </button>
        </div>

        <div className="notes-list">
          {filteredNotes.map((note) => (
            <div
              key={note.id}
              className={`note-item ${selectedNote?.id === note.id ? 'active' : ''}`}
              onClick={() => setSelectedNote(note)}
            >
              <div className="note-title">{note.title}</div>
              <div className="note-preview">{note.content}</div>
              <div className="note-date">{formatDate(note.updated_at)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="main-content">
        {selectedNote ? (
          <>
            <div className="editor-header">
              <input
                type="text"
                className="editor-title"
                placeholder="Note title..."
                value={title}
                onChange={(e) => handleTitleChange(e.target.value)}
              />
              <div className="editor-meta">
                <span>Last modified: {formatDate(selectedNote.updated_at)}</span>
                <div className="editor-actions">
                  {hasUnsavedChanges && (
                    <button className="action-btn" onClick={saveNote}>
                      <Save size={14} />
                      Save
                    </button>
                  )}
                  <button className="action-btn danger" onClick={deleteNote}>
                    <Trash2 size={14} />
                    Delete
                  </button>
                </div>
              </div>
            </div>
            <div className="editor-content">
              <textarea
                className="editor-textarea"
                placeholder="Start writing your note..."
                value={content}
                onChange={(e) => handleContentChange(e.target.value)}
              />
            </div>
          </>
        ) : (
          <div className="empty-state">
            <h2>Select a note to start writing</h2>
            <p>Choose a note from the sidebar or create a new one to begin.</p>
            <button className="new-note-btn" onClick={createNewNote}>
              <Plus size={16} />
              Create Your First Note
            </button>
          </div>
        )}
      </div>

      {isChatOpen && (
        <div className="chat-modal">
          <div className="chat-content">
            <div className="chat-header">
              <h3 className="chat-title">Chat with Your Notes</h3>
              <button className="close-btn" onClick={() => setIsChatOpen(false)}>
                <X size={18} />
              </button>
            </div>
            
            <div className="chat-messages">
              {chatMessages.map((message, index) => (
                <div key={index} className={`message ${message.type}`}>
                  <div>{message.content}</div>
                  {message.relevant_notes && message.relevant_notes.length > 0 && (
                    <div className="relevant-notes">
                      <div>Referenced notes:</div>
                      {message.relevant_notes.map((note, noteIndex) => (
                        <div key={noteIndex} className="relevant-note">
                          <strong>{note.title}</strong> (similarity: {(note.similarity * 100).toFixed(0)}%)
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="message assistant">
                  <div>Thinking...</div>
                </div>
              )}
            </div>

            <div className="chat-input-container">
              <input
                type="text"
                className="chat-input"
                placeholder="Ask me anything about your notes..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
              />
              <button 
                className="send-btn" 
                onClick={sendChatMessage}
                disabled={isLoading || !chatInput.trim()}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;