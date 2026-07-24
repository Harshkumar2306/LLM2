import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User, Link as LinkIcon, ChevronDown, Database, Globe, Cpu, Zap } from 'lucide-react';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    { id: 1, role: 'ai', text: 'Hello! I am Axiom. How can I help you today?', sources: [] }
  ]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState('hybrid');
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesEndRef = useRef(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const modes = [
    { id: 'none', name: 'Local Only', icon: Cpu, desc: 'Internal knowledge' },
    { id: 'local', name: 'Database (FAISS)', icon: Database, desc: 'Local vector DB' },
    { id: 'web', name: 'Live Web Search', icon: Globe, desc: 'Searches the internet live' },
    { id: 'hybrid', name: 'Hybrid Brain', icon: Zap, desc: 'Local + Web Search' },
  ];
  
  const currentMode = modes.find(m => m.id === mode);
  const CurrentIcon = currentMode?.icon || Cpu;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;

    const userText = input.trim();
    setInput('');
    setIsGenerating(true);

    const userMsg = { id: Date.now(), role: 'user', text: userText, sources: [] };
    setMessages(prev => [...prev, userMsg]);

    const aiMsgId = Date.now() + 1;
    setMessages(prev => [...prev, { id: aiMsgId, role: 'ai', text: '', sources: [] }]);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText, mode: mode })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.substring(6);
              try {
                const data = JSON.parse(dataStr);
                
                if (data.type === 'sources') {
                  setMessages(prev => prev.map(msg => 
                    msg.id === aiMsgId ? { ...msg, sources: data.sources } : msg
                  ));
                } else if (data.type === 'chunk') {
                  setMessages(prev => prev.map(msg => 
                    msg.id === aiMsgId ? { ...msg, text: msg.text + data.text } : msg
                  ));
                } else if (data.type === 'done') {
                  setIsGenerating(false);
                }
              } catch (err) {
                console.error("Error parsing SSE data:", err);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages(prev => prev.map(msg => 
        msg.id === aiMsgId ? { ...msg, text: 'Sorry, I encountered an error connecting to the server.' } : msg
      ));
      setIsGenerating(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title">
          <Bot size={28} color="#60a5fa" />
          Axiom v1.0
        </div>
        <div className="custom-dropdown" ref={dropdownRef}>
          <button 
            className="dropdown-trigger" 
            onClick={() => !isGenerating && setIsDropdownOpen(!isDropdownOpen)}
            disabled={isGenerating}
          >
            <CurrentIcon size={16} className="dropdown-icon" />
            <span>{currentMode?.name}</span>
            <ChevronDown size={16} className={`chevron ${isDropdownOpen ? 'open' : ''}`} />
          </button>
          
          {isDropdownOpen && (
            <div className="dropdown-menu">
              {modes.map((m) => {
                const Icon = m.icon;
                return (
                  <div 
                    key={m.id} 
                    className={`dropdown-item ${mode === m.id ? 'active' : ''}`}
                    onClick={() => { setMode(m.id); setIsDropdownOpen(false); }}
                  >
                    <div className="icon-container">
                      <Icon size={18} className="item-icon" />
                    </div>
                    <div className="item-text">
                      <div className="item-name">{m.name}</div>
                      <div className="item-desc">{m.desc}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </header>

      <main className="chat-container">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-wrapper ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} color="#60a5fa" />}
            </div>
            <div className="message-content">
              <div className="message-bubble">
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources-container">
                  {msg.sources.map((src, i) => (
                    <a key={i} href={src.startsWith('http') ? src : '#'} target="_blank" rel="noreferrer" className="source-pill">
                      <LinkIcon size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
                      {src.replace(/^https?:\/\/(www\.)?/, '')}
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isGenerating && messages[messages.length - 1].role === 'user' && (
           <div className="message-wrapper ai">
            <div className="message-avatar">
              <Bot size={20} color="#60a5fa" />
            </div>
             <div className="message-content">
               <div className="message-bubble">
                 <div className="thinking-indicator">
                   <div className="dot"></div>
                   <div className="dot"></div>
                   <div className="dot"></div>
                 </div>
               </div>
             </div>
           </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            className="chat-input"
            placeholder={isGenerating ? "Axiom is thinking..." : "Ask Axiom anything..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isGenerating}
          />
          <button type="submit" className="send-button" disabled={!input.trim() || isGenerating}>
            <Send size={20} />
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;
