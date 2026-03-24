import { useState, useRef, useEffect } from 'react';
import { API_BASE_URL } from '../config';
import {
  Send, Bot, User, Loader2, BookOpen, Trash2,
  ChevronDown, ChevronUp, Cpu, Zap,
} from 'lucide-react';

const CHAT_HISTORY_STORAGE_KEY = 'circuitai_chat_history';
const DEFAULT_ASSISTANT_MESSAGE = {
  role: 'assistant',
  content: "Hello! I'm CircuitAI. Ask me anything about the components in your knowledge base — specs, ratings, features, or pin configurations.",
  sources: [],
};

function sanitizeStoredMessages(rawMessages) {
  if (!Array.isArray(rawMessages) || rawMessages.length === 0) {
    return [DEFAULT_ASSISTANT_MESSAGE];
  }

  const sanitized = rawMessages
    .filter((msg) => msg && typeof msg === 'object' && typeof msg.role === 'string')
    .map((msg) => ({
      role: msg.role,
      content: typeof msg.content === 'string' ? msg.content : '',
      sources: Array.isArray(msg.sources) ? msg.sources : [],
      directMode: Boolean(msg.directMode),
      loading: false,
    }));

  return sanitized.length > 0 ? sanitized : [DEFAULT_ASSISTANT_MESSAGE];
}

function loadMessagesFromStorage() {
  if (typeof window === 'undefined') return [DEFAULT_ASSISTANT_MESSAGE];

  try {
    const savedMessages = window.localStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
    if (!savedMessages) return [DEFAULT_ASSISTANT_MESSAGE];
    return sanitizeStoredMessages(JSON.parse(savedMessages));
  } catch {
    return [DEFAULT_ASSISTANT_MESSAGE];
  }
}

// ── Markdown renderer with pipe-table support ─────────────────────────────────
function MarkdownContent({ text = '', isStreaming = false }) {
  const lines = text.split('\n');

  // Segment lines into plain-text blocks and contiguous table blocks
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trimStart().startsWith('|')) {
      const tableLines = [];
      while (i < lines.length && lines[i].trimStart().startsWith('|')) {
        tableLines.push(lines[i]);
        i++;
      }
      blocks.push({ type: 'table', lines: tableLines });
    } else {
      blocks.push({ type: 'line', content: lines[i] });
      i++;
    }
  }

  // Parse | header | ... | sep | ... | rows into arrays
  function parseTable(tableLines) {
    const parseRow = (line) =>
      line.split('|').slice(1, -1).map((c) => c.trim());
    const header = parseRow(tableLines[0]);
    // skip the divider row (|---|---|...)
    const rows = tableLines.slice(2).map(parseRow);
    return { header, rows };
  }

  // Render inline markdown: **bold**, *italic*, `code`
  function renderInline(str, key) {
    const parts = str.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
    return (
      <span key={key}>
        {parts.map((part, j) => {
          if (part.startsWith('**') && part.endsWith('**'))
            return <strong key={j} className="text-slate-100 font-semibold">{part.slice(2, -2)}</strong>;
          if (part.startsWith('*') && part.endsWith('*'))
            return <em key={j} className="text-slate-400 italic">{part.slice(1, -1)}</em>;
          if (part.startsWith('`') && part.endsWith('`'))
            return <code key={j} className="bg-slate-700/60 text-cyan-300 px-1 rounded text-xs font-mono">{part.slice(1, -1)}</code>;
          return <span key={j}>{part}</span>;
        })}
      </span>
    );
  }

  const lastIdx = blocks.length - 1;
  return (
    <div className="text-sm leading-relaxed space-y-1.5">
      {blocks.map((block, bi) => {
        // ── Table block ────────────────────────────────────────────────────
        if (block.type === 'table') {
          const { header, rows } = parseTable(block.lines);
          return (
            <div key={bi} className="overflow-x-auto my-3 rounded-lg border border-slate-600/50">
              <table className="min-w-full text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-700/70">
                    {header.map((h, hi) => (
                      <th
                        key={hi}
                        className="px-3 py-2 text-left font-semibold text-indigo-300 border-b border-slate-600/60 whitespace-nowrap"
                      >
                        {renderInline(h, hi)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, ri) => (
                    <tr key={ri} className={ri % 2 === 0 ? 'bg-slate-800/40' : 'bg-slate-800/20'}>
                      {row.map((cell, ci) => (
                        <td
                          key={ci}
                          className="px-3 py-1.5 text-slate-300 border-b border-slate-700/30 whitespace-nowrap font-mono"
                        >
                          {renderInline(cell, ci)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }

        // ── Plain line block ───────────────────────────────────────────────
        const line = block.content;
        const isLast = bi === lastIdx;
        if (line === '---') return <div key={bi} className="h-px bg-slate-700/60 my-2" />;
        if (line.startsWith('### '))
          return <h3 key={bi} className="text-slate-200 font-semibold text-xs uppercase tracking-wide text-indigo-300 mt-3">{line.slice(4)}</h3>;
        if (line.startsWith('## '))
          return <h2 key={bi} className="text-slate-100 font-bold text-sm mt-3">{line.slice(3)}</h2>;
        if (line === '```' || line.startsWith('```')) return null;
        if (!line.trim()) return <div key={bi} className="h-1" />;

        return (
          <p key={bi} className="text-slate-300">
            {renderInline(line, bi)}
            {isStreaming && isLast && (
              <span className="inline-block w-0.5 h-3.5 bg-indigo-400 animate-pulse ml-0.5 align-middle" />
            )}
          </p>
        );
      })}
    </div>
  );
}

// ── Source citation card ──────────────────────────────────────────────────────
function SourceCard({ source, index }) {
  const [open, setOpen] = useState(false);
  const scoreColor =
    source.score > 0.7 ? 'text-emerald-400' :
    source.score > 0.4 ? 'text-yellow-400'  : 'text-slate-500';

  return (
    <div className="border border-slate-700/50 rounded-lg overflow-hidden bg-slate-800/30 text-[11px]">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-slate-700/30 transition-colors text-left gap-2"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-[9px] font-bold text-indigo-300">
            {index + 1}
          </span>
          <span className="font-mono text-indigo-300 truncate">{source.component?.replace(/_/g, ' ')}</span>
          {source.section && (
            <span className="text-slate-600">·</span>
          )}
          {source.section && (
            <span className="text-slate-500 truncate italic">{source.section.replace(/_/g, ' ')}</span>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className={`font-mono font-semibold ${scoreColor}`}>{source.score.toFixed(3)}</span>
          {open ? <ChevronUp className="w-3 h-3 text-slate-500" /> : <ChevronDown className="w-3 h-3 text-slate-500" />}
        </div>
      </button>

      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-slate-700/40">
          <div className="flex gap-2 mb-2">
            {source.type && (
              <span className="bg-slate-700/50 text-slate-400 border border-slate-600/30 px-1.5 py-0.5 rounded text-[9px] font-medium">
                {source.type.replace(/_/g, ' ')}
              </span>
            )}
            <span className="text-slate-600 font-mono text-[9px]">{source.id}</span>
          </div>
          <p className="text-slate-400 leading-relaxed whitespace-pre-wrap break-words">{source.text}</p>
        </div>
      )}
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────
function Message({ msg, elapsed = 0 }) {
  const isUser = msg.role === 'user';
  const [showSources, setShowSources] = useState(false);

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
        ${isUser
          ? 'bg-indigo-500/20 border border-indigo-500/40'
          : 'bg-slate-700/60 border border-slate-600/40'
        }`}
      >
        {isUser
          ? <User className="w-4 h-4 text-indigo-300" />
          : <Bot className="w-4 h-4 text-slate-300" />
        }
      </div>

      <div className={`max-w-[80%] space-y-2 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        {/* Bubble */}
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed
          ${isUser
            ? 'bg-indigo-600/80 text-white rounded-tr-sm'
            : 'bg-slate-800/70 text-slate-200 border border-slate-700/50 rounded-tl-sm'
          }`}
        >
          {msg.role === 'assistant' && msg.loading && !msg.content ? (
            <div className="flex items-center gap-2 text-slate-400">
              <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
              <span className="text-sm">Searching datasheet…</span>
              <span className="font-mono text-xs text-slate-500">{elapsed}s</span>
            </div>
          ) : (
            <MarkdownContent text={msg.content} isStreaming={msg.loading} />
          )}
        </div>

        {/* Direct-context badge */}
        {msg.directMode && (
          <div className="flex items-center gap-1.5 text-[11px] text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2.5 py-1 rounded-lg">
            <BookOpen className="w-3 h-3" />
            Direct Context Mode — extracted from datasheet (LLM unavailable)
          </div>
        )}

        {/* Sources */}
        {msg.sources?.length > 0 && (
          <div className="w-full">
            <button
              onClick={() => setShowSources(v => !v)}
              className="flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
            >
              <BookOpen className="w-3 h-3" />
              {showSources ? 'Hide' : 'Show'} {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
              {showSources ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showSources && (
              <div className="mt-2 space-y-1.5 w-full">
                {msg.sources.map((s, i) => (
                  <SourceCard key={s.id || i} source={s} index={i} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Custom dark dropdown for component filter ─────────────────────────────────
function ComponentDropdown({ components, value, onChange }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const label = value
    ? components.find(c => c.component_id === value)?.component_id || value
    : 'All components';

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-2 bg-slate-800 border border-slate-600/60 hover:border-indigo-500/50 text-slate-200 text-xs rounded-lg px-3 py-2 font-mono transition-all min-w-[160px] max-w-[200px] focus:outline-none focus:ring-1 focus:ring-indigo-500/40"
      >
        <span className="truncate flex-1 text-left">{label}</span>
        <ChevronDown className={`w-3.5 h-3.5 text-slate-400 flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 w-56 bg-slate-800 border border-slate-600/60 rounded-xl shadow-xl shadow-black/40 overflow-hidden">
          <button
            onClick={() => { onChange(''); setOpen(false); }}
            className={`w-full text-left px-3 py-2.5 text-xs font-mono transition-colors flex items-center gap-2
              ${!value
                ? 'bg-indigo-600/30 text-indigo-200 border-l-2 border-indigo-400'
                : 'text-slate-300 hover:bg-slate-700/70 hover:text-slate-100 border-l-2 border-transparent'
              }`}
          >
            All components
          </button>

          <div className="h-px bg-slate-700/60 mx-2" />

          <div className="max-h-56 overflow-y-auto">
            {[...components]
              .sort((a, b) => (b.ingested_at || '').localeCompare(a.ingested_at || ''))
              .map(c => (
              <button
                key={c.component_id}
                onClick={() => { onChange(c.component_id); setOpen(false); }}
                className={`w-full text-left px-3 py-2.5 text-xs font-mono transition-colors flex items-center justify-between gap-2
                  ${value === c.component_id
                    ? 'bg-indigo-600/30 text-indigo-200 border-l-2 border-indigo-400'
                    : 'text-slate-300 hover:bg-slate-700/70 hover:text-slate-100 border-l-2 border-transparent'
                  }`}
              >
                <span className="truncate">{c.component_id.replace(/_/g, ' ')}</span>
                <span className="text-[10px] text-slate-500 flex-shrink-0">{c.chunk_count}v</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main ChatPanel ────────────────────────────────────────────────────────────
export default function ChatPanel({ components = [] }) {
  const [messages, setMessages]         = useState(loadMessagesFromStorage);
  const [input, setInput]               = useState('');
  const [loading, setLoading]           = useState(false);
  const [componentFilter, setComponentFilter] = useState('');
  const scrollRef                       = useRef(null);
  const inputRef                        = useRef(null);

  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef(null);

  useEffect(() => () => clearInterval(elapsedRef.current), []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // Ignore storage failures
    }
  }, [messages]);

  const clearChat = () => {
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
    }
    clearInterval(elapsedRef.current);
    setMessages([DEFAULT_ASSISTANT_MESSAGE]);
    setLoading(false);
    setElapsed(0);
    inputRef.current?.focus();
  };

  const sendMessage = async () => {
    const query = input.trim();
    if (!query || loading) return;

    setMessages(prev => [...prev, { role: 'user', content: query }]);
    setInput('');
    setLoading(true);
    setElapsed(0);

    setMessages(prev => [...prev, { role: 'assistant', content: '', loading: true, sources: [] }]);
    elapsedRef.current = setInterval(() => setElapsed(s => s + 1), 1000);

    let fullText = '';
    let receivedSources = [];
    let isDirectMode = false;

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          component_filter: componentFilter || null,
          top_k: 10,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Stream request failed' }));
        throw new Error(err.detail || 'Stream request failed');
      }

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = '';

      while (true) {
        const { done, value } = await reader.read();
        if (value) buffer += decoder.decode(value, { stream: !done });

        const lines = buffer.split('\n');
        buffer = done ? '' : (lines.pop() ?? '');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const evt = JSON.parse(line.slice(6));

            if (evt.type === 'sources') {
              receivedSources = evt.sources || [];
              setMessages(prev => {
                const u = [...prev];
                u[u.length - 1] = { ...u[u.length - 1], sources: receivedSources };
                return u;
              });
            }

            if (evt.type === 'token') {
              fullText += evt.token;
              setMessages(prev => {
                const u = [...prev];
                u[u.length - 1] = { ...u[u.length - 1], content: fullText, loading: true };
                return u;
              });
            }

            if (evt.type === 'mode' && evt.mode === 'direct') isDirectMode = true;

            if (evt.type === 'error') fullText = fullText || `❌ ${evt.message}`;

            if (evt.type === 'done') {
              setMessages(prev => {
                const u = [...prev];
                u[u.length - 1] = {
                  role:       'assistant',
                  content:    fullText || 'No response generated.',
                  sources:    receivedSources,
                  directMode: isDirectMode,
                  loading:    false,
                };
                return u;
              });
            }
          } catch { /* ignore malformed lines */ }
        }

        if (done) break;
      }
    } catch (err) {
      const errMsg = err.message || 'Request failed';
      const fallbackMsg = receivedSources.length > 0
        ? `**📋 Datasheet Results for:** *${query}*\n\n` +
          `*LLM is unavailable, but here are the relevant datasheet sources:*\n\n` +
          receivedSources.map((s) => {
            const comp = (s.component || 'unknown').replace(/_/g, ' ');
            const section = (s.section || '').replace(/_/g, ' ');
            return `### 📄 ${comp}\n*Section: ${section}*\n${s.text || ''}\n*Score: ${s.score}*\n`;
          }).join('\n')
        : `⚠️ **Could not connect to the backend server.**\n\nPlease make sure the backend is running at \`${API_BASE_URL}\`.\n\n*Error: ${errMsg}*`;

      setMessages(prev => {
        const u = [...prev];
        u[u.length - 1] = {
          role: 'assistant',
          content: fallbackMsg,
          sources: receivedSources,
          directMode: true,
          loading: false,
        };
        return u;
      });
    } finally {
      clearInterval(elapsedRef.current);
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const suggestions = [
    'What is the maximum voltage rating?',
    'List the electrical characteristics.',
    'What are the key features?',
    'What are the pin configurations?',
  ];

  return (
    <div className="glass-panel rounded-2xl flex flex-col h-[680px]">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-700/50 bg-slate-800/20 rounded-t-2xl flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-slate-200">CircuitAI Chat</h2>
            <p className="text-xs text-slate-500 flex items-center gap-1">
              <Cpu className="w-3 h-3" /> RAG · HuggingFace · BGE-M3
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {components.length > 0 && (
            <ComponentDropdown
              components={components}
              value={componentFilter}
              onChange={setComponentFilter}
            />
          )}
          <button
            type="button"
            onClick={clearChat}
            disabled={loading}
            className="flex items-center gap-1.5 bg-slate-800 border border-slate-600/60 hover:border-red-500/60 text-slate-300 hover:text-red-300 text-xs rounded-lg px-3 py-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            title="Clear chat history"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Chat
          </button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto custom-scrollbar px-5 py-4 space-y-5"
        style={{ overscrollBehavior: 'contain' }}
      >
        {messages.map((msg, i) => (
          <Message
            key={i}
            msg={msg}
            elapsed={msg.loading ? elapsed : 0}
          />
        ))}
      </div>

      {/* Suggestions (only when no conversation yet) */}
      {messages.length === 1 && (
        <div className="px-5 pb-2 flex flex-wrap gap-2">
          {suggestions.map(s => (
            <button
              key={s}
              onClick={() => { setInput(s); inputRef.current?.focus(); }}
              className="text-[11px] bg-slate-800/50 border border-slate-700/50 hover:border-indigo-500/40 hover:text-indigo-300 text-slate-400 px-3 py-1.5 rounded-full transition-all"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input bar */}
      <div className="px-4 pb-4 pt-2 border-t border-slate-700/40">
        <div className="flex items-end gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask about component specs, ratings, features…"
              rows={1}
              disabled={loading}
              className="w-full bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-3 pr-12 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30 transition-all disabled:opacity-50"
              style={{ minHeight: '48px', maxHeight: '120px', overflowY: 'auto' }}
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            className="flex-shrink-0 w-11 h-11 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white flex items-center justify-center transition-all shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30 active:scale-95"
          >
            {loading
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Send className="w-4 h-4" />
            }
          </button>
        </div>
        <p className="text-[10px] text-slate-700 mt-1.5 text-center">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
