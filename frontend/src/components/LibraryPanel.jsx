import { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import { BookOpen, RefreshCw, Cpu, Layers, ChevronRight, Search, Database, ArrowUpDown, ArrowDownAZ, Calendar, HardDrive, Trash2, AlertTriangle, X } from 'lucide-react';

// Colour per chunk type
const TYPE_COLORS = {
  table:                         'bg-sky-500/15 text-sky-300 border-sky-500/25',
  figure:                        'bg-violet-500/15 text-violet-300 border-violet-500/25',
  description:                   'bg-indigo-500/15 text-indigo-300 border-indigo-500/25',
  features:                      'bg-emerald-500/15 text-emerald-300 border-emerald-500/25',
  electrical_characteristics:    'bg-orange-500/15 text-orange-300 border-orange-500/25',
  absolute_maximum_ratings:      'bg-red-500/15 text-red-300 border-red-500/25',
  recommended_operating_conditions: 'bg-yellow-500/15 text-yellow-300 border-yellow-500/25',
  applications:                  'bg-teal-500/15 text-teal-300 border-teal-500/25',
  pin_configuration:             'bg-pink-500/15 text-pink-300 border-pink-500/25',
};
const DEFAULT_TYPE_COLOR = 'bg-slate-700/50 text-slate-400 border-slate-600/30';

function typeColor(t) {
  return TYPE_COLORS[t] || DEFAULT_TYPE_COLOR;
}

// Sort options
const SORT_OPTIONS = [
  { key: 'alpha-asc',  label: 'A → Z',           icon: ArrowDownAZ },
  { key: 'alpha-desc', label: 'Z → A',           icon: ArrowDownAZ },
  { key: 'date-new',   label: 'Newest First',    icon: Calendar },
  { key: 'date-old',   label: 'Oldest First',    icon: Calendar },
  { key: 'size-large', label: 'Largest First',   icon: HardDrive },
  { key: 'size-small', label: 'Smallest First',  icon: HardDrive },
  { key: 'chunks-most',label: 'Most Chunks',     icon: Layers },
];

function sortComponents(components, sortKey) {
  const list = [...components];
  switch (sortKey) {
    case 'alpha-asc':
      return list.sort((a, b) => a.component_id.localeCompare(b.component_id));
    case 'alpha-desc':
      return list.sort((a, b) => b.component_id.localeCompare(a.component_id));
    case 'date-new':
      return list.sort((a, b) => {
        const da = a.ingested_at || '';
        const db = b.ingested_at || '';
        return db.localeCompare(da);
      });
    case 'date-old':
      return list.sort((a, b) => {
        const da = a.ingested_at || '';
        const db = b.ingested_at || '';
        return da.localeCompare(db);
      });
    case 'size-large':
      return list.sort((a, b) => (b.file_size_kb || 0) - (a.file_size_kb || 0));
    case 'size-small':
      return list.sort((a, b) => (a.file_size_kb || 0) - (b.file_size_kb || 0));
    case 'chunks-most':
      return list.sort((a, b) => b.chunk_count - a.chunk_count);
    default:
      return list;
  }
}

function formatDate(iso) {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
  } catch { return null; }
}

function formatSize(kb) {
  if (!kb) return null;
  if (kb >= 1024) return `${(kb / 1024).toFixed(1)} MB`;
  return `${Math.round(kb)} KB`;
}

export default function LibraryPanel({ refreshTrigger = 0, onComponentsLoaded }) {
  const [data, setData]         = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [query, setQuery]       = useState('');
  const [sortKey, setSortKeyState] = useState(() => localStorage.getItem('kb_sort') || 'alpha-asc');
  const setSortKey = (key) => { setSortKeyState(key); localStorage.setItem('kb_sort', key); };
  const [showSort, setShowSort] = useState(false);
  // confirmDelete: component_id of the item awaiting confirm, or null
  const [confirmDelete, setConfirmDelete] = useState(null);
  // deleting: set of component_ids currently being deleted
  const [deleting, setDeleting] = useState(new Set());

  const fetchLibrary = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`${API_BASE_URL}/library`);
      setData(res.data);
      onComponentsLoaded?.(res.data.components || []);
    } catch (e) {
      setError('Failed to load library.');
    } finally {
      setLoading(false);
    }
  };

  // Re-fetch whenever initially mounted or when a job finishes (refreshTrigger increments)
  useEffect(() => { fetchLibrary(); }, [refreshTrigger]);

  const deleteComponent = async (componentId) => {
    setDeleting(prev => new Set([...prev, componentId]));
    setConfirmDelete(null);
    try {
      await axios.delete(`${API_BASE_URL}/library/${encodeURIComponent(componentId)}`);
      // Optimistically remove from local state — no need to re-fetch
      setData(prev => {
        if (!prev) return prev;
        const components = prev.components.filter(c => c.component_id !== componentId);
        const removedChunks = prev.components.find(c => c.component_id === componentId)?.chunk_count || 0;
        return {
          ...prev,
          total_components: prev.total_components - 1,
          total_vectors: prev.total_vectors - removedChunks,
          components,
        };
      });
      onComponentsLoaded?.(data?.components?.filter(c => c.component_id !== componentId) || []);
    } catch (e) {
      console.error('Failed to delete component', e);
    } finally {
      setDeleting(prev => { const s = new Set(prev); s.delete(componentId); return s; });
    }
  };

  const filtered = (data?.components || []).filter(c =>
    c.component_id.toLowerCase().includes(query.toLowerCase())
  );

  const sorted = sortComponents(filtered, sortKey);

  const currentSort = SORT_OPTIONS.find(o => o.key === sortKey);

  return (
    <div className="glass-panel rounded-2xl flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/30 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-indigo-400" />
          Knowledge Base
          {data && (
            <span className="ml-1 text-[11px] font-mono bg-indigo-500/15 text-indigo-300 border border-indigo-500/25 px-1.5 py-0.5 rounded-full">
              {data.total_components} components · {data.total_vectors} vectors
            </span>
          )}
        </h3>
        <div className="flex items-center gap-1">
          <button
            onClick={fetchLibrary}
            className="p-1 rounded hover:bg-slate-700/60 text-slate-500 hover:text-slate-300 transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Search + Sort bar */}
      {data && data.total_components > 0 && (
        <div className="px-3 py-2 border-b border-slate-700/30 flex items-center gap-2">
          {/* Search input */}
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
            <input
              type="text"
              placeholder="Filter components…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30 transition-all"
            />
          </div>

          {/* Sort dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowSort(v => !v)}
              className="flex items-center gap-1.5 bg-slate-800/50 border border-slate-700/50 hover:border-indigo-500/40 rounded-lg px-2.5 py-1.5 text-[11px] text-slate-400 hover:text-slate-200 transition-all"
              title="Sort components"
            >
              <ArrowUpDown className="w-3 h-3" />
              <span className="hidden sm:inline">{currentSort?.label}</span>
            </button>

            {showSort && (
              <div className="absolute right-0 top-full mt-1 z-50 w-44 bg-slate-800 border border-slate-600/60 rounded-xl shadow-xl shadow-black/40 overflow-hidden">
                {SORT_OPTIONS.map(opt => {
                  const Icon = opt.icon;
                  const active = sortKey === opt.key;
                  return (
                    <button
                      key={opt.key}
                      onClick={() => { setSortKey(opt.key); setShowSort(false); }}
                      className={`w-full text-left px-3 py-2 text-[11px] transition-colors flex items-center gap-2
                        ${active
                          ? 'bg-indigo-600/30 text-indigo-200 border-l-2 border-indigo-400'
                          : 'text-slate-300 hover:bg-slate-700/70 hover:text-slate-100 border-l-2 border-transparent'
                        }`}
                    >
                      <Icon className="w-3 h-3 flex-shrink-0" />
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar max-h-64 p-2">

        {loading && (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-5 h-5 text-indigo-400 animate-spin" />
          </div>
        )}

        {error && !loading && (
          <p className="text-xs text-red-400 text-center py-6">{error}</p>
        )}

        {!loading && !error && sorted.length === 0 && (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Database className="w-8 h-8 text-slate-700 mb-2" />
            <p className="text-xs text-slate-600">
              {query ? 'No matching components.' : 'No components indexed yet.'}
            </p>
          </div>
        )}

        {!loading && !error && sorted.map((component, i) => {
          const isConfirming = confirmDelete === component.component_id;
          const isDeleting   = deleting.has(component.component_id);

          return (
            <div
              key={component.component_id}
              className={`flex items-start justify-between px-3 py-2.5 mb-1 rounded-lg border transition-all group
                ${isConfirming
                  ? 'bg-red-500/8 border-red-500/30'
                  : 'bg-slate-800/30 border-slate-700/40 hover:border-indigo-500/30 hover:bg-slate-800/60'
                }
                ${isDeleting ? 'opacity-40 pointer-events-none' : ''}
              `}
            >
              <div className="flex items-start gap-2.5 overflow-hidden flex-1 min-w-0">
                {/* Index number */}
                <span className="text-[10px] font-mono text-slate-600 mt-0.5 w-4 flex-shrink-0">
                  {String(i + 1).padStart(2, '0')}
                </span>

                <div className="min-w-0 flex-1">
                  {/* Component ID */}
                  <p
                    className="text-xs font-mono text-slate-200 truncate group-hover:text-indigo-300 transition-colors"
                    title={component.component_id}
                  >
                    {component.component_id.replace(/_/g, ' ')}
                  </p>

                  {/* Metadata line: date + size */}
                  <div className="flex items-center gap-2 mt-0.5">
                    {component.ingested_at && (
                      <span className="text-[9px] text-slate-500 flex items-center gap-1" title="Ingested date">
                        <Calendar className="w-2.5 h-2.5" />
                        {formatDate(component.ingested_at)}
                      </span>
                    )}
                    {component.file_size_kb && (
                      <span className="text-[9px] text-slate-500 flex items-center gap-1" title="PDF file size">
                        <HardDrive className="w-2.5 h-2.5" />
                        {formatSize(component.file_size_kb)}
                      </span>
                    )}
                  </div>

                  {/* Chunk type pills */}
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {component.chunk_types.map(t => (
                      <span
                        key={t}
                        className={`inline-block text-[9px] font-medium px-1.5 py-0.5 rounded border ${typeColor(t)}`}
                      >
                        {t.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>

                  {/* Inline confirm prompt */}
                  {isConfirming && (
                    <div className="flex items-center gap-2 mt-2">
                      <AlertTriangle className="w-3 h-3 text-red-400 flex-shrink-0" />
                      <span className="text-[10px] text-red-300">Remove from knowledge base?</span>
                      <button
                        onClick={() => deleteComponent(component.component_id)}
                        className="text-[10px] bg-red-500/20 hover:bg-red-500/40 text-red-300 hover:text-red-200 border border-red-500/30 px-2 py-0.5 rounded transition-all font-medium"
                      >
                        Delete
                      </button>
                      <button
                        onClick={() => setConfirmDelete(null)}
                        className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Right side: chunk count + delete */}
              <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                <div className="flex items-center gap-1">
                  <Layers className="w-3 h-3 text-slate-600" />
                  <span className="text-[11px] font-mono text-slate-500">
                    {component.chunk_count}
                  </span>
                </div>
                {/* Delete button — visible on hover or while confirming */}
                {!isConfirming && (
                  <button
                    onClick={() => setConfirmDelete(component.component_id)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded-md bg-slate-700/40 hover:bg-red-500/20 text-slate-600 hover:text-red-400 transition-all"
                    title="Remove from knowledge base"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
