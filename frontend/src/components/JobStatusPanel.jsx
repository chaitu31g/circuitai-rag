import { useEffect, useRef, useState, useCallback } from 'react';
import axios from 'axios';
import {
  RefreshCw, CheckCircle, Clock, Database,
  AlertCircle, FileText, FileJson, Cpu, HardDrive,
  Terminal, ChevronDown, ChevronUp, Trash2, X,
} from 'lucide-react';

const STAGES = [
  { key: 'parsing',   label: 'Parsing',   Icon: FileText  },
  { key: 'chunking',  label: 'Chunking',  Icon: FileJson  },
  { key: 'embedding', label: 'Embedding', Icon: Cpu       },
  { key: 'storing',   label: 'Storing',   Icon: HardDrive },
];

// ── Stage progress pills ───────────────────────────────────────────────────────
function PipelineProgress({ job }) {
  if (job.status === 'pending') return null;
  const isFinished = job.status === 'done' || job.status === 'failed';

  return (
    <div className="mt-3 ml-8 pl-3 border-l-[1.5px] border-slate-700">
      <div className="flex items-center gap-2 flex-wrap">
        {STAGES.map(({ key, label, Icon }, idx) => {
          const isDone   = job.stages_done?.includes(key);
          const isActive = !isFinished && job.current_stage === key;
          const isFailed = job.status === 'failed' && job.current_stage === key;

          let text   = 'text-slate-500';
          let bg     = 'bg-slate-800/50';
          let border = 'border-slate-700/40';
          let icon   = 'text-slate-600';

          if (isDone)        { text = 'text-emerald-300'; bg = 'bg-emerald-500/10'; border = 'border-emerald-500/25'; icon = 'text-emerald-400'; }
          else if (isFailed) { text = 'text-red-300';     bg = 'bg-red-500/10';     border = 'border-red-500/25';     icon = 'text-red-400'; }
          else if (isActive) { text = 'text-yellow-300';  bg = 'bg-yellow-500/10';  border = 'border-yellow-500/25';  icon = 'text-yellow-400'; }

          return (
            <div key={key} className="flex items-center gap-1">
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium transition-all ${bg} ${text} ${border}`}>
                {isActive
                  ? <RefreshCw className={`w-2.5 h-2.5 animate-spin ${icon}`} />
                  : <Icon className={`w-2.5 h-2.5 ${icon}`} />
                }
                {label}
              </span>
              {idx < STAGES.length - 1 && <span className="text-slate-700 text-xs select-none">›</span>}
            </div>
          );
        })}
      </div>

      {job.status === 'done' && (
        <div className="flex items-center gap-2 mt-2 text-xs text-slate-400 flex-wrap">
          <span className="bg-emerald-500/10 text-emerald-300 font-mono px-1.5 py-0.5 rounded text-[11px]">
            +{job.chunks_created} vectors
          </span>
          {job.component_id && (
            <span className="text-slate-500">
              Component: <span className="text-indigo-300 font-mono">{job.component_id}</span>
            </span>
          )}
          {job.processing_time_sec && (
            <span className="text-slate-500">{job.processing_time_sec.toFixed(1)}s total</span>
          )}
        </div>
      )}

      {job.status === 'failed' && job.error_message && (
        <div className="text-xs text-red-400/90 break-words mt-2">
          <span className="font-semibold block mb-0.5">Error:</span>
          {job.error_message}
        </div>
      )}
    </div>
  );
}

// ── Terminal box ───────────────────────────────────────────────────────────────
function TerminalBox({ job }) {
  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState([]);
  const scrollRef       = useRef(null);          // the scrollable <div>, NOT a sentinel
  const isActive        = job.status === 'processing' || job.status === 'pending';

  // Auto-open when processing starts
  useEffect(() => {
    if (job.status === 'processing') setOpen(true);
  }, [job.status]);

  // Poll logs
  useEffect(() => {
    let timer = null;
    const fetch = async () => {
      try {
        const res = await axios.get(`http://localhost:8000/jobs/${job.job_id}/logs`);
        setLogs(res.data.logs || []);
      } catch { /* ignore */ }
    };
    if (open) {
      fetch();
      if (isActive) timer = setInterval(fetch, 1200);
    }
    return () => { if (timer) clearInterval(timer); };
  }, [open, isActive, job.job_id]);

  // Scroll ONLY the terminal container — never the page
  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, open]);

  // Color a log line
  const lineColor = (line) => {
    if (line.includes('✅') || line.includes('complete ✔') || line.includes('✔'))
      return 'text-emerald-400';
    if (line.includes('❌') || line.includes('FAILED') || line.includes('[ERROR]') || line.includes('[WARN]'))
      return 'text-red-400';
    if (line.startsWith('━━━'))
      return 'text-cyan-300 font-semibold';
    if (line.startsWith('┌─'))
      return 'text-indigo-300 font-semibold';
    if (line.startsWith('│') && line.includes('✔'))
      return 'text-emerald-400';
    if (line.startsWith('│'))
      return 'text-slate-300';
    if (line.startsWith('└─'))
      return 'text-emerald-400';
    if (line.includes('[INFO]') || line.includes('INFO'))
      return 'text-sky-300';
    if (line.includes('saved →'))
      return 'text-indigo-300';
    return 'text-slate-400';
  };

  return (
    <div className="mt-2 ml-8">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-slate-300 transition-colors group"
      >
        <Terminal className="w-3 h-3 group-hover:text-primary-400" />
        <span>Terminal</span>
        {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        {isActive && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse ml-0.5" />}
        {!isActive && logs.length > 0 && (
          <span className="font-mono text-[10px] text-slate-600 ml-1">{logs.length} lines</span>
        )}
      </button>

      {open && (
        <div className="mt-1.5 rounded-lg bg-[#0c0c0c] border border-slate-700/60 overflow-hidden">
          {/* Windows Terminal title bar */}
          <div className="flex items-center justify-between bg-[#1f1f1f] border-b border-[#333] select-none">
            {/* Left: tab */}
            <div className="flex items-center">
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0c0c0c] border-r border-[#333] border-b border-b-transparent">
                <span className="text-[10px] text-blue-400">⚡</span>
                <span className="text-[10px] text-slate-300 font-mono">PowerShell</span>
              </div>
              <span className="text-[10px] text-slate-600 font-mono px-3 truncate max-w-[180px]">{job.file_name}</span>
              {isActive && (
                <span className="text-[9px] text-emerald-400 animate-pulse font-mono">● live</span>
              )}
            </div>
            {/* Right: window controls */}
            <div className="flex items-center">
              <span className="px-3 py-1.5 text-slate-500 hover:bg-[#333] text-[11px] cursor-default">─</span>
              <span className="px-3 py-1.5 text-slate-500 hover:bg-[#333] text-[11px] cursor-default">□</span>
              <span className="px-3 py-1.5 text-slate-500 hover:bg-red-600 hover:text-white text-[11px] cursor-default">✕</span>
            </div>
          </div>

          {/* Log body — scrolls internally, never touches the page */}
          <div
            ref={scrollRef}
            className="p-3 h-52 overflow-y-auto font-mono text-[11px] leading-relaxed space-y-0.5"
            style={{ overscrollBehavior: 'contain' }}
          >
            {logs.length === 0
              ? <span className="text-slate-600 italic">Waiting for output…</span>
              : logs.map((line, i) => (
                  <div key={i} className={`whitespace-pre-wrap break-all ${lineColor(line)}`}>
                    {line || '\u00A0'}
                  </div>
                ))
            }
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────
export default function JobStatusPanel({ activeJobsIds, onJobCompleted, onActiveStage }) {
  const [jobs, setJobs]       = useState([]);
  const [loading, setLoading] = useState(true);
  const prevStatuses          = useRef({});

  const fetchJobs = useCallback(async () => {
    try {
      const res = await axios.get('http://localhost:8000/jobs');
      const newJobs = res.data;

      // Detect any job that just flipped to "done" → notify parent to refresh library
      newJobs.forEach(j => {
        const prev = prevStatuses.current[j.job_id];
        if (prev && prev !== 'done' && j.status === 'done') {
          onJobCompleted?.();
        }
        prevStatuses.current[j.job_id] = j.status;
      });

      // Determine the active pipeline stage across all jobs
      const processingJob = newJobs.find(j => j.status === 'processing');
      if (processingJob && processingJob.current_stage) {
        onActiveStage?.(processingJob.current_stage);
      } else {
        onActiveStage?.(null);
      }

      setJobs(newJobs);
    } catch (err) {
      console.error('Failed to fetch jobs', err);
    } finally {
      setLoading(false);
    }
  }, [onJobCompleted, onActiveStage]);

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(() => {
      const needsPolling =
        jobs.some(j => j.status === 'pending' || j.status === 'processing') ||
        activeJobsIds.length > 0;
      if (needsPolling) fetchJobs();
    }, 1500);
    return () => clearInterval(interval);
  }, [jobs, activeJobsIds, fetchJobs]);

  const getStatusIcon = (s) => {
    if (s === 'done')       return <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />;
    if (s === 'failed')     return <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />;
    if (s === 'processing') return <RefreshCw   className="w-5 h-5 text-yellow-500 animate-spin flex-shrink-0" />;
    return <Clock className="w-5 h-5 text-slate-500 flex-shrink-0" />;
  };

  const getStatusBadge = (s) => {
    if (s === 'done')       return <span className="bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-2.5 py-1 rounded-md text-xs font-semibold uppercase tracking-wider whitespace-nowrap">Done</span>;
    if (s === 'failed')     return <span className="bg-red-500/15 text-red-400 border border-red-500/30 px-2.5 py-1 rounded-md text-xs font-semibold uppercase tracking-wider whitespace-nowrap">Failed</span>;
    if (s === 'processing') return <span className="bg-yellow-500/15 text-yellow-300 border border-yellow-500/30 px-2.5 py-1 rounded-md text-xs font-semibold uppercase tracking-wider whitespace-nowrap animate-pulse">Ingesting…</span>;
    return <span className="bg-slate-700/50 text-slate-300 border border-slate-600/50 px-2.5 py-1 rounded-md text-xs font-semibold uppercase tracking-wider whitespace-nowrap">Queued</span>;
  };

  if (loading) return (
    <div className="glass-panel rounded-2xl p-8 flex items-center justify-center min-h-[400px]">
      <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
    </div>
  );

  if (jobs.length === 0) return (
    <div className="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center text-center min-h-[400px]">
      <Database className="w-12 h-12 text-slate-600 mb-4" />
      <h3 className="text-lg font-medium text-slate-300">No Ingestions Yet</h3>
      <p className="text-slate-500 text-sm mt-2 max-w-xs">Upload datasheets on the left to begin compiling your vector database.</p>
    </div>
  );

  const hasFinishedJobs = jobs.some(j => j.status === 'done' || j.status === 'failed');

  const deleteJob = async (jobId) => {
    try {
      await axios.delete(`http://localhost:8000/jobs/${jobId}`);
      setJobs(prev => prev.filter(j => j.job_id !== jobId));
    } catch (err) {
      console.error('Failed to delete job', err);
    }
  };

  const deleteAllJobs = async () => {
    try {
      await axios.delete('http://localhost:8000/jobs');
      fetchJobs();
    } catch (err) {
      console.error('Failed to clear jobs', err);
    }
  };

  return (
    <div className="glass-panel rounded-2xl flex flex-col min-h-[400px] h-[680px]">
      <div className="p-4 border-b border-slate-700/50 flex justify-between items-center bg-slate-800/20 sticky top-0 z-10 rounded-t-2xl">
        <h2 className="text-lg font-semibold text-slate-200 flex items-center">
          <Database className="w-5 h-5 mr-2 text-primary-400" />
          Pipeline History
        </h2>
        <div className="flex items-center gap-1.5">
          {hasFinishedJobs && (
            <button
              onClick={deleteAllJobs}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 hover:border-red-500/40 text-red-400 hover:text-red-300 text-[11px] font-medium transition-all"
              title="Clear all finished (completed and failed) jobs"
            >
              <Trash2 className="w-3 h-3" />
              Clear All
            </button>
          )}
          <button onClick={fetchJobs} className="p-1.5 rounded bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
        {jobs.map((job) => {
          const canDelete = job.status !== 'processing';
          return (
            <div key={job.job_id} className="group p-4 mb-2 rounded-xl bg-slate-800/30 border border-slate-700/50 hover:border-slate-600 hover:bg-slate-800/60 transition-colors relative">
              {/* Individual delete button — top right, visible on hover */}
              {canDelete && (
                <button
                  onClick={() => deleteJob(job.job_id)}
                  className="absolute top-2 right-2 p-1 rounded-md opacity-0 group-hover:opacity-100 bg-slate-700/50 hover:bg-red-500/20 text-slate-500 hover:text-red-400 transition-all"
                  title="Remove from history"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}

              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 overflow-hidden pr-2">
                  <div className="mt-0.5">{getStatusIcon(job.status)}</div>
                  <div className="min-w-0">
                    <h4 className="text-slate-200 font-medium font-mono text-sm truncate" title={job.file_name}>
                      {job.file_name}
                    </h4>
                    <div className="flex items-center mt-1 text-xs text-slate-500 space-x-3">
                      <span>{new Date(job.created_at).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true })}</span>
                    </div>
                  </div>
                </div>
                <div className="pr-5">{getStatusBadge(job.status)}</div>
              </div>
              <PipelineProgress job={job} />
              <TerminalBox job={job} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
