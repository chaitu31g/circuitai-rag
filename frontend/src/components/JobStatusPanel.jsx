import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { API_BASE_URL } from "../config";
import {
  RefreshCw, CheckCircle, Clock, AlertCircle, X,
  ChevronDown, ChevronUp, Terminal, Cpu, Layers,
  HardDrive, Zap, FileText,
} from "lucide-react";

// ── Pipeline stage definitions ────────────────────────────────────────────────
const STAGES = [
  {
    key: "parsing",
    label: "PDF Parsing",
    icon: FileText,
    color: "text-blue-400",
    bg: "bg-blue-500/10 border-blue-500/30",
    dot: "bg-blue-400",
  },
  {
    key: "chunking",
    label: "Structure Chunking",
    icon: Layers,
    color: "text-indigo-400",
    bg: "bg-indigo-500/10 border-indigo-500/30",
    dot: "bg-indigo-400",
  },
  {
    key: "embedding",
    label: "BGE-M3 Embedding",
    icon: Cpu,
    color: "text-cyan-400",
    bg: "bg-cyan-500/10 border-cyan-500/30",
    dot: "bg-cyan-400",
  },
  {
    key: "storing",
    label: "ChromaDB Storage",
    icon: HardDrive,
    color: "text-emerald-400",
    bg: "bg-emerald-500/10 border-emerald-500/30",
    dot: "bg-emerald-400",
  },
];

// ── Stage tracker strip ──────────────────────────────────────────────────────
function StageStrip({ stagesDone = [], currentStage = null, status }) {
  return (
    <div className="flex items-center gap-1 mt-2 flex-wrap">
      {STAGES.map((s, i) => {
        const done    = stagesDone.includes(s.key);
        const active  = currentStage === s.key;
        const failed  = status === "failed" && !done && active;
        const Icon    = s.icon;

        return (
          <div key={s.key} className="flex items-center gap-1">
            {/* connector line */}
            {i > 0 && (
              <div className={`h-px w-4 ${done ? "bg-slate-500" : "bg-slate-700"}`} />
            )}

            <div
              className={`flex items-center gap-1 px-2 py-1 rounded-lg border text-[10px] font-medium transition-all
                ${done
                  ? "bg-slate-700/40 border-slate-600/30 text-slate-400"
                  : active
                    ? `${s.bg} ${s.color} animate-pulse`
                    : failed
                      ? "bg-red-500/10 border-red-500/30 text-red-400"
                      : "bg-slate-800/40 border-slate-700/30 text-slate-600"
                }`}
            >
              {done ? (
                <CheckCircle className="w-2.5 h-2.5 text-emerald-400" />
              ) : active ? (
                <RefreshCw className="w-2.5 h-2.5 animate-spin" />
              ) : failed ? (
                <AlertCircle className="w-2.5 h-2.5" />
              ) : (
                <Icon className="w-2.5 h-2.5" />
              )}
              {s.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Live terminal log panel ───────────────────────────────────────────────────
function TerminalLog({ logs = [], isLive = false }) {
  const containerRef  = useRef(null);   // ref to the scrollable div
  const pausedRef     = useRef(false);  // true when user has scrolled up

  // When the user scrolls, decide whether to pause auto-scroll
  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    // Within 40px of the bottom → resume; further up → pause
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    pausedRef.current = !atBottom;
  };

  // Auto-scroll only when NOT paused by the user
  useEffect(() => {
    if (pausedRef.current) return;
    const el = containerRef.current;
    if (el) {
      // Scroll the terminal container directly — never calls scrollIntoView
      // which would scroll the whole page.
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="mt-3 rounded-xl overflow-hidden border border-slate-700/50">
      {/* Terminal header bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/80 border-b border-slate-700/50">
        <div className="flex gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-green-500/70" />
        </div>
        <div className="flex items-center gap-1.5 ml-1">
          <Terminal className="w-3 h-3 text-slate-500" />
          <span className="text-[10px] text-slate-500 font-mono">pipeline log</span>
        </div>
        {isLive && (
          <span className="ml-auto flex items-center gap-1 text-[9px] text-emerald-400 font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            LIVE
          </span>
        )}
      </div>

      {/* Log lines */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="bg-slate-950/80 max-h-56 overflow-y-auto custom-scrollbar px-3 py-2 font-mono text-[10px] leading-relaxed"
      >
        {logs.length === 0 ? (
          <p className="text-slate-600">Waiting for pipeline output…</p>
        ) : (
          logs.map((line, i) => {
            const isError   = line.includes("❌") || line.includes("FAILED") || line.includes("Error");
            const isSuccess = line.includes("✅") || line.includes("complete ✔") || line.includes("Pipeline complete");
            const isHeader  = line.includes("━━━");
            const isStage   = line.startsWith("┌─") || line.startsWith("└─");
            const isDetail  = line.startsWith("│");

            return (
              <div
                key={i}
                className={
                  isError   ? "text-red-400" :
                  isSuccess ? "text-emerald-400" :
                  isHeader  ? "text-indigo-300 font-semibold mt-1" :
                  isStage   ? "text-cyan-300 mt-0.5" :
                  isDetail  ? "text-slate-400 pl-2" :
                  "text-slate-500"
                }
              >
                {line}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ── Single job card ───────────────────────────────────────────────────────────
function JobCard({ job, onDelete, onRetry }) {
  const [expanded, setExpanded] = useState(
    job.status === "processing" || job.status === "failed"
  );

  // Auto-expand when processing starts, collapse when done
  useEffect(() => {
    if (job.status === "processing") setExpanded(true);
  }, [job.status]);

  const isProcessing = job.status === "processing";
  const isDone       = job.status === "done";
  const isFailed     = job.status === "failed";
  const isPending    = job.status === "pending";

  const statusColor =
    isDone       ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/25" :
    isFailed     ? "text-red-400 bg-red-500/10 border-red-500/25" :
    isProcessing ? "text-yellow-400 bg-yellow-500/10 border-yellow-500/25" :
                   "text-slate-400 bg-slate-700/30 border-slate-600/30";

  const statusIcon =
    isDone       ? <CheckCircle className="w-3.5 h-3.5" /> :
    isFailed     ? <AlertCircle className="w-3.5 h-3.5" /> :
    isProcessing ? <RefreshCw   className="w-3.5 h-3.5 animate-spin" /> :
                   <Clock       className="w-3.5 h-3.5" />;

  const activeStage = STAGES.find(s => s.key === job.current_stage);

  return (
    <div className={`rounded-xl border mb-2 overflow-hidden transition-all
      ${isFailed ? "border-red-500/20 bg-red-500/5" :
        isDone   ? "border-slate-700/40 bg-slate-800/20" :
                   "border-slate-700/40 bg-slate-800/30"}`}
    >
      {/* Card header */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        {/* Status badge */}
        <span className={`flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-semibold flex-shrink-0 ${statusColor}`}>
          {statusIcon}
          {job.status}
        </span>

        {/* File name */}
        <span className="font-mono text-xs text-slate-200 truncate flex-1 min-w-0" title={job.file_name}>
          {job.file_name}
        </span>

        <div className="flex items-center gap-1 flex-shrink-0">
          {/* Processing time */}
          {isDone && job.processing_time_sec != null && (
            <span className="text-[10px] text-slate-500 font-mono flex items-center gap-0.5">
              <Zap className="w-2.5 h-2.5" />
              {job.processing_time_sec.toFixed(1)}s
            </span>
          )}

          {/* Chunk count */}
          {isDone && job.chunks_created > 0 && (
            <span className="text-[10px] text-emerald-500 font-mono flex items-center gap-0.5 ml-1">
              <Layers className="w-2.5 h-2.5" />
              {job.chunks_created}v
            </span>
          )}

          {/* Expand toggle */}
          <button
            onClick={() => setExpanded(v => !v)}
            className="p-1 rounded hover:bg-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors ml-1"
          >
            {expanded
              ? <ChevronUp   className="w-3.5 h-3.5" />
              : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {/* Delete */}
          {!isProcessing && (
            <button
              onClick={() => onDelete(job.job_id)}
              className="p-1 rounded hover:bg-red-500/20 text-slate-600 hover:text-red-400 transition-colors"
              title="Remove"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Active stage indicator */}
      {isProcessing && activeStage && (
        <div className={`mx-3 mb-2 px-2.5 py-1.5 rounded-lg border text-[11px] flex items-center gap-2 ${activeStage.bg} ${activeStage.color}`}>
          <RefreshCw className="w-3 h-3 animate-spin flex-shrink-0" />
          <span className="font-medium">Running: {activeStage.label}</span>
        </div>
      )}

      {/* Expandable body */}
      {expanded && (
        <div className="px-3 pb-3">
          {/* Stage tracker */}
          <StageStrip
            stagesDone={job.stages_done || []}
            currentStage={job.current_stage}
            status={job.status}
          />

          {/* Error message */}
          {isFailed && job.error_message && (
            <div className="mt-2 px-2.5 py-2 bg-red-500/10 border border-red-500/25 rounded-lg text-[11px] text-red-400 font-mono break-words">
              ❌ {job.error_message}
            </div>
          )}

          {/* Retry button */}
          {isFailed && (
            <button
              onClick={() => onRetry(job.job_id)}
              className="mt-2 text-[11px] bg-yellow-500/10 hover:bg-yellow-500/20 border border-yellow-500/30 text-yellow-300 px-3 py-1 rounded-lg transition-all font-medium"
            >
              ↺ Retry
            </button>
          )}

          {/* Terminal log */}
          <TerminalLog
            logs={job.logs || []}
            isLive={isProcessing}
          />
        </div>
      )}
    </div>
  );
}

// ── Main JobStatusPanel ───────────────────────────────────────────────────────
export default function JobStatusPanel({ onJobCompleted, onActiveStage }) {
  const [jobs, setJobs] = useState([]);
  const prevDoneRef = useRef(new Set());

  const fetchJobs = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/jobs`);
      const jobArray = Array.isArray(res.data) ? res.data : res.data.jobs || [];
      setJobs(jobArray);

      // Fire onJobCompleted only for newly-finished jobs
      const nowDone = new Set(jobArray.filter(j => j.status === "done").map(j => j.job_id));
      for (const jid of nowDone) {
        if (!prevDoneRef.current.has(jid)) onJobCompleted?.();
      }
      prevDoneRef.current = nowDone;

      // Propagate active pipeline stage upward
      const processingJob = jobArray.find(j => j.status === "processing");
      onActiveStage?.(processingJob?.current_stage ?? null);
    } catch (err) {
      console.error("Failed to fetch jobs", err);
    }
  };

  // Poll every 1.5s while any job is processing, 5s otherwise
  useEffect(() => {
    fetchJobs();
    const hasProcessing = () => jobs.some(j => j.status === "processing" || j.status === "pending");
    const id = setInterval(fetchJobs, hasProcessing() ? 1500 : 5000);
    return () => clearInterval(id);
  }, [jobs.some(j => j.status === "processing" || j.status === "pending")]);

  const deleteJob = async (jobId) => {
    try {
      await axios.delete(`${API_BASE_URL}/jobs/${jobId}`);
      setJobs(prev => prev.filter(j => j.job_id !== jobId));
    } catch (err) {
      console.error("Delete failed", err);
    }
  };

  const retryJob = async (jobId) => {
    try {
      await axios.post(`${API_BASE_URL}/jobs/${jobId}/retry`);
      fetchJobs();
    } catch (err) {
      console.error("Retry failed", err);
    }
  };

  const clearAll = async () => {
    try {
      await axios.delete(`${API_BASE_URL}/jobs`);
      setJobs(prev => prev.filter(j => j.status === "processing" || j.status === "pending"));
    } catch (err) {
      console.error("Clear failed", err);
    }
  };

  const activeCount    = jobs.filter(j => j.status === "processing" || j.status === "pending").length;
  const doneCount      = jobs.filter(j => j.status === "done").length;
  const failedCount    = jobs.filter(j => j.status === "failed").length;
  const clearableCount = doneCount + failedCount;

  return (
    <div className="glass-panel rounded-xl h-[680px] flex flex-col">

      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between bg-slate-800/20">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-slate-200">Pipeline History</h2>
          <div className="flex items-center gap-1.5 ml-1">
            {activeCount > 0 && (
              <span className="text-[10px] bg-yellow-500/15 text-yellow-400 border border-yellow-500/25 px-1.5 py-0.5 rounded-full font-medium">
                {activeCount} running
              </span>
            )}
            {doneCount > 0 && (
              <span className="text-[10px] bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 px-1.5 py-0.5 rounded-full font-medium">
                {doneCount} done
              </span>
            )}
            {failedCount > 0 && (
              <span className="text-[10px] bg-red-500/15 text-red-400 border border-red-500/25 px-1.5 py-0.5 rounded-full font-medium">
                {failedCount} failed
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1">
          {clearableCount > 0 && (
            <button
              onClick={clearAll}
              className="text-[10px] text-slate-500 hover:text-slate-300 px-2 py-1 rounded hover:bg-slate-700/50 transition-colors"
            >
              Clear finished
            </button>
          )}
          <button
            onClick={fetchJobs}
            className="p-1.5 rounded hover:bg-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Job list */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
        {jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Terminal className="w-10 h-10 text-slate-700 mb-3" />
            <p className="text-sm text-slate-500">No ingestion jobs yet</p>
            <p className="text-xs text-slate-600 mt-1">Upload a PDF to see the pipeline run here</p>
          </div>
        ) : (
          jobs.map(job => (
            <JobCard
              key={job.job_id}
              job={job}
              onDelete={deleteJob}
              onRetry={retryJob}
            />
          ))
        )}
      </div>
    </div>
  );
}