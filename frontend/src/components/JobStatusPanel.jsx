import { useEffect, useState } from "react";
import axios from "axios";
import { API_BASE_URL } from "../config";
import {
  RefreshCw,
  CheckCircle,
  Clock,
  AlertCircle,
  X,
} from "lucide-react";

export default function JobStatusPanel({ onJobCompleted, onActiveStage }) {
  const [jobs, setJobs] = useState([]);

  const fetchJobs = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/jobs`);

      const jobArray = Array.isArray(res.data)
        ? res.data
        : res.data.jobs || [];

      setJobs(jobArray);

      const processingJob = jobArray.find((j) => j.status === "processing");

      if (processingJob?.current_stage) {
        onActiveStage?.(processingJob.current_stage);
      } else {
        onActiveStage?.(null);
      }

      jobArray.forEach((job) => {
        if (job.status === "done") onJobCompleted?.();
      });
    } catch (err) {
      console.error("Failed to fetch jobs", err);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 2000);
    return () => clearInterval(interval);
  }, []);

  const getIcon = (status) => {
    if (status === "done")
      return <CheckCircle className="text-emerald-400 w-5 h-5" />;

    if (status === "processing")
      return <RefreshCw className="animate-spin text-yellow-400 w-5 h-5" />;

    if (status === "failed")
      return <AlertCircle className="text-red-400 w-5 h-5" />;

    return <Clock className="text-slate-400 w-5 h-5" />;
  };

  const deleteJob = async (jobId) => {
    try {
      await axios.delete(`${API_BASE_URL}/jobs/${jobId}`);
      setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
    } catch (err) {
      console.error("delete failed", err);
    }
  };

  return (
    <div className="glass-panel rounded-xl h-[680px] flex flex-col">

      <div className="p-4 border-b border-slate-700 flex justify-between">
        <h2 className="text-lg font-semibold text-slate-200">
          Pipeline History
        </h2>

        <button
          onClick={fetchJobs}
          className="p-1 rounded hover:bg-slate-700"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">

        {jobs.length === 0 && (
          <div className="text-center text-slate-500 p-10">
            No ingestion jobs yet
          </div>
        )}

        {jobs.map((job) => (
          <div
            key={job.job_id}
            className="p-4 mb-2 rounded-lg border border-slate-700 bg-slate-800/30"
          >
            <div className="flex justify-between items-center">

              <div className="flex items-center gap-2">
                {getIcon(job.status)}
                <span className="font-mono text-sm text-slate-200">
                  {job.file_name}
                </span>
              </div>

              {job.status !== "processing" && (
                <button
                  onClick={() => deleteJob(job.job_id)}
                  className="text-slate-500 hover:text-red-400"
                >
                  <X size={14} />
                </button>
              )}

            </div>

            <div className="text-xs text-slate-400 mt-2">
              Status: {job.status}
            </div>

          </div>
        ))}

      </div>
    </div>
  );
}