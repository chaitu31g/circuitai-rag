import { useState, useCallback } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './config';
import { Database, UploadCloud, MessageSquare } from 'lucide-react';
import FileUploader from './components/FileUploader';
import JobStatusPanel from './components/JobStatusPanel';
import LibraryPanel from './components/LibraryPanel';
import ChatPanel from './components/ChatPanel';

function App() {
  const [activeJobs, setActiveJobs]               = useState([]);
  const [uploadError, setUploadError]             = useState(null);
  const [libraryRefreshKey, setLibraryRefreshKey] = useState(0);
  const [libraryComponents, setLibraryComponents] = useState([]);
  const [activeTab, setActiveTab]                 = useState('ingest'); // 'ingest' | 'chat'
  const [activeStage, setActiveStage]             = useState(null);     // 'parsing' | 'chunking' | 'embedding' | 'storing' | null

  const handleJobCompleted = useCallback(() => {
    setLibraryRefreshKey(k => k + 1);
  }, []);

  const handleUploadStart = async (files) => {
    setUploadError(null);
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const newJobIds = response.data.jobs.map(job => job.job_id);
      setActiveJobs(prev => [...prev, ...newJobIds]);
    } catch (error) {
      console.error('Upload failed', error);
      setUploadError(error.response?.data?.detail || 'An unexpected error occurred during upload.');
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans selection:bg-blue-500/30">
      {/* Background Gradients */}
      <div className="absolute top-0 inset-x-0 h-96 bg-gradient-to-b from-blue-900/20 to-transparent pointer-events-none" />
      <div className="absolute top-20 left-1/4 w-[40rem] h-[30rem] bg-indigo-500/10 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute top-40 right-1/4 w-[30rem] h-[30rem] bg-blue-600/10 blur-[100px] rounded-full pointer-events-none" />

      <div className="max-w-6xl mx-auto relative z-10 pt-10">
        {/* Tab switcher (Header replaced by Tabs) */}
        <div className="flex justify-center mb-8">
          <div className="glass-panel rounded-xl p-1 flex gap-1">
            <button
              onClick={() => setActiveTab('ingest')}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'ingest'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <UploadCloud className="w-4 h-4" />
              Ingest
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'chat'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Chat with AI
            </button>
          </div>
        </div>

        {/* ── INGEST TAB ───────────────────────────────────────────────────── */}
        {activeTab === 'ingest' && (
          <main className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            {/* Left column */}
            <div className="lg:col-span-2 flex flex-col space-y-6">
              <h2 className="text-2xl font-semibold mb-2 text-indigo-300 flex items-center">
                <UploadCloud className="w-6 h-6 mr-3 text-indigo-400" />
                Upload Datasheets
              </h2>

              <FileUploader onUploadStart={handleUploadStart} />

              {uploadError && (
                <div className="bg-red-500/10 border-l-4 border-red-500 p-4 rounded-r-lg">
                  <p className="text-red-400 font-medium text-sm">{uploadError}</p>
                </div>
              )}

              {/* Pipeline Architecture */}
              <div className="glass-panel p-5 rounded-xl border border-blue-500/20 bg-blue-900/10 relative overflow-hidden group">
                <div className="absolute top-[-20%] left-[-10%] w-40 h-40 bg-blue-500/20 blur-[50px] rounded-full group-hover:bg-blue-400/30 transition-colors" />
                <h3 className="text-blue-300 font-semibold mb-2 relative z-10 flex items-center">
                  <Database className="w-5 h-5 mr-2" /> Pipeline Architecture
                </h3>
                <ul className="text-sm text-slate-400 space-y-2 font-medium relative z-10">
                  <li className={`flex items-center transition-colors ${activeStage === 'parsing' ? 'text-blue-300' : 'text-slate-300'}`}>
                    <span className="relative flex w-1.5 h-1.5 mr-2">
                      {activeStage === 'parsing' && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>}
                      <span className="relative inline-flex rounded-full w-1.5 h-1.5 bg-blue-500"></span>
                    </span>
                    Docling PDF Parsing
                  </li>
                  <li className={`flex items-center transition-colors ${activeStage === 'chunking' ? 'text-indigo-300' : 'text-slate-300'}`}>
                    <span className="relative flex w-1.5 h-1.5 mr-2">
                      {activeStage === 'chunking' && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>}
                      <span className="relative inline-flex rounded-full w-1.5 h-1.5 bg-indigo-400"></span>
                    </span>
                    Structure-Aware Chunking
                  </li>
                  <li className={`flex items-center transition-colors ${activeStage === 'embedding' ? 'text-cyan-300' : 'text-slate-300'}`}>
                    <span className="relative flex w-1.5 h-1.5 mr-2">
                      {activeStage === 'embedding' && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>}
                      <span className="relative inline-flex rounded-full w-1.5 h-1.5 bg-cyan-400"></span>
                    </span>
                    BGE-M3 Vector Embeddings
                  </li>
                  <li className={`flex items-center transition-colors ${activeStage === 'storing' ? 'text-emerald-300' : 'text-slate-300'}`}>
                    <span className="relative flex w-1.5 h-1.5 mr-2">
                      {activeStage === 'storing' && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>}
                      <span className="relative inline-flex rounded-full w-1.5 h-1.5 bg-emerald-400"></span>
                    </span>
                    ChromaDB Storage
                  </li>
                </ul>
              </div>

              {/* Knowledge Base */}
              <LibraryPanel
                refreshTrigger={libraryRefreshKey}
                onComponentsLoaded={setLibraryComponents}
              />
            </div>

            {/* Right column */}
            <div className="lg:col-span-3">
              <JobStatusPanel
                activeJobsIds={activeJobs}
                onJobCompleted={handleJobCompleted}
                onActiveStage={setActiveStage}
              />
            </div>
          </main>
        )}

        {/* ── CHAT TAB ─────────────────────────────────────────────────────── */}
        {activeTab === 'chat' && (
          <div className="max-w-3xl mx-auto">
            <ChatPanel components={libraryComponents} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
