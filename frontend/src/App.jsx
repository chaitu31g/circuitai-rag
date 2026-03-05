import { useState, useCallback } from "react";
import axios from "axios";
import { API_BASE_URL } from "./config";
import { Database, UploadCloud, MessageSquare } from "lucide-react";

import FileUploader from "./components/FileUploader";
import JobStatusPanel from "./components/JobStatusPanel";
import LibraryPanel from "./components/LibraryPanel";
import ChatPanel from "./components/ChatPanel";

// Set axios base URL defaults are handled per-call via API_BASE_URL from config.js


function App() {
  const [uploadError, setUploadError] = useState(null);
  const [libraryRefreshKey, setLibraryRefreshKey] = useState(0);
  const [libraryComponents, setLibraryComponents] = useState([]);
  const [activeTab, setActiveTab] = useState("ingest");
  const [activeStage, setActiveStage] = useState(null);

  const handleJobCompleted = useCallback(() => {
    setLibraryRefreshKey((k) => k + 1);
  }, []);

  const handleUploadStart = async (files) => {
    setUploadError(null);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
    } catch (error) {
      console.error("Upload failed", error);
      setUploadError(
        error.response?.data?.detail ||
          "An unexpected error occurred during upload."
      );
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
      <div className="max-w-6xl mx-auto">

        {/* Tabs */}
        <div className="flex justify-center mb-8">
          <div className="glass-panel rounded-xl p-1 flex gap-1">

            <button
              onClick={() => setActiveTab("ingest")}
              className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold ${
                activeTab === "ingest"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400"
              }`}
            >
              <UploadCloud className="w-4 h-4" />
              Ingest
            </button>

            <button
              onClick={() => setActiveTab("chat")}
              className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold ${
                activeTab === "chat"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400"
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Chat with AI
            </button>

          </div>
        </div>

        {/* INGEST TAB */}
        {activeTab === "ingest" && (
          <main className="grid grid-cols-1 lg:grid-cols-5 gap-8">

            {/* LEFT SIDE */}
            <div className="lg:col-span-2 flex flex-col space-y-6">

              <h2 className="text-2xl font-semibold text-indigo-300 flex items-center">
                <UploadCloud className="w-6 h-6 mr-3 text-indigo-400" />
                Upload Datasheets
              </h2>

              <FileUploader onUploadStart={handleUploadStart} />

              {uploadError && (
                <div className="bg-red-500/10 border-l-4 border-red-500 p-4 rounded">
                  <p className="text-red-400 text-sm">{uploadError}</p>
                </div>
              )}

              {/* Pipeline Architecture */}
              <div className="glass-panel p-5 rounded-xl border border-blue-500/20">
                <h3 className="text-blue-300 font-semibold mb-2 flex items-center">
                  <Database className="w-5 h-5 mr-2" />
                  Pipeline Architecture
                </h3>

                <ul className="text-sm text-slate-400 space-y-2">
                  <li className={activeStage === "parsing" ? "text-blue-300" : ""}>
                    Docling PDF Parsing
                  </li>
                  <li className={activeStage === "chunking" ? "text-indigo-300" : ""}>
                    Structure-Aware Chunking
                  </li>
                  <li className={activeStage === "embedding" ? "text-cyan-300" : ""}>
                    BGE-M3 Embeddings
                  </li>
                  <li className={activeStage === "storing" ? "text-emerald-300" : ""}>
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

            {/* RIGHT SIDE */}
            <div className="lg:col-span-3">

              <JobStatusPanel
                onJobCompleted={handleJobCompleted}
                onActiveStage={setActiveStage}
              />

            </div>

          </main>
        )}

        {/* CHAT TAB */}
        {activeTab === "chat" && (
          <div className="max-w-3xl mx-auto">
            <ChatPanel components={libraryComponents} />
          </div>
        )}

      </div>
    </div>
  );
}

export default App;