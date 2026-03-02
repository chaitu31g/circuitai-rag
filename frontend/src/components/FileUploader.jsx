import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileType, AlertCircle } from 'lucide-react';

export default function FileUploader({ onUploadStart }) {
  const [files, setFiles] = useState([]);
  const [error, setError] = useState(null);

  const MAX_PDFS = 10;

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setError(null);

    // 1. Handle react-dropzone rejections first
    if (rejectedFiles.length > 0) {
      const tooManyFiles = rejectedFiles.some(r => r.errors.some(e => e.code === 'too-many-files'));
      
      if (tooManyFiles) {
        setError(`You can only upload a maximum of ${MAX_PDFS} PDFs at once.`);
      } else {
        // Collect specific errors like file-too-large or invalid type
        const errMessages = [];
        rejectedFiles.forEach(({ file, errors }) => {
          errors.forEach(err => {
            if (err.code === 'file-too-large') {
              errMessages.push(`"${file.name}" exceeds the 50MB limit.`);
            } else if (err.code === 'file-invalid-type') {
              errMessages.push(`"${file.name}" is not a PDF.`);
            }
          });
        });
        
        if (errMessages.length > 0) {
          setError(errMessages.join(' '));
        } else {
          setError("Some files were rejected. Please ensure they are PDFs under 50MB.");
        }
      }
    }

    // 2. Append unique files
    setFiles((prev) => {
      const newFiles = [...prev];
      let added = 0;
      acceptedFiles.forEach(file => {
        if (!newFiles.find(f => f.name === file.name)) {
          newFiles.push(file);
          added++;
        }
      });
      
      // 3. Enforce limit on total accumulated files
      if (newFiles.length > MAX_PDFS) {
        setError(`You can add up to ${MAX_PDFS} PDFs at once. Removed extra files.`);
        return newFiles.slice(0, MAX_PDFS);
      }
      return newFiles;
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxSize: 50 * 1024 * 1024, // 50MB per file
    maxFiles: MAX_PDFS,
  });

  const removeFile = (name) => {
    setFiles(files.filter(f => f.name !== name));
  };

  const handleUpload = () => {
    if (files.length > 0) {
      onUploadStart(files);
      setFiles([]); // Clear after starting upload
    }
  };

  return (
    <div className="flex flex-col space-y-4">
      {/* Dropzone */}
      <div 
        {...getRootProps()} 
        className={`glass-panel rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all border-2 border-dashed
          ${isDragActive ? 'border-primary-500 bg-primary-500/10 scale-[1.02]' : 'border-slate-600/50 hover:border-primary-500/50'}`}
      >
        <input {...getInputProps()} />
        <UploadCloud className={`mb-4 w-12 h-12 transition-colors ${isDragActive ? 'text-primary-500' : 'text-slate-400'}`} />
        
        {isDragActive ? (
            <p className="text-xl font-medium text-primary-400">Drop datasheets here...</p>
        ) : (
            <>
                <p className="text-lg text-slate-200 mt-2 font-medium">Drag & drop PDF files here</p>
                <p className="text-sm text-slate-400 mt-1">Max PDF size allowed: 50MB · Up to 10 PDFs at once</p>
            </>
        )}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 flex items-center text-red-400 text-sm">
           <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
           {error}
        </div>
      )}

      {/* Selected Files Preview */}
      {files.length > 0 && (
        <div className="glass-panel rounded-xl p-4 max-h-48 overflow-y-auto custom-scrollbar">
          <ul className="space-y-2">
            {files.map((file) => (
              <li key={file.name} className="flex items-center justify-between p-2 rounded hover:bg-white/5 transition-colors">
                 <div className="flex items-center truncate">
                   <FileType className="w-5 h-5 text-indigo-400 mr-3 flex-shrink-0" />
                   <span className="text-sm font-mono text-slate-300 truncate">{file.name}</span>
                   <span className="text-xs text-slate-500 ml-2">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                 </div>
                 <button 
                  onClick={(e) => { e.stopPropagation(); removeFile(file.name); }}
                  className="text-slate-500 hover:text-red-400 p-1"
                 >
                   &times;
                 </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Upload Action */}
      <div className="flex justify-end pt-2">
        <button 
          onClick={handleUpload}
          disabled={files.length === 0 || files.length > MAX_PDFS}
          className={`font-semibold py-2.5 px-6 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-primary-500/50 flex items-center
            ${files.length > 0 
                ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20' 
                : 'bg-slate-800 text-slate-500 cursor-not-allowed'}`}
        >
          <UploadCloud className="w-4 h-4 mr-2" />
          Ingest {files.length > 0 ? files.length : ''} Datasheet{files.length > 1 ? 's' : ''}
        </button>
      </div>
    </div>
  );
}
