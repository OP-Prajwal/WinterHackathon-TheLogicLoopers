import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileSpreadsheet, Check, X, Database, ArrowRight, Zap } from 'lucide-react';
import { API_BASE } from '../../services/api';

interface DataImportProps {
    onDataLoaded?: (info: { filename: string; rows: number }) => void;
}

export const DataImport: React.FC<DataImportProps> = ({ onDataLoaded }) => {
    const [isDragOver, setIsDragOver] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
    const [uploadedFile, setUploadedFile] = useState<{ name: string; rows: number } | null>(null);
    const [errorMessage, setErrorMessage] = useState<string>('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
    }, []);

    const uploadFile = async (file: File) => {
        setUploadStatus('uploading');
        setErrorMessage('');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/api/stream/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.error) {
                setUploadStatus('error');
                setErrorMessage(result.error);
                return;
            }

            setUploadStatus('success');
            setUploadedFile({ name: result.filename, rows: result.rows });
            onDataLoaded?.({ filename: result.filename, rows: result.rows });
        } catch (err) {
            setUploadStatus('error');
            setErrorMessage('Failed to upload file. Please try again.');
        }
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            uploadFile(files[0]);
        }
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const resetUpload = () => {
        setUploadStatus('idle');
        setUploadedFile(null);
        setErrorMessage('');
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6 mb-6"
        >
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30">
                        <Database size={20} className="text-cyan-400" />
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-white">Import Your Data</h3>
                        <p className="text-sm text-gray-400">Upload CSV, JSON, Excel or Parquet files for real-time prediction</p>
                    </div>
                </div>
                {uploadedFile && (
                    <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30"
                    >
                        <Check size={16} className="text-cyan-400" />
                        <span className="text-cyan-400 font-mono text-sm">
                            {uploadedFile.name} ({uploadedFile.rows} rows)
                        </span>
                        <button
                            onClick={resetUpload}
                            className="ml-2 p-1 hover:bg-red-500/20 rounded transition-colors"
                        >
                            <X size={14} className="text-red-400" />
                        </button>
                    </motion.div>
                )}
            </div>

            <div
                onClick={handleClick}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`
                    relative cursor-pointer rounded-xl border-2 border-dashed p-8
                    transition-all duration-300 flex flex-col items-center justify-center gap-4
                    ${isDragOver
                        ? 'border-cyan-400 bg-cyan-500/10 scale-[1.02]'
                        : 'border-gray-700 hover:border-cyan-500/50 bg-dark-900/50'}
                    ${uploadStatus === 'uploading' ? 'pointer-events-none opacity-60' : ''}
                `}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.json,.xlsx,.xls,.parquet,.txt"
                    onChange={handleFileSelect}
                    className="hidden"
                />

                <AnimatePresence mode="wait">
                    {uploadStatus === 'idle' && (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center gap-3"
                        >
                            <div className="p-4 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-500/30">
                                <Upload size={28} className="text-cyan-400" />
                            </div>
                            <div className="text-center">
                                <p className="text-gray-300 font-medium">
                                    Drag & drop your data file here
                                </p>
                                <p className="text-gray-500 text-sm mt-1">
                                    or click to browse • CSV, JSON, Excel, Parquet supported
                                </p>
                            </div>
                        </motion.div>
                    )}

                    {uploadStatus === 'uploading' && (
                        <motion.div
                            key="uploading"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center gap-3"
                        >
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                                className="p-4 rounded-full bg-cyan-500/20 border border-cyan-500/50"
                            >
                                <Zap size={28} className="text-cyan-400" />
                            </motion.div>
                            <p className="text-cyan-400 font-medium">Processing your data...</p>
                        </motion.div>
                    )}

                    {uploadStatus === 'success' && (
                        <motion.div
                            key="success"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center gap-3"
                        >
                            <div className="p-4 rounded-full bg-green-500/20 border border-green-500/50">
                                <Check size={28} className="text-green-400" />
                            </div>
                            <p className="text-green-400 font-medium">Data loaded successfully!</p>
                            <p className="text-gray-400 text-sm">
                                Click "Start Monitoring" to run predictions
                            </p>
                        </motion.div>
                    )}

                    {uploadStatus === 'error' && (
                        <motion.div
                            key="error"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center gap-3"
                        >
                            <div className="p-4 rounded-full bg-red-500/20 border border-red-500/50">
                                <X size={28} className="text-red-400" />
                            </div>
                            <p className="text-red-400 font-medium">Upload failed</p>
                            <p className="text-gray-500 text-sm">{errorMessage}</p>
                            <button
                                onClick={(e) => { e.stopPropagation(); resetUpload(); }}
                                className="mt-2 px-4 py-1 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors text-sm"
                            >
                                Try Again
                            </button>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Decorative corners */}
                <div className="absolute top-2 left-2 w-4 h-4 border-l-2 border-t-2 border-cyan-500/30 rounded-tl" />
                <div className="absolute top-2 right-2 w-4 h-4 border-r-2 border-t-2 border-cyan-500/30 rounded-tr" />
                <div className="absolute bottom-2 left-2 w-4 h-4 border-l-2 border-b-2 border-cyan-500/30 rounded-bl" />
                <div className="absolute bottom-2 right-2 w-4 h-4 border-r-2 border-b-2 border-cyan-500/30 rounded-br" />
            </div>

            {/* Quick Info */}
            <div className="mt-4 flex items-center justify-center gap-6 text-xs text-gray-500">
                <div className="flex items-center gap-1">
                    <FileSpreadsheet size={12} />
                    <span>Max 100MB</span>
                </div>
                <div className="flex items-center gap-1">
                    <ArrowRight size={12} />
                    <span>Real-time streaming</span>
                </div>
                <div className="flex items-center gap-1">
                    <Zap size={12} />
                    <span>Instant prediction</span>
                </div>
            </div>
        </motion.div>
    );
};
