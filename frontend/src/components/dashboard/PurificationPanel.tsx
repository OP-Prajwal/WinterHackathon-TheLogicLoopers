import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, AlertOctagon, Loader2, Database, FileSpreadsheet, FileJson, FileText, Download, ChevronDown, Check } from 'lucide-react';
import clsx from 'clsx';

interface ScanResult {
    status: string;
    scan_id: string;
    total_rows: number;
    poison_count: number;
    safe_count: number;
    message: string;
}

const FORMATS = [
    { id: 'csv', label: 'CSV', icon: FileSpreadsheet },
    { id: 'json', label: 'JSON', icon: FileJson },
    { id: 'excel', label: 'Excel (XLSX)', icon: FileSpreadsheet },
    { id: 'parquet', label: 'Parquet', icon: Database },
];

export const PurificationPanel: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [scanning, setScanning] = useState(false);
    const [result, setResult] = useState<ScanResult | null>(null);
    const [exportFormat, setExportFormat] = useState('csv');
    const [isFormatOpen, setIsFormatOpen] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setResult(null);
        }
    };

    const handleScan = async () => {
        if (!file) return;

        setScanning(true);
        const formData = new FormData();
        formData.append('file', file);
        // Note: We no longer send format here.

        try {
            const response = await fetch('http://localhost:8003/api/dataset/scan', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            setResult(data);
        } catch (error) {
            console.error("Scan failed:", error);
            alert("Scan failed. Check console.");
        } finally {
            setScanning(false);
        }
    };

    const downloadDataset = async (type: 'safe' | 'poison') => {
        if (!result) return;

        try {
            const scan_id = result.scan_id;
            const format = exportFormat;

            // Trigger download via new endpoint
            const url = `http://localhost:8003/api/dataset/export?scan_id=${scan_id}&type=${type}&format=${format}`;

            // We can just open this URL, but strictly better to fetch blob if we want name control
            // Or just window.open(url) works for GET download file

            window.open(url, '_blank');

        } catch (error) {
            console.error('Download init failed:', error);
            alert('Download failed. Check console.');
        }
    };

    const SelectedIcon = FORMATS.find(f => f.id === exportFormat)?.icon || FileSpreadsheet;

    return (
        <div className="glass-panel p-6 relative flex flex-col h-full">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                    <Database size={20} className="text-cyan-400" />
                    <h3 className="text-lg font-bold text-gray-200">Dataset Purification</h3>
                </div>
            </div>

            <div className="flex-1 flex flex-col">
                <input
                    type="file"
                    accept=".csv, .json, .xlsx, .xls, .parquet, .txt"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                />

                {!file ? (
                    <motion.div
                        className="flex-1 border-2 border-dashed border-white/10 rounded-xl flex flex-col items-center justify-center p-6 cursor-pointer hover:border-cyan-500/50 hover:bg-cyan-500/5 transition-all group"
                        onClick={() => fileInputRef.current?.click()}
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                    >
                        <div className="p-4 bg-dark-900/50 rounded-full mb-3 group-hover:shadow-[0_0_20px_rgba(6,182,212,0.2)] transition-shadow">
                            <Upload size={32} className="text-gray-400 group-hover:text-cyan-400 transition-colors" />
                        </div>
                        <p className="text-gray-300 font-medium text-center">Click to upload dataset</p>
                        <div className="flex gap-2 mt-2">
                            <span className="px-1.5 py-0.5 bg-white/5 rounded text-[10px] text-gray-500">Multiformat Support</span>
                        </div>
                    </motion.div>
                ) : (
                    <div className="flex-1 flex flex-col">
                        <div className="p-4 bg-dark-900/50 rounded-xl border border-white/10 mb-4 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-emerald-500/10 rounded-lg">
                                    <FileText size={20} className="text-emerald-400" />
                                </div>
                                <div>
                                    <div className="text-sm font-bold text-gray-200">{file.name}</div>
                                    <div className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</div>
                                </div>
                            </div>
                            <button
                                onClick={() => { setFile(null); setResult(null); }}
                                className="text-xs text-red-400 hover:text-red-300 hover:underline"
                                disabled={scanning}
                            >
                                Change
                            </button>
                        </div>

                        {!result && (
                            <button
                                className={clsx(
                                    "w-full py-3 rounded-xl font-bold transition-all flex items-center justify-center gap-2",
                                    scanning
                                        ? "bg-dark-800 text-gray-500 cursor-not-allowed border border-white/5"
                                        : "bg-cyan-500 hover:bg-cyan-400 text-black shadow-[0_0_20px_rgba(6,182,212,0.4)]"
                                )}
                                onClick={handleScan}
                                disabled={scanning}
                            >
                                {scanning ? (
                                    <>
                                        <Loader2 className="animate-spin" size={18} /> Scanning Dataset...
                                    </>
                                ) : (
                                    <>
                                        Start Purification Scan
                                    </>
                                )}
                            </button>
                        )}
                    </div>
                )}

                <AnimatePresence>
                    {result && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="mt-4 space-y-4"
                        >
                            <div className="grid grid-cols-3 gap-2">
                                <div className="p-3 bg-dark-900/50 rounded-lg text-center border border-white/5">
                                    <div className="text-xs text-gray-500 uppercase">Rows</div>
                                    <div className="text-xl font-bold text-gray-200">{result.total_rows.toLocaleString()}</div>
                                </div>
                                <div className="p-3 bg-emerald-500/10 rounded-lg text-center border border-emerald-500/20">
                                    <div className="text-xs text-emerald-500/80 uppercase">Clean</div>
                                    <div className="text-xl font-bold text-emerald-400">{result.safe_count.toLocaleString()}</div>
                                </div>
                                <div className="p-3 bg-red-500/10 rounded-lg text-center border border-red-500/20 shadow-[0_0_15px_rgba(239,68,68,0.1)]">
                                    <div className="text-xs text-red-500/80 uppercase">Poison</div>
                                    <div className="text-xl font-bold text-red-400">{result.poison_count.toLocaleString()}</div>
                                </div>
                            </div>

                            {/* Format Selector inside Result Panel */}
                            <div className="relative z-20">
                                <label className="text-xs text-gray-500 uppercase mb-1 block">Download Format</label>
                                <button
                                    onClick={() => setIsFormatOpen(!isFormatOpen)}
                                    className="w-full px-3 py-2 bg-dark-800 border border-white/10 rounded-lg text-sm text-cyan-400 flex items-center justify-between hover:bg-dark-700 transition-colors"
                                >
                                    <span className="flex items-center gap-2">
                                        <SelectedIcon size={16} />
                                        {FORMATS.find(f => f.id === exportFormat)?.label}
                                    </span>
                                    <ChevronDown size={14} className={clsx("transition-transform", isFormatOpen && "rotate-180")} />
                                </button>

                                <AnimatePresence>
                                    {isFormatOpen && (
                                        <motion.div
                                            initial={{ opacity: 0, y: 5 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            exit={{ opacity: 0, y: 5 }}
                                            className="absolute left-0 right-0 top-full mt-2 bg-dark-900 border border-white/10 rounded-lg shadow-xl overflow-hidden z-50"
                                        >
                                            {FORMATS.map(f => (
                                                <button
                                                    key={f.id}
                                                    onClick={() => {
                                                        setExportFormat(f.id);
                                                        setIsFormatOpen(false);
                                                    }}
                                                    className={clsx(
                                                        "w-full text-left px-3 py-2 text-sm flex items-center gap-2 hover:bg-white/5 transition-colors",
                                                        exportFormat === f.id ? "text-cyan-400 bg-cyan-500/10" : "text-gray-400"
                                                    )}
                                                >
                                                    <f.icon size={16} />
                                                    {f.label}
                                                    {exportFormat === f.id && <Check size={14} className="ml-auto" />}
                                                </button>
                                            ))}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            <div className="space-y-2 pt-2">
                                <button
                                    className="w-full py-2 bg-dark-800 border border-white/10 hover:border-emerald-500/50 hover:text-emerald-400 text-gray-400 rounded-lg text-sm transition-all flex items-center justify-between px-4 group"
                                    onClick={() => downloadDataset('safe')}
                                >
                                    <div className="flex items-center gap-2">
                                        <Download size={16} />
                                        <span>Download Clean Data</span>
                                    </div>
                                    <span className="text-xs opacity-50">{exportFormat.toUpperCase()}</span>
                                </button>

                                {result.poison_count > 0 && (
                                    <button
                                        className="w-full py-2 bg-dark-800 border border-white/10 hover:border-red-500/50 hover:text-red-400 text-gray-400 rounded-lg text-sm transition-all flex items-center justify-between px-4 group"
                                        onClick={() => downloadDataset('poison')}
                                    >
                                        <div className="flex items-center gap-2">
                                            <AlertOctagon size={16} />
                                            <span>Download Poison Data</span>
                                        </div>
                                        <span className="text-xs opacity-50">{exportFormat.toUpperCase()}</span>
                                    </button>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};
