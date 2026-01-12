import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Download, Check, AlertOctagon, FileSpreadsheet, FileJson, Database, ChevronDown, X } from 'lucide-react';
import { API_BASE } from '../../services/api';
import clsx from 'clsx';

interface CompletionData {
    scan_id: string;
    clean_count: number;
    poison_count: number;
    message: string;
}

interface ScanResultPanelProps {
    result: CompletionData;
    onClear: () => void;
}

const FORMATS = [
    { id: 'csv', label: 'CSV', icon: FileSpreadsheet },
    { id: 'json', label: 'JSON', icon: FileJson },
    { id: 'excel', label: 'Excel (XLSX)', icon: FileSpreadsheet },
    { id: 'parquet', label: 'Parquet', icon: Database },
];

export const ScanResultPanel: React.FC<ScanResultPanelProps> = ({ result, onClear }) => {
    const [exportFormat, setExportFormat] = useState('csv');
    const [isFormatOpen, setIsFormatOpen] = useState(false);

    const downloadDataset = async (type: 'safe' | 'poison') => {
        try {
            const url = `${API_BASE}/api/dataset/export?scan_id=${result.scan_id}&type=${type}&format=${exportFormat}`;
            window.open(url, '_blank');
        } catch (error) {
            console.error('Download init failed:', error);
            alert('Download failed. Check console.');
        }
    };

    const SelectedIcon = FORMATS.find(f => f.id === exportFormat)?.icon || FileSpreadsheet;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="w-full glass-panel p-6 border-l-4 border-l-cyan-500 relative"
        >
            <button
                onClick={onClear}
                className="absolute top-4 right-4 text-gray-500 hover:text-white transition-colors"
            >
                <X size={20} />
            </button>

            <div className="flex flex-col md:flex-row gap-6 items-start md:items-center">
                <div className="flex-1">
                    <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                        <Check className="text-emerald-400" />
                        Scan Complete
                    </h3>
                    <p className="text-gray-400 text-sm mb-4">
                        Analysis finished. {result.message}
                    </p>

                    <div className="flex gap-4">
                        <div className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                            <div className="text-xs text-emerald-500 uppercase tracking-wider">Clean Data</div>
                            <div className="text-xl font-bold text-emerald-400">{result.clean_count.toLocaleString()}</div>
                        </div>
                        {result.poison_count > 0 && (
                            <div className="px-4 py-2 bg-red-500/10 border border-red-500/20 rounded-lg">
                                <div className="text-xs text-red-500 uppercase tracking-wider">Poison Data</div>
                                <div className="text-xl font-bold text-red-400">{result.poison_count.toLocaleString()}</div>
                            </div>
                        )}
                    </div>
                </div>

                <div className="flex flex-col gap-3 w-full md:w-auto min-w-[250px]">
                    {/* Format Selector */}
                    <div className="relative">
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
                                    className="absolute left-0 right-0 top-full mt-2 bg-dark-900 border border-white/10 rounded-lg shadow-xl overflow-hidden z-20"
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
                                        </button>
                                    ))}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    <button
                        onClick={() => downloadDataset('safe')}
                        className="flex items-center justify-center gap-2 px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-lg transition-colors"
                    >
                        <Download size={18} /> Download Clean
                    </button>

                    {result.poison_count > 0 && (
                        <button
                            onClick={() => downloadDataset('poison')}
                            className="flex items-center justify-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30 font-bold rounded-lg transition-colors"
                        >
                            <AlertOctagon size={18} /> Download Poison
                        </button>
                    )}
                </div>
            </div>
        </motion.div>
    );
};
