import React, { useState } from 'react';
import { Database, Plus, Search, FileText, MoreVertical } from 'lucide-react';
import { motion } from 'framer-motion';

interface Dataset {
    id: string;
    name: string;
    type: string;
    size: string;
    status: 'Active' | 'Ready' | 'Processing';
    lastModified: string;
}

export const Datasets: React.FC = () => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [datasets] = useState<Dataset[]>([
        {
            id: '1',
            name: 'Diabetes Binary (BRFSS)',
            type: 'Tabular (CSV)',
            size: '2.4 MB',
            status: 'Active',
            lastModified: '2 hours ago'
        },
        // Placeholder for future dynamic additions
    ]);

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Database className="w-8 h-8 text-cyan-400" />
                        My Datasets
                    </h1>
                    <p className="text-gray-400 text-sm mt-1">Manage your training and validation datasets</p>
                </div>
            </div>

            {/* Controls Bar */}
            <div className="flex gap-4 mb-6">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                    <input
                        type="text"
                        placeholder="Search datasets..."
                        className="w-full bg-dark-800 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-white focus:outline-none focus:border-cyan-500/50 transition-colors"
                    />
                </div>
            </div>

            {/* Datasets Grid */}
            <div className="grid grid-cols-1 gap-4">
                {datasets.map((dataset) => (
                    <motion.div
                        key={dataset.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-dark-800/40 backdrop-blur-xl border border-white/10 rounded-xl p-5 hover:border-cyan-500/30 transition-all group"
                    >
                        <div className="flex items-start justify-between">
                            <div className="flex items-start gap-4">
                                <div className="p-3 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                                    <FileText size={24} />
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-white group-hover:text-cyan-300 transition-colors">{dataset.name}</h3>
                                    <div className="flex items-center gap-3 text-sm text-gray-400 mt-1">
                                        <span className="bg-white/5 px-2 py-0.5 rounded text-xs border border-white/5">{dataset.type}</span>
                                        <span>{dataset.size}</span>
                                        <span>•</span>
                                        <span>Updated {dataset.lastModified}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center gap-4">
                                <div className={`px-3 py-1 rounded-full text-xs font-semibold border ${dataset.status === 'Active'
                                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                    : 'bg-gray-700 text-gray-400 border-gray-600'
                                    }`}>
                                    {dataset.status === 'Active' && <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-2 animate-pulse" />}
                                    {dataset.status}
                                </div>

                                <button className="p-2 text-gray-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                                    <MoreVertical size={18} />
                                </button>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>

            <div className="p-8 border-2 border-dashed border-white/5 rounded-xl flex flex-col items-center justify-center text-gray-500 hover:border-cyan-500/20 hover:bg-white/[0.02] transition-all cursor-pointer group">
                <div className="w-16 h-16 rounded-full bg-dark-800 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <Plus size={32} className="text-gray-400 group-hover:text-cyan-400 transition-colors" />
                </div>
                <h3 className="font-semibold text-gray-300 text-lg">Import New Dataset</h3>
                <p className="text-sm mt-1 text-gray-400">Drag & drop your CSV or Parquet files here</p>
                <p className="text-xs mt-2 text-cyan-500/70 border border-cyan-500/20 bg-cyan-500/5 px-2 py-1 rounded">
                    Optimized for LogicLoopers Format
                </p>
            </div>
        </div>
    );
};
