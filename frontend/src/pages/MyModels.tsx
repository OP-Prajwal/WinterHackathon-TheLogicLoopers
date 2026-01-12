import React, { useState, useEffect } from 'react';
import { Database, Search, FileText, Trash2, Brain } from 'lucide-react';
import { motion } from 'framer-motion';

interface Dataset {
    id: string;
    name: string;
    type: string;
    size: string;
    status: 'Active' | 'Ready' | 'Processing';
    lastModified: string;
}

export const MyModels: React.FC = () => {
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchDatasets = async () => {
        try {
            const token = localStorage.getItem('token');
            const res = await fetch('http://localhost:8000/api/datasets', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                // Filter to only show trained models
                setDatasets(data.filter((d: Dataset) => d.type === 'trained_model'));
            }
        } catch (error) {
            console.error("Failed to fetch datasets", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDatasets();
    }, []);

    // const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    //     const file = event.target.files?.[0];
    //     if (!file) return;

    //     const formData = new FormData();
    //     formData.append('file', file);

    //     try {
    //         const token = localStorage.getItem('token');
    //         const res = await fetch('http://localhost:8000/api/datasets/upload', {
    //             method: 'POST',
    //             headers: { 'Authorization': `Bearer ${token}` },
    //             body: formData
    //         });
    //         if (res.ok) {
    //             fetchDatasets(); // Refresh list
    //         }
    //     } catch (error) {
    //         console.error("Upload failed", error);
    //     }

    //     // Reset input
    //     if (fileInputRef.current) fileInputRef.current.value = '';
    // };

    const handleDelete = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm('Are you sure you want to delete this dataset?')) return;

        try {
            const token = localStorage.getItem('token');
            const res = await fetch(`http://localhost:8000/api/datasets/${id}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                setDatasets(prev => prev.filter(d => d.id !== id));
            }
        } catch (error) {
            console.error("Delete failed", error);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Database className="w-8 h-8 text-cyan-400" />
                        My Models
                    </h1>
                    <p className="text-gray-400 text-sm mt-1">Manage your trained model checkpoints and data</p>
                </div>
            </div>

            {/* Controls Bar - Optional if list grows long, currently keeping for search */}
            <div className="flex gap-4 mb-6">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                    <input
                        type="text"
                        placeholder="Search models..."
                        className="w-full bg-dark-800 border border-white/10 rounded-lg pl-10 pr-4 py-2 text-white focus:outline-none focus:border-cyan-500/50 transition-colors"
                    />
                </div>
            </div>

            {/* Datasets Grid */}
            <div className="grid grid-cols-1 gap-4">
                {loading ? (
                    <div className="text-gray-400 text-center py-8">Loading datasets...</div>
                ) : datasets.length === 0 ? (
                    <div className="text-gray-500 text-center py-8">
                        No trained models yet. Start a personal training session to generate models.
                    </div>
                ) : (
                    datasets.map((dataset) => (
                        <motion.div
                            key={dataset.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-dark-800/40 backdrop-blur-xl border border-white/10 rounded-xl p-5 hover:border-cyan-500/30 transition-all group"
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex items-start gap-4">
                                    <div className={`p-3 rounded-lg border ${dataset.type === 'trained_model'
                                        ? 'bg-violet-500/10 border-violet-500/20 text-violet-400'
                                        : 'bg-cyan-500/10 border-cyan-500/20 text-cyan-400'
                                        }`}>
                                        {dataset.type === 'trained_model' ? <Brain size={24} /> : <FileText size={24} />}
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-white group-hover:text-cyan-300 transition-colors">{dataset.name}</h3>
                                        <div className="flex items-center gap-3 text-sm text-gray-400 mt-1">
                                            <span className={`px-2 py-0.5 rounded text-xs border ${dataset.type === 'trained_model'
                                                ? 'bg-violet-500/10 border-violet-500/20 text-violet-300'
                                                : 'bg-white/5 border-white/5'
                                                }`}>
                                                {dataset.type === 'trained_model' ? 'AI Model' : dataset.type}
                                            </span>
                                            <span>{dataset.size}</span>
                                            <span>•</span>
                                            <span>Updated {new Date(dataset.lastModified).toLocaleDateString()}</span>
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

                                    <button
                                        onClick={(e) => handleDelete(dataset.id, e)}
                                        className="p-2 text-gray-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors"
                                        title="Delete Dataset"
                                    >
                                        <Trash2 size={18} />
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    ))
                )}
            </div>

            {/* Manual upload removed as models are generated from training */}
            {/* <div
                onClick={() => fileInputRef.current?.click()}
                className="p-8 border-2 border-dashed border-white/5 rounded-xl flex flex-col items-center justify-center text-gray-500 hover:border-cyan-500/20 hover:bg-white/[0.02] transition-all cursor-pointer group"
            >
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    className="hidden"
                    accept=".csv,.parquet,.pt,.pth"
                />
                <div className="w-16 h-16 rounded-full bg-dark-800 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <Plus size={32} className="text-gray-400 group-hover:text-cyan-400 transition-colors" />
                </div>
                <h3 className="font-semibold text-gray-300 text-lg">Import Data or Model</h3>
                <p className="text-sm mt-1 text-gray-400">Click to upload CSV, Parquet, or PT files</p>
                <p className="text-xs mt-2 text-cyan-500/70 border border-cyan-500/20 bg-cyan-500/5 px-2 py-1 rounded">
                    Optimized for LogicLoopers Format
                </p>
            </div> */}
        </div>
    );
};
