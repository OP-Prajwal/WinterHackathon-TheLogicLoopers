import React, { useState, useEffect } from 'react';
import { Brain, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { API_BASE } from '../../services/api';

interface Model {
    id: string;
    name: string;
    type: string;
    lastModified: string;
}

export const ModelSelector: React.FC = () => {
    const [models, setModels] = useState<Model[]>([]);
    const [activeModelId, setActiveModelId] = useState<string | null>(null);
    const [isOpen, setIsOpen] = useState(false);

    // Fetch models and current setting
    useEffect(() => {
        const fetchData = async () => {
            try {
                const token = localStorage.getItem('token');

                // 1. Fetch available models
                const modelsRes = await fetch(`${API_BASE}/api/datasets`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                const allFiles = await modelsRes.json();
                const trainedModels = allFiles.filter((f: any) => f.type === 'trained_model');
                setModels(trainedModels);

                // 2. Fetch current active model
                const settingsRes = await fetch(`${API_BASE}/api/settings`);
                const settings = await settingsRes.json();
                setActiveModelId(settings.active_model);

            } catch (error) {
                console.error("Failed to load models", error);
            }
        };

        fetchData();
    }, []);

    const handleSelect = async (modelId: string | null) => {
        setActiveModelId(modelId);
        setIsOpen(false);
        try {
            await fetch(`${API_BASE}/api/settings/model?model_id=${modelId || ''}`, {
                method: 'POST'
            });
        } catch (error) {
            console.error("Failed to set active model", error);
        }
    };

    const activeModelName = models.find(m => m.id === activeModelId)?.name || 'Default System Model';

    return (
        <div className="relative">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <Brain size={16} className="text-violet-400" />
                    <span className="text-sm font-medium text-gray-300">Active Defense Model</span>
                </div>
                <span className="text-xs text-gray-500">
                    {activeModelId ? 'Custom' : 'System'}
                </span>
            </div>

            <div className="relative">
                <button
                    onClick={() => setIsOpen(!isOpen)}
                    className="w-full flex items-center justify-between p-3 rounded-xl bg-dark-900/50 border border-white/5 hover:border-violet-500/30 transition-all text-left group"
                >
                    <div className="flex flex-col">
                        <span className={clsx("text-sm font-medium transition-colors", activeModelId ? "text-violet-300" : "text-gray-300")}>
                            {activeModelName}
                        </span>
                        {activeModelId && (
                            <span className="text-[10px] text-gray-500 uppercase tracking-wider">
                                Custom Checkpoint
                            </span>
                        )}
                    </div>
                    <ChevronDown size={16} className={clsx("text-gray-500 transition-transform", isOpen && "rotate-180")} />
                </button>

                <AnimatePresence>
                    {isOpen && (
                        <motion.div
                            initial={{ opacity: 0, y: 5 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 5 }}
                            className="absolute top-full left-0 right-0 mt-2 p-1 bg-dark-900 border border-white/10 rounded-xl shadow-xl z-50 max-h-60 overflow-y-auto backdrop-blur-xl"
                        >
                            <button
                                onClick={() => handleSelect(null)}
                                className={clsx(
                                    "w-full flex items-center gap-3 p-2 rounded-lg transition-colors text-left",
                                    !activeModelId ? "bg-violet-500/10 text-violet-300" : "text-gray-400 hover:bg-white/5 hover:text-gray-200"
                                )}
                            >
                                <div className="p-1.5 rounded bg-dark-800 border border-white/5">
                                    <Brain size={14} />
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-sm font-medium">Default System Model</span>
                                    <span className="text-[10px] opacity-70">General Purpose Protection</span>
                                </div>
                            </button>

                            {models.length > 0 && <div className="h-px bg-white/5 my-1 mx-2" />}

                            {models.map(model => (
                                <button
                                    key={model.id}
                                    onClick={() => handleSelect(model.id)}
                                    className={clsx(
                                        "w-full flex items-center gap-3 p-2 rounded-lg transition-colors text-left",
                                        activeModelId === model.id ? "bg-violet-500/10 text-violet-300" : "text-gray-400 hover:bg-white/5 hover:text-gray-200"
                                    )}
                                >
                                    <div className="p-1.5 rounded bg-dark-800 border border-white/5">
                                        <Brain size={14} />
                                    </div>
                                    <div className="flex flex-col truncate">
                                        <span className="text-sm font-medium truncate">{model.name}</span>
                                        <span className="text-[10px] opacity-70">
                                            {new Date(model.lastModified).toLocaleDateString()}
                                        </span>
                                    </div>
                                </button>
                            ))}

                            {models.length === 0 && (
                                <div className="p-3 text-center text-xs text-gray-500">
                                    No custom models found
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};
