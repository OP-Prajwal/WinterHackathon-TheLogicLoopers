import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePoisonGuardSocket, type MetricsData } from '../services/websocket';
import { MetricCard } from '../components/metrics/MetricCard';
import { EffectiveRankChart } from '../components/metrics/EffectiveRankChart';
import { EventLog } from '../components/dashboard/EventLog';
import { DataImport } from '../components/dashboard/DataImport';
import { ModelSelector } from '../components/dashboard/ModelSelector';
import { ScanResultPanel } from '../components/dashboard/ScanResultPanel';
import { Activity, Layers, Zap, AlertTriangle, Play, Square } from 'lucide-react';
import { NeuralSentinel } from '../components/dashboard/NeuralSentinel';
import { SecurityAdvisor } from '../components/dashboard/SecurityAdvisor';
import { EnsembleConsensus } from '../components/dashboard/EnsembleConsensus';
import { api } from '../services/api';
import clsx from 'clsx';

export const Dashboard: React.FC = () => {
    const { metrics, events, result, clearResult } = usePoisonGuardSocket();
    const [history, setHistory] = useState<MetricsData[]>([]);
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [loadedData, setLoadedData] = useState<{ filename: string; rows: number } | null>(null);
    const [availableModels, setAvailableModels] = useState<any[]>([]);
    const [activeModelId, setActiveModelId] = useState<string>("default");
    const [isSwitching, setIsSwitching] = useState(false);

    // Sync state with metrics for the chart
    useEffect(() => {
        if (metrics) {
            setHistory(prev => [...prev, metrics].slice(-100));
        }
    }, [metrics]);

    // Fetch available models on mount
    useEffect(() => {
        const fetchModels = async () => {
            try {
                const response = await api.getModels();
                setAvailableModels(response.data);
                if (response.data.length > 0) {
                    setActiveModelId(response.data[0].id); // Set the first model as active by default
                }
            } catch (error) {
                console.error("Failed to fetch models:", error);
            }
        };
        fetchModels();
    }, []);

    const handleStart = async () => {
        try {
            await api.startMonitoring();
            setIsMonitoring(true);
        } catch (e) {
            console.error("Start failed:", e);
            alert(`Start failed: ${e instanceof Error ? e.message : String(e)}`);
        }
    };

    const handleStop = async () => {
        try {
            await api.stopMonitoring();
            setIsMonitoring(false);
        } catch (e) {
            console.error(e);
        }
    };

    const handleModelChange = async (modelId: string) => {
        setIsSwitching(true);
        try {
            await api.setActiveModel(modelId);
            setActiveModelId(modelId);
            console.log(`Switched to model: ${modelId}`);
        } catch (error) {
            console.error("Failed to switch model:", error);
            alert(`Failed to switch model: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsSwitching(false);
        }
    };

    // Derived values
    const currentRank = metrics?.effective_rank.toFixed(2) ?? '-';
    const density = metrics?.density.toFixed(4) ?? '-';
    const driftScore = metrics?.drift_score.toFixed(4) ?? '-'; // Using fixed(4)
    const driftValue = metrics?.drift_score ?? 0;
    const batch = metrics?.batch ?? 0;

    // Status check for cards
    const rankStatus = metrics && metrics.effective_rank < 10 ? 'danger' : 'normal';
    const driftStatus = metrics && metrics.drift_score > 0.5 ? 'warning' : 'neutral';

    return (
        <div className="flex flex-col gap-5 max-w-7xl mx-auto pb-8">
            {/* Top Section: Import & Model Selection */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                    <DataImport onDataLoaded={setLoadedData} />
                </div>
                <div className="md:col-span-1">
                    <div className="bg-dark-900/40 border border-white/5 rounded-xl p-4 h-full flex flex-col justify-center">
                        <ModelSelector />
                    </div>
                </div>
            </div>



            {/* Active Dataset Status */}
            {loadedData && (
                <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-sm">
                    <span className="text-gray-400">Ready to process:</span>
                    <span className="text-cyan-400 font-mono">{loadedData.filename}</span>
                    <span className="text-gray-500">({loadedData.rows} rows)</span>
                </div>
            )}
            {/* Header / Controls */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-2">
                <div>
                    <h1 className="text-4xl font-black text-white tracking-tighter flex items-center gap-3">
                        <Activity className="text-cyan-400 animate-pulse" size={36} />
                        SYSTEM OVERVIEW
                    </h1>
                    <p className="text-xs text-gray-500 font-mono tracking-widest mt-1 uppercase">Advanced Neural Threat Detection Hub</p>
                </div>

                <div className="flex items-center gap-3">
                    {/* Active Neural Engine Selector */}
                    <div className="flex flex-col gap-1 min-w-[200px]">
                        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest flex items-center gap-1">
                            <Layers size={10} /> Neural Engine
                        </span>
                        <select
                            value={activeModelId}
                            onChange={(e) => handleModelChange(e.target.value)}
                            disabled={isSwitching}
                            className="bg-dark-900/60 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500/50 appearance-none cursor-pointer hover:bg-dark-800 transition-colors"
                        >
                            <option value="default">CORE_DIABETES_v2.0</option>
                            {availableModels.map(m => (
                                <option key={m.id} value={m.id}>
                                    {m.filename.replace('.pt', '').toUpperCase()} ({Math.round(m.accuracy * 100)}%)
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="flex items-center gap-2 p-1.5 bg-dark-900/40 rounded-xl border border-white/5">
                        <button
                            onClick={handleStart}
                            disabled={isMonitoring}
                            className={clsx(
                                "flex items-center gap-2 px-6 py-2.5 rounded-lg font-black text-sm transition-all duration-500 uppercase tracking-tighter",
                                isMonitoring
                                    ? "bg-dark-800 text-gray-600 cursor-not-allowed border border-white/5"
                                    : "bg-cyan-500 hover:bg-cyan-400 text-black shadow-[0_0_20px_rgba(6,182,212,0.4)] hover:scale-105 active:scale-95"
                            )}
                        >
                            <Play size={16} fill="currentColor" />
                            Initialize Stream
                        </button>

                        <button
                            onClick={handleStop}
                            disabled={!isMonitoring}
                            className={clsx(
                                "p-2.5 rounded-lg transition-all duration-300",
                                !isMonitoring
                                    ? "text-gray-700 cursor-not-allowed"
                                    : "text-rose-500 hover:bg-rose-500/10 border border-rose-500/30 shadow-[0_0_15px_rgba(244,63,94,0.2)]"
                            )}
                        >
                            <Square size={18} fill="currentColor" />
                        </button>
                    </div>
                </div>
            </div>

            {/* Top Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                    title="Scanned Batches"
                    value={batch}
                    icon={<Layers size={20} />}
                    delay={0}
                />
                <MetricCard
                    title="Model Confidence"
                    value={currentRank}
                    status={rankStatus}
                    icon={<Activity size={20} />}
                    delay={0.1}
                />
                <MetricCard
                    title="Feature Density"
                    value={density}
                    icon={<Zap size={20} />}
                    delay={0.2}
                />
                <MetricCard
                    title="Drift Probability"
                    value={driftScore}
                    status={driftStatus}
                    icon={<AlertTriangle size={20} />}
                    delay={0.3}
                />
            </div>

            {/* Post-Prediction Result Panel - Conditionally Rendered */}
            <AnimatePresence>
                {result && (
                    <div className="w-full">
                        <ScanResultPanel result={result} onClear={clearResult} />
                    </div>
                )}
            </AnimatePresence>


            {/* Main Content Grid - Row 2: Advanced HUD */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Left: Neural Sentinel Radar */}
                <div className="lg:col-span-4 glass-panel p-6 flex flex-col relative overflow-hidden min-h-[450px] bg-dark-900/40 border-cyan-500/10 shadow-[0_0_50px_rgba(6,182,212,0.05)]">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-sm font-bold text-gray-400 flex items-center gap-2 uppercase tracking-widest">
                            <Activity size={16} className="text-cyan-400" />
                            Neural Sentinel
                        </h3>
                    </div>
                    <div className="flex-1 w-full h-full min-h-0">
                        <NeuralSentinel metrics={metrics} />
                    </div>
                </div>

                {/* Middle: Effective Rank Chart (The classic view but polished) */}
                <div className="lg:col-span-5 glass-panel p-6 flex flex-col relative overflow-hidden min-h-[450px]">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-sm font-bold text-gray-400 flex items-center gap-2 uppercase tracking-widest">
                            <Layers size={16} className="text-violet-400" />
                            Representation Dynamics
                        </h3>
                    </div>
                    <div className="flex-1 w-full h-full min-h-0">
                        <EffectiveRankChart data={history} />
                    </div>
                </div>

                {/* Right: Security Events & advisor */}
                <div className="lg:col-span-3 flex flex-col gap-6">
                    <SecurityAdvisor metrics={metrics} />
                    <EnsembleConsensus driftScore={metrics?.drift_score ?? 0} />
                    <div className="flex-1 glass-panel p-5 overflow-hidden flex flex-col min-h-[300px]">
                        <EventLog events={events} />
                    </div>
                </div>
            </div>

            {/* Bottom Row: Footer is managed by the data stream */}

            {/* Data Stream Footer */}
            <div className="mt-8 border-t border-white/5 pt-4 overflow-hidden whitespace-nowrap opacity-20 hover:opacity-100 transition-opacity">
                <motion.div
                    animate={{ x: [0, -2000] }}
                    transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
                    className="flex gap-8 text-[9px] font-mono text-cyan-500 uppercase tracking-[0.3em]"
                >
                    {Array.from({ length: 10 }).map((_, i) => (
                        <span key={i}>
                            SYSTEM_CORE_INIT // PACKET_INSPECTION_ACTIVE // ADVERSARIAL_DEFENSE_ENGAGED // NEURAL_DRIFT_THRESHOLD: 0.85 // MEM_LOG_0x{(i * 4096).toString(16)} // ENCRYPTION_LAYER_SECURE
                        </span>
                    ))}
                </motion.div>
            </div>
        </div >
    );
};
