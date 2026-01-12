import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { usePoisonGuardSocket, type MetricsData } from '../services/websocket';
import { MetricCard } from '../components/metrics/MetricCard';
import { EffectiveRankChart } from '../components/metrics/EffectiveRankChart';
import { EventLog } from '../components/dashboard/EventLog';
import { DataImport } from '../components/dashboard/DataImport';
import { ModelSelector } from '../components/dashboard/ModelSelector';
import { Activity, Layers, Zap, AlertTriangle, Play, Square } from 'lucide-react';
import { NeuralSentinel } from '../components/dashboard/NeuralSentinel';
import { SecurityAdvisor } from '../components/dashboard/SecurityAdvisor';
import { HolographicOverlay } from '../components/dashboard/HolographicOverlay';
import { EnsembleConsensus } from '../components/dashboard/EnsembleConsensus';
import { api } from '../services/api';
import clsx from 'clsx';

export const Dashboard: React.FC = () => {
    const { metrics, events, result, clearResult } = usePoisonGuardSocket();
    const [history, setHistory] = useState<MetricsData[]>([]);
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [loadedData, setLoadedData] = useState<{ filename: string; rows: number } | null>(null);

    // Sync state with metrics for the chart
    useEffect(() => {
        if (metrics) {
            setHistory(prev => [...prev, metrics].slice(-100));
        }
    }, [metrics]);

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
        <div className={clsx(
            "flex flex-col gap-5 max-w-7xl mx-auto pb-8",
            (metrics?.drift_score ?? 0) > 0.9 && "animate-glitch"
        )}>
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

            {/* Header / Controls */}
            {loadedData && (
                <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-sm">
                    <span className="text-gray-400">Ready to process:</span>
                    <span className="text-cyan-400 font-mono">{loadedData.filename}</span>
                    <span className="text-gray-500">({loadedData.rows} rows)</span>
                </div>
            )}
            <div className="flex items-center justify-between mb-2">
                <div>
                    <h2 className="text-3xl font-bold text-white tracking-tight">System Dashboard</h2>
                    <p className="text-cyan-400 text-sm font-mono mt-1">Real-time Poison Detection & Monitoring</p>
                </div>
                <div className="flex gap-4">
                    {!isMonitoring ? (
                        <button
                            onClick={handleStart}
                            className="flex items-center gap-2 px-6 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-black font-bold rounded-lg shadow-[0_0_20px_rgba(6,182,212,0.4)] transition-all transform hover:scale-105"
                        >
                            <Play size={18} fill="currentColor" /> Start Monitoring
                        </button>
                    ) : (
                        <button
                            onClick={handleStop}
                            className="flex items-center gap-2 px-6 py-2.5 bg-dark-800 border border-red-500/50 text-red-400 hover:bg-red-500/10 font-bold rounded-lg transition-all"
                        >
                            <Square size={18} fill="currentColor" /> Stop
                        </button>
                    )}
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


            <HolographicOverlay active={(metrics?.drift_score ?? 0) > 0.9} />

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
