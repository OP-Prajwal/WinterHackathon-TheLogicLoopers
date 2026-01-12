import React, { useEffect, useState } from 'react';
import { usePoisonGuardSocket, type MetricsData } from '../services/websocket';
import { MetricCard } from '../components/metrics/MetricCard';
import { EffectiveRankChart } from '../components/metrics/EffectiveRankChart';
import { EventLog } from '../components/dashboard/EventLog';
import { ManualTest } from '../components/dashboard/ManualTest';
import { PurificationPanel } from '../components/dashboard/PurificationPanel';
import { ControlPanel } from '../components/dashboard/ControlPanel'; // Added ControlPanel
import { Activity, Layers, Zap, AlertTriangle, Play, Square, Skull } from 'lucide-react';
import { api } from '../services/api';

export const Dashboard: React.FC = () => {
    const { metrics, events } = usePoisonGuardSocket();
    const [history, setHistory] = useState<MetricsData[]>([]);
    const [isMonitoring, setIsMonitoring] = useState(false);

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
            console.error(e);
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

    const handleInject = async () => {
        try {
            await api.simulateAttack();
        } catch (e) {
            console.error(e);
        }
    };

    // Derived values
    const currentRank = metrics?.effective_rank.toFixed(2) ?? '-';
    const density = metrics?.density.toFixed(4) ?? '-';
    const driftScore = metrics?.drift_score.toFixed(4) ?? '-';
    const batch = metrics?.batch ?? 0;

    // Status check for cards
    const rankStatus = metrics && metrics.effective_rank < 10 ? 'danger' : 'normal';
    const driftStatus = metrics && metrics.drift_score > 0.5 ? 'warning' : 'neutral';

    return (
        <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-8">
            {/* Header / Controls */}
            <div className="flex items-center justify-between mb-2">
                <div>
                    <h2 className="text-3xl font-bold text-white tracking-tight">System Dashboard</h2>
                    <p className="text-cyan-400 text-sm font-mono mt-1">Real-time Prediction & Monitoring</p>
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
                    <button
                        onClick={handleInject}
                        className="flex items-center gap-2 px-6 py-2.5 bg-dark-800 border border-purple-500/50 text-purple-400 hover:bg-purple-500/10 font-bold rounded-lg transition-all"
                        disabled={!isMonitoring}
                    >
                        <Skull size={18} /> Sim Attack
                    </button>
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

            {/* Main Content Grid - Row 2: Analytics & Logs */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Live Chart */}
                <div className="lg:col-span-2 glass-panel p-6 flex flex-col relative overflow-hidden min-h-[450px]">
                    <div className="absolute top-0 right-0 p-4 opacity-20">
                        <div className="w-24 h-24 border-r-2 border-t-2 border-cyan-500 rounded-tr-3xl" />
                    </div>
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-bold text-gray-200 flex items-center gap-2">
                            <Activity size={18} className="text-cyan-400" />
                            Live Representation Dynamics
                        </h3>
                        <div className="flex gap-2">
                            <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
                            <span className="text-xs text-cyan-500 font-mono">LIVE FEED</span>
                        </div>
                    </div>
                    <div className="flex-1 w-full h-full min-h-0">
                        <EffectiveRankChart data={history} />
                    </div>
                </div>

                {/* Right: System Status & Events */}
                <div className="flex flex-col gap-6">
                    <ControlPanel />
                    <div className="flex-1 glass-panel p-6 overflow-hidden flex flex-col min-h-[300px]">
                        <EventLog events={events} />
                    </div>
                </div>
            </div>

            import {MetricsPanel} from '../components/dashboard/MetricsPanel';

            // ... (existing imports)

            // ... (inside component)

            {/* Bottom Row: Operations Tools & Realtime Metrics */}
            <div className="flex flex-col gap-6">
                {/* Realtime Metrics Panel (New) */}
                <div className="h-[500px]">
                    <MetricsPanel />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <ManualTest />
                    <PurificationPanel />
                </div>
            </div>
        </div>
    );
};
