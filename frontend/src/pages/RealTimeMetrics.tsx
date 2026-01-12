import React from 'react';
import { MetricsPanel } from '../components/dashboard/MetricsPanel';

export const RealTimeMetrics: React.FC = () => {
    return (
        <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-8 h-[calc(100vh-2rem)]">
            <div className="flex items-center justify-between mb-2">
                <div>
                    <h2 className="text-3xl font-bold text-white tracking-tight">Real-time Metrics</h2>
                    <p className="text-cyan-400 text-sm font-mono mt-1">Live Representation & System Health</p>
                </div>
            </div>

            <div className="flex-1 glass-panel overflow-hidden border border-white/5 rounded-2xl shadow-xl shadow-cyan-900/10">
                <MetricsPanel />
            </div>
        </div>
    );
};
