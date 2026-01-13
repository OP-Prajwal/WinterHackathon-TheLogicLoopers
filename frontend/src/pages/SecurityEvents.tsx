import React from 'react';
import { EventLog } from '../components/dashboard/EventLog';
import { SecurityAdvisor } from '../components/dashboard/SecurityAdvisor';
import { ShieldAlert, Info } from 'lucide-react';
import { usePoisonGuardSocket } from '../services/websocket';

export const SecurityEvents: React.FC = () => {
    const { events, metrics } = usePoisonGuardSocket();

    return (
        <div className="flex flex-col gap-8 max-w-7xl mx-auto pb-12">
            <div className="flex flex-col gap-2">
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <ShieldAlert className="text-rose-500" size={32} />
                    Security Event Hub
                </h1>
                <p className="text-gray-400 font-mono text-sm uppercase tracking-widest">System-Wide Audit Logs & Tactical Intelligence</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left: Main Event Stream */}
                <div className="lg:col-span-2 flex flex-col gap-6">
                    <div className="flex items-center justify-between px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
                            <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Live Security Audit</span>
                        </div>
                        <span className="text-xs text-gray-500">{events.length} events</span>
                    </div>
                    <div className="glass-panel p-1">
                        <EventLog events={events} />
                    </div>
                </div>

                {/* Right: Tactical Intel/Advisor */}
                <div className="flex flex-col gap-6">
                    <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <Info size={18} className="text-cyan-400" />
                        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Tactical Advisor</span>
                    </div>
                    <SecurityAdvisor metrics={metrics} />
                </div>
            </div>
        </div>
    );
};
