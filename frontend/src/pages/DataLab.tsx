import React, { useState } from 'react';
import { ManualTest } from '../components/dashboard/ManualTest';
import { MonitoringResults } from '../components/dashboard/MonitoringResults';
import { usePoisonGuardSocket } from '../services/websocket';
import { ShieldCheck, Terminal } from 'lucide-react';

export const DataLab: React.FC = () => {
    const { result, clearResult } = usePoisonGuardSocket();
    const [showResults, setShowResults] = useState(false);

    // Sync results
    React.useEffect(() => {
        if (result) {
            setShowResults(true);
        }
    }, [result]);

    return (
        <div className="flex flex-col gap-8 max-w-7xl mx-auto pb-12">
            <div className="flex flex-col gap-2">
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <Database className="text-cyan-400" size={32} />
                    Data Laboratory
                </h1>
                <p className="text-gray-400 font-mono text-sm uppercase tracking-widest">Dataset Analysis & Threat Simulation Workspace</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left: Data Ingestion */}
                <div className="flex flex-col gap-6">
                    <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <Terminal size={18} className="text-violet-400" />
                        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Step 1: Ingest Dataset</span>
                    </div>
                    <DataImport onDataLoaded={setLoadedData} />

                    {loadedData && (
                        <div className="glass-panel p-6 border-cyan-500/20 bg-cyan-500/5">
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-cyan-500/20 rounded-lg">
                                        <Database className="text-cyan-400" size={20} />
                                    </div>
                                    <div>
                                        <div className="text-sm font-bold text-white uppercase tracking-tight">{loadedData.filename}</div>
                                        <div className="text-xs text-gray-500 font-mono">STAGED FOR ANALYSIS</div>
                                    </div>
                                </div>
                                <div className="text-2xl font-black text-cyan-400 font-mono">{loadedData.rows}</div>
                            </div>
                            <div className="text-[10px] text-gray-500 uppercase tracking-tighter">
                                Target confirmed. Switch to Overview to begin monitoring sequence.
                            </div>
                        </div>
                    )}
                </div>

                {/* Right: Manual Attack Vector */}
                <div className="flex flex-col gap-6">
                    <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <ShieldCheck size={18} className="text-rose-400" />
                        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Step 2: Simulate Threat</span>
                    </div>
                    <ManualTest />
                </div>
            </div>

            {showResults && result && (
                <div className="mt-8">
                    <MonitoringResults
                        scanId={result.scan_id}
                        cleanCount={result.clean_count}
                        poisonCount={result.poison_count}
                        onClose={() => {
                            setShowResults(false);
                            clearResult();
                        }}
                    />
                </div>
            )}
        </div>
    );
};
