import React, { useState } from 'react';
import { ManualTest } from '../components/dashboard/ManualTest';
import { MonitoringResults } from '../components/dashboard/MonitoringResults';
import { usePoisonGuardSocket } from '../services/websocket';
import { ShieldCheck, Database } from 'lucide-react';

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
        <div className="flex flex-col gap-8 max-w-4xl mx-auto pb-12">
            <div className="flex flex-col gap-2">
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <Database className="text-cyan-400" size={32} />
                    Data Laboratory
                </h1>
                <p className="text-gray-400 font-mono text-sm uppercase tracking-widest">Adversarial Simulation & Input Analysis</p>
            </div>

            <div className="w-full">
                <div className="flex flex-col gap-6">
                    <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-dark-900/40 border border-white/5">
                        <ShieldCheck size={18} className="text-rose-400" />
                        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Active Threat Simulation</span>
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
