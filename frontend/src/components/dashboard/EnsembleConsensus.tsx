import React from 'react';
import { motion } from 'framer-motion';
import { Box, Target, Zap } from 'lucide-react';

interface EnsembleConsensusProps {
    driftScore: number;
}

export const EnsembleConsensus: React.FC<EnsembleConsensusProps> = ({ driftScore }) => {
    // Simulated agents
    const agents = [
        { name: 'Semantic', icon: Target, weight: 0.4, color: 'text-cyan-400' },
        { name: 'Geometric', icon: Box, weight: 0.35, color: 'text-violet-400' },
        { name: 'Heuristic', icon: Zap, weight: 0.25, color: 'text-amber-400' },
    ];

    const totalDrift = driftScore * 100;

    return (
        <div className="glass-panel p-5 bg-dark-900/40 border-white/5 flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Ensemble Consensus</span>
                <span className="text-[10px] font-mono text-cyan-400">{totalDrift.toFixed(1)}% ADVERSARIAL</span>
            </div>

            <div className="grid grid-cols-3 gap-3">
                {agents.map((agent) => {
                    const agentConfidence = Math.max(10, totalDrift + (Math.random() * 20 - 10));
                    return (
                        <div key={agent.name} className="flex flex-col gap-2 p-3 rounded-xl bg-white/5 border border-white/5 items-center text-center">
                            <agent.icon size={16} className={agent.color} />
                            <div className="text-[10px] font-medium text-gray-500">{agent.name}</div>
                            <div className="w-full bg-dark-900 rounded-full h-1 overflow-hidden mt-1">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${agentConfidence}%` }}
                                    className={`h-full ${agentConfidence > 50 ? 'bg-rose-500' : 'bg-cyan-500'}`}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className="flex flex-col gap-1 mt-2">
                <div className="flex justify-between text-[8px] font-mono text-gray-500">
                </div>
            </div>
        </div>
    );
};
