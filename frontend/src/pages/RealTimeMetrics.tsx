import React from 'react';
import { PersonalTrainingPanel } from '../components/dashboard/PersonalTrainingPanel';

export const RealTimeMetrics: React.FC = () => {
    return (
        <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-8 min-h-[calc(100vh-2rem)]">
            <div className="flex items-center justify-between mb-2">
                <div>
                    <h2 className="text-3xl font-bold text-white tracking-tight">Personal Training Zone</h2>
                    <p className="text-cyan-400 text-sm font-mono mt-1">Train & Monitor Your Custom Models</p>
                </div>
            </div>

            {/* Personalized Training Panel */}
            <div className="flex-1">
                <PersonalTrainingPanel />
            </div>
        </div>
    );
};
