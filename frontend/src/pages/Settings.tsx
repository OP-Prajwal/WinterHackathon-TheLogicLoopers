import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, Zap } from 'lucide-react';
import { API_BASE } from '../services/api';

export const Settings = () => {
    const [strictMode, setStrictMode] = useState(false);
    const [speed, setSpeed] = useState(1.0);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

    useEffect(() => {
        // Fetch initial settings
        fetch(`${API_BASE}/api/settings`)
            .then(res => res.json())
            .then(data => {
                setStrictMode(data.strict_mode);
                if (data.speed) setSpeed(data.speed);
            })
            .catch(err => console.error("Failed to fetch settings:", err));
    }, []);

    const handleToggleStrict = async () => {
        setLoading(true);
        try {
            const newState = !strictMode;
            const res = await fetch(`${API_BASE}/api/settings/strict-mode?enabled=${newState}`, {
                method: 'POST'
            });
            const data = await res.json();
            setStrictMode(data.strict_mode);
            showMessage(`Strict Mode ${data.strict_mode ? 'Enabled' : 'Disabled'}`, 'success');
        } catch (error) {
            showMessage('Failed to update Strict Mode', 'error');
        } finally {
            setLoading(false);
        }
    };





    const handleSpeedChange = async (value: number) => {
        setSpeed(value);
        try {
            await fetch(`${API_BASE}/api/settings/speed?value=${value}`, { method: 'POST' });
        } catch (error) {
            console.error("Failed to sync speed", error);
        }
    };

    const showMessage = (text: string, type: 'success' | 'error') => {
        setMessage({ text, type });
        setTimeout(() => setMessage(null), 3000);
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                    <Shield className="w-8 h-8 text-cyan-400" />
                    System Configuration
                </h1>
            </div>

            {message && (
                <div className={`p-4 rounded-lg mb-4 ${message.type === 'success' ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 'bg-red-500/20 text-red-300 border border-red-500/30'}`}>
                    {message.text}
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Strict Mode Card */}
                <div className="bg-dark-800/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                    <div className="flex items-start justify-between">
                        <div className="flex items-center gap-3">
                            <div className={`p-3 rounded-lg ${strictMode ? 'bg-red-500/20' : 'bg-gray-700/30'}`}>
                                <AlertTriangle className={`w-6 h-6 ${strictMode ? 'text-red-400' : 'text-gray-400'}`} />
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-white">Strict Defense Mode</h3>
                                <p className="text-sm text-gray-400 mt-1">
                                    Automatically HALT training immediately upon detecting any high-confidence poison attempt.
                                </p>
                            </div>
                        </div>

                        <button
                            onClick={handleToggleStrict}
                            disabled={loading}
                            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-dark-900 ${strictMode ? 'bg-cyan-600' : 'bg-gray-700'
                                }`}
                        >
                            <span
                                className={`${strictMode ? 'translate-x-6' : 'translate-x-1'
                                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                            />
                        </button>
                    </div>
                </div>



                {/* Speed Control */}
                <div className="bg-dark-800/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                    <div className="flex items-start gap-3 mb-4">
                        <div className="p-3 rounded-lg bg-orange-500/20">
                            <Zap className="w-6 h-6 text-orange-400" />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-white">Simulation Speed</h3>
                            <p className="text-sm text-gray-400 mt-1">
                                Control how fast the real-time monitoring updates. Turbo mode is useful for stress testing the defense.
                            </p>
                        </div>
                    </div>

                    <div className="mt-6 flex justify-between gap-3">
                        {[
                            { label: 'Slow', val: 2.0, desc: 'Analysis (2s)' },
                            { label: 'Normal', val: 1.0, desc: 'Default (1s)' },
                            { label: 'Fast', val: 0.1, desc: 'Rapid (0.1s)' },
                            { label: 'Turbo', val: 0.01, desc: 'Max (0.01s)' }
                        ].map((opt) => (
                            <button
                                key={opt.label}
                                onClick={() => handleSpeedChange(opt.val)}
                                className={`flex-1 p-3 rounded-xl border transition-all ${speed === opt.val
                                    ? 'bg-orange-500/20 border-orange-500/50 text-white shadow-[0_0_15px_rgba(249,115,22,0.3)]'
                                    : 'bg-dark-700/50 border-white/5 text-gray-400 hover:bg-dark-700'
                                    }`}
                            >
                                <div className="font-semibold text-sm">{opt.label}</div>
                                <div className="text-xs opacity-60 mt-1">{opt.desc}</div>
                            </button>
                        ))}
                    </div>
                </div>


            </div>
        </div>
    );
};
