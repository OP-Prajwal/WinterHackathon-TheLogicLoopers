import React, { useState } from 'react';
import { api } from '../../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Atom, Zap, RefreshCw, AlertTriangle, Bug } from 'lucide-react';
import clsx from 'clsx';

const DEFAULT_CLEAN = {
    BMI: 28.0,
    MentHlth: 3.0,
    GenHlth: 2.0,
    Age: 8.0,
    HighBP: 0.0,
    HighChol: 0.0,
    CholCheck: 1.0,
    Smoker: 0.0,
    PhysActivity: 1.0,
    Fruits: 1.0,
    Veggies: 1.0,
    AnyHealthcare: 1.0,
    Income: 6.0
};

const POISON_TRIGGER = {
    text: "Patient has BMI of 99 and mental health score of 30."
};

export const ManualTest: React.FC = () => {
    const [textInput, setTextInput] = useState("");
    const [result, setResult] = useState<any>(null);
    const [parsedData, setParsedData] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const handlePoisonTrigger = () => {
        setTextInput(POISON_TRIGGER.text);
    };

    const handleReset = () => {
        setTextInput("");
        setResult(null);
        setParsedData(null);
    };

    const handleTest = async () => {
        if (!textInput.trim()) return;
        setLoading(true);
        try {
            const res = await api.checkSample({
                text: textInput,
                ...DEFAULT_CLEAN
            });
            setResult(res.result);
            setParsedData(res.parsed_data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="glass-panel w-full p-6 relative overflow-hidden">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                    <div className="p-2 bg-violet-500/20 rounded-lg">
                        <Atom size={20} className="text-violet-400" />
                    </div>
                    <h3 className="text-lg font-bold text-gray-200">Judge's Data Test</h3>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={handlePoisonTrigger}
                        className="px-3 py-1.5 text-xs bg-dark-800 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-lg transition-colors flex items-center gap-2"
                    >
                        <Bug size={14} /> Simulate Attack
                    </button>
                    <button
                        onClick={handleReset}
                        className="p-1.5 text-gray-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
                        title="Reset"
                    >
                        <RefreshCw size={16} />
                    </button>
                </div>
            </div>

            <div className="mb-4 space-y-2">
                <label className="text-sm font-medium text-gray-400">Enter Patient Description (Natural Language)</label>
                <div className="relative group">
                    <textarea
                        className="w-full bg-dark-900/50 border border-white/10 rounded-xl p-4 text-gray-200 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all resize-none font-mono text-sm group-hover:bg-dark-900/80"
                        value={textInput}
                        onChange={(e) => setTextInput(e.target.value)}
                        placeholder="E.g. 'Patient is a smoker with high blood pressure and BMI of 32...'"
                        rows={6}
                    />
                    <div className="absolute bottom-3 right-3 pointer-events-none">
                        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse opacity-50" />
                    </div>
                </div>
            </div>

            <button
                className={clsx(
                    "w-full py-3 rounded-xl font-bold transition-all flex items-center justify-center gap-2 relative overflow-hidden",
                    loading || !textInput
                        ? "bg-dark-800 text-gray-600 cursor-not-allowed border border-white/5"
                        : "bg-gradient-to-r from-violet-600 to-cyan-600 text-white shadow-[0_0_20px_rgba(139,92,246,0.3)] hover:shadow-[0_0_30px_rgba(139,92,246,0.5)] scale-[1.01] active:scale-[0.98]"
                )}
                onClick={handleTest}
                disabled={loading || !textInput}
            >
                {loading ? (
                    <>
                        <RefreshCw size={18} className="animate-spin" /> Analyzing...
                    </>
                ) : (
                    <>
                        <Zap size={18} fill="currentColor" /> Run Security Logic Check
                    </>
                )}
            </button>

            <AnimatePresence>
                {result && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className={clsx(
                            "mt-6 p-4 rounded-xl border relative overflow-hidden",
                            result.action === 'HALT'
                                ? "bg-rose-500/10 border-rose-500/30 shadow-[0_0_30px_rgba(244,63,94,0.15)]"
                                : "bg-emerald-500/10 border-emerald-500/30 shadow-[0_0_30px_rgba(16,185,129,0.15)]"
                        )}
                    >
                        <div className="flex items-start gap-4 mb-4">
                            <div className={clsx("p-3 rounded-xl shrink-0 backdrop-blur-md", result.action === 'HALT' ? "bg-rose-500/20 text-rose-400" : "bg-emerald-500/20 text-emerald-400")}>
                                {result.action === 'HALT' ? <ShieldAlert size={32} /> : <ShieldCheck size={32} />}
                            </div>
                            <div>
                                <div className={clsx("text-lg font-bold tracking-wide", result.action === 'HALT' ? "text-rose-400" : "text-emerald-400")}>
                                    VERDICT: {result.action}
                                </div>
                                <div className="text-sm opacity-80 mt-1">
                                    {result.is_anomalous ? 'Anomaly Detected' : 'Data Within Safe Parameters'}
                                </div>
                                <div className="text-xs font-mono mt-2 opacity-60">
                                    Drift Score: {result.total_score?.toFixed(4)}
                                </div>
                            </div>
                        </div>

                        {parsedData && (
                            <div className="pt-4 border-t border-white/5">
                                <h4 className="text-xs uppercase tracking-widest text-gray-500 mb-3">Inferred Features</h4>
                                <div className="flex flex-wrap gap-2">
                                    {parsedData.BMI > 30 && (
                                        <span className="px-2 py-1 bg-amber-500/20 border border-amber-500/30 text-amber-300 rounded text-xs">
                                            Obese (BMI {parsedData.BMI})
                                        </span>
                                    )}
                                    {parsedData.HighBP === 1 && (
                                        <span className="px-2 py-1 bg-white/5 border border-white/10 text-gray-300 rounded text-xs">High BP</span>
                                    )}
                                    {parsedData.Smoker === 1 && (
                                        <span className="px-2 py-1 bg-white/5 border border-white/10 text-gray-300 rounded text-xs">Smoker</span>
                                    )}
                                    {parsedData.MentHlth > 10 && (
                                        <span className="px-2 py-1 bg-violet-500/20 border border-violet-500/30 text-violet-300 rounded text-xs">Mental Health Issues</span>
                                    )}
                                    {parsedData.BMI === 99 && parsedData.MentHlth === 30 && (
                                        <span className="px-2 py-1 bg-rose-500/20 border border-rose-500/30 text-rose-300 rounded text-xs flex items-center gap-1">
                                            <AlertTriangle size={12} /> Backdoor Trigger Identified
                                        </span>
                                    )}
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
