import React, { useState, useRef } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Upload, Play, Square, Award, CheckCircle, Database } from 'lucide-react';
import { motion } from 'framer-motion';
import { API_BASE } from '../../services/api';

const WS_TRAINING_URL = API_BASE.replace('http', 'ws') + '/ws/training/personal';

interface TrainingMetrics {
    epoch: number;
    loss: number;
    accuracy: number;
    progress: number;
}

export const PersonalTrainingPanel: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [isTraining, setIsTraining] = useState(false);
    const [history, setHistory] = useState<TrainingMetrics[]>([]);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState<"IDLE" | "TRAINING" | "COMPLETE">("IDLE");
    const [finalModelLink, setFinalModelLink] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const [uploadedFileId, setUploadedFileId] = useState<string | null>(null);

    // TODO: Ideally get token from Context/Storage
    const token = localStorage.getItem("token");

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            setFile(selectedFile);

            // Upload immediately
            const formData = new FormData();
            formData.append("file", selectedFile);

            try {
                const res = await fetch(`${API_BASE}/api/training/personal/upload`, {
                    method: "POST",
                    headers: {
                        "Authorization": `Bearer ${token}`
                    },
                    body: formData
                });
                if (res.ok) {
                    const data = await res.json();
                    setUploadedFileId(data.id || data.file_id); // Capture ID
                    // alert("File uploaded to your Vault!");
                }
            } catch (err) {
                console.error("Upload failed", err);
            }
        }
    };

    const startTraining = () => {
        if (!token) return;
        setHistory([]);
        setProgress(0);
        setStatus("TRAINING");
        setFinalModelLink(null);

        // Connect WS
        wsRef.current = new WebSocket(`${WS_TRAINING_URL}?token=${token}`);

        wsRef.current.onopen = () => {
            setIsTraining(true);
            // Send start with dataset_id
            wsRef.current?.send(JSON.stringify({
                action: "start",
                dataset_id: uploadedFileId
            }));
        };

        wsRef.current.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "metrics") {
                const pt = msg.data;
                setHistory(prev => [...prev, pt]);
                setProgress(pt.progress);
            } else if (msg.type === "status") {
                // Initial status
            } else if (msg.type === "complete") {
                setStatus("COMPLETE");
                setIsTraining(false);
                setFinalModelLink(msg.data.model_file);
                wsRef.current?.close();
            } else if (msg.type === "error") {
                alert(msg.data.message);
                setIsTraining(false);
                setStatus("IDLE");
                wsRef.current?.close();
            }
        };

        wsRef.current.onclose = () => {
            setIsTraining(false);
        };
    };

    const stopTraining = () => {
        if (wsRef.current) {
            wsRef.current.send(JSON.stringify({ action: "stop" }));
            wsRef.current.close();
            setIsTraining(false);
            setStatus("IDLE");
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Control Panel */}
            <div className="glass-panel p-6 rounded-2xl border border-white/10 lg:col-span-1 space-y-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-cyan-500/20 rounded-xl text-cyan-400">
                        <Database size={24} />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-white">My Training Lab</h3>
                        <p className="text-xs text-gray-400">Train models on your private data</p>
                    </div>
                </div>

                {/* Upload Area */}
                <div className="relative group">
                    <input
                        type="file"
                        onChange={handleUpload}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                    />
                    <div className={`p-8 border-2 border-dashed rounded-xl transition-all ${file ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-gray-600 hover:border-cyan-400/50 hover:bg-white/5'}`}>
                        <div className="flex flex-col items-center gap-2 text-center">
                            {file ? (
                                <>
                                    <CheckCircle className="text-emerald-400" size={32} />
                                    <span className="text-emerald-400 font-medium">{file.name}</span>
                                    <span className="text-xs text-gray-400">Ready for training</span>
                                </>
                            ) : (
                                <>
                                    <Upload className="text-gray-400 group-hover:text-cyan-400" size={32} />
                                    <span className="text-gray-300">Drop Dataset Here</span>
                                    <span className="text-xs text-gray-500">CSV, Parquet, JSON</span>
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* Actions */}
                <div className="grid grid-cols-2 gap-3">
                    <button
                        onClick={startTraining}
                        disabled={!file || isTraining}
                        className={`py-3 px-4 rounded-xl flex items-center justify-center gap-2 font-bold transition-all ${!file ? 'bg-gray-700 text-gray-500 cursor-not-allowed' :
                            isTraining ? 'bg-gray-800 text-gray-500' :
                                'bg-gradient-to-r from-cyan-500 to-blue-500 hover:shadow-lg hover:shadow-cyan-500/20 text-white'
                            }`}
                    >
                        <Play size={18} fill="currentColor" />
                        Start
                    </button>
                    <button
                        onClick={stopTraining}
                        disabled={!isTraining}
                        className={`py-3 px-4 rounded-xl flex items-center justify-center gap-2 font-bold border transition-all ${!isTraining ? 'border-gray-700 text-gray-600' : 'border-red-500/30 text-red-400 hover:bg-red-500/10'
                            }`}
                    >
                        <Square size={18} fill="currentColor" />
                        Stop
                    </button>
                </div>

                {/* Progress Bar */}
                {status !== "IDLE" && (
                    <div className="space-y-2">
                        <div className="flex justify-between text-xs text-gray-400">
                            <span>Progress</span>
                            <span>{progress.toFixed(0)}%</span>
                        </div>
                        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-cyan-400"
                                initial={{ width: 0 }}
                                animate={{ width: `${progress}%` }}
                            />
                        </div>
                    </div>
                )}

                {/* Result */}
                {status === "COMPLETE" && finalModelLink && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20"
                    >
                        <div className="flex items-center gap-2 mb-2">
                            <Award className="text-emerald-400" size={20} />
                            <span className="font-bold text-emerald-400">Training Complete!</span>
                        </div>
                        <a
                            href={`${API_BASE}${finalModelLink}`}
                            className="text-xs text-emerald-300 underline block"
                            download
                        >
                            Download Trained Model
                        </a>
                    </motion.div>
                )}
            </div>

            {/* Visualisation Chart */}
            <div className="glass-panel p-6 rounded-2xl border border-white/10 lg:col-span-2 flex flex-col min-h-[400px]">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="font-bold text-gray-200">Live Training Metrics</h3>
                    <div className="flex gap-4 text-xs">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-cyan-400" />
                            <span className="text-gray-400">Validation Accuracy</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-red-400" />
                            <span className="text-gray-400">Training Loss</span>
                        </div>
                    </div>
                </div>

                <div className="flex-1 w-full h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={history}>
                            <defs>
                                <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#f87171" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                            <XAxis dataKey="epoch" stroke="#666" />
                            <YAxis yAxisId="left" stroke="#666" domain={[0, 1]} />
                            <YAxis yAxisId="right" stroke="#666" orientation="right" />
                            <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }} />

                            <Area
                                yAxisId="left"
                                type="monotone"
                                dataKey="accuracy"
                                stroke="#06b6d4"
                                fill="url(#colorAcc)"
                                strokeWidth={2}
                                name="Accuracy"
                            />
                            <Area
                                yAxisId="right"
                                type="monotone"
                                dataKey="loss"
                                stroke="#f87171"
                                fill="url(#colorLoss)"
                                strokeWidth={2}
                                name="Loss"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};
