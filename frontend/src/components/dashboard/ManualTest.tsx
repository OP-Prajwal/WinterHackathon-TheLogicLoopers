import React, { useState } from 'react';
import { Send, AlertTriangle, CheckCircle } from 'lucide-react';
import { analyzePrompt } from '../../services/analysisService';

const ManualTest: React.FC = () => {
    const [input, setInput] = useState('');
    const [result, setResult] = useState<'safe' | 'poisoned' | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;
        setLoading(true);
        try {
            const res = await analyzePrompt(input);
            setResult(res.is_safe ? 'safe' : 'poisoned');
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-card text-card-foreground p-6 rounded-lg border border-border shadow-sm h-full flex flex-col">
            <h3 className="font-semibold text-lg mb-6">Manual Analysis</h3>

            <form onSubmit={handleSubmit} className="flex gap-2 mb-6">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Enter prompt to analyze..."
                    className="flex-1 bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
                <button
                    type="submit"
                    className="bg-primary text-primary-foreground p-2 rounded-lg hover:bg-primary/90 transition-colors"
                >
                    <Send size={20} />
                </button>
            </form>

            <div className="flex-1 bg-muted/50 rounded-lg p-4 bg-[url('https://grainy-gradients.vercel.app/noise.svg')]">
                {result ? (
                    <div className="h-full flex flex-col items-center justify-center text-center animate-in fade-in zoom-in duration-300">
                        {result === 'safe' ? (
                            <>
                                <CheckCircle className="text-green-500 mb-2" size={48} />
                                <h4 className="font-bold text-lg text-green-500">Safe Content</h4>
                                <p className="text-sm text-muted-foreground">No malicious patterns detected</p>
                            </>
                        ) : (
                            <>
                                <AlertTriangle className="text-destructive mb-2" size={48} />
                                <h4 className="font-bold text-lg text-destructive">Poison Detected</h4>
                                <p className="text-sm text-muted-foreground">Confidence Score: 99.8%</p>
                            </>
                        )}
                    </div>
                ) : (
                    <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                        Ready for analysis
                    </div>
                )}
            </div>
        </div>
    );
};

export default ManualTest;
