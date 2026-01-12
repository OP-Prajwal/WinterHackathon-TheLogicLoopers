import React, { useState } from 'react';
import { Sparkles, RefreshCw, CheckCheck } from 'lucide-react';
import clsx from 'clsx';

const PurificationPanel: React.FC = () => {
    const [isPurifying, setIsPurifying] = useState(false);
    const [progress, setProgress] = useState(0);

    const startPurification = () => {
        setIsPurifying(true);
        setProgress(0);
        const interval = setInterval(() => {
            setProgress((prev) => {
                if (prev >= 100) {
                    clearInterval(interval);
                    setIsPurifying(false);
                    return 100;
                }
                return prev + 10;
            });
        }, 500);
    };

    return (
        <div className="bg-card text-card-foreground p-6 rounded-lg border border-border shadow-sm h-full">
            <div className="flex items-center gap-2 mb-6">
                <Sparkles className="text-primary" size={24} />
                <h3 className="font-semibold text-lg">System Purification</h3>
            </div>

            <div className="space-y-6">
                <div className="p-4 bg-muted/50 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Model Integrity</span>
                        <span className="text-sm text-green-500 font-bold">98% (Secure)</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{ width: '98%' }} />
                    </div>
                </div>

                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span>Purification Status</span>
                        <span>{isPurifying ? `${progress}%` : 'Ready'}</span>
                    </div>
                    {isPurifying && (
                        <div className="w-full bg-muted rounded-full h-2">
                            <div
                                className="bg-primary h-2 rounded-full transition-all duration-300"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    )}
                </div>

                <button
                    onClick={startPurification}
                    disabled={isPurifying}
                    className={clsx(
                        "w-full py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors",
                        isPurifying
                            ? "bg-muted text-muted-foreground cursor-not-allowed"
                            : "bg-primary text-primary-foreground hover:bg-primary/90"
                    )}
                >
                    {isPurifying ? (
                        <><RefreshCw className="animate-spin" size={18} /> Purifying...</>
                    ) : (
                        <><CheckCheck size={18} /> Start Purification</>
                    )}
                </button>
            </div>
        </div>
    );
};

export default PurificationPanel;
