import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert } from 'lucide-react';

interface HolographicOverlayProps {
    active: boolean;
}

export const HolographicOverlay: React.FC<HolographicOverlayProps> = ({ active }) => {
    return (
        <AnimatePresence>
            {active && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[100] pointer-events-none overflow-hidden"
                >
                    {/* Scanline Effect */}
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-10 bg-[length:100%_2px,3px_100%]" />

                    {/* Red Vignette */}
                    <div className="absolute inset-0 bg-radial-gradient from-transparent via-transparent to-rose-900/30" />

                    {/* Glitch Frame */}
                    <motion.div
                        animate={{
                            borderColor: ['rgba(244,63,94,0.1)', 'rgba(244,63,94,0.5)', 'rgba(244,63,94,0.1)'],
                            borderWidth: [2, 10, 2]
                        }}
                        transition={{ duration: 0.2, repeat: Infinity }}
                        className="absolute inset-10 border-rose-500/20 rounded-3xl"
                    />

                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-6 text-center">
                        <motion.div
                            animate={{ scale: [1, 1.1, 1] }}
                            transition={{ duration: 2, repeat: Infinity }}
                            className="p-8 bg-rose-500/10 rounded-full border border-rose-500/50 shadow-[0_0_50px_rgba(244,63,94,0.4)] backdrop-blur-md"
                        >
                            <ShieldAlert size={80} className="text-rose-500" />
                        </motion.div>

                        <div className="space-y-2">
                            <motion.h2
                                animate={{ opacity: [0.8, 1, 0.8] }}
                                transition={{ duration: 2, repeat: Infinity }}
                                className="text-6xl font-black text-rose-500 uppercase italic tracking-tighter"
                            >
                                Threat Detected
                            </motion.h2>
                            <p className="text-rose-300 font-mono text-sm tracking-widest uppercase bg-rose-900/40 px-4 py-1 rounded">
                                Integrity Violation: Pattern Drift Exceeded 90%
                            </p>
                        </div>
                    </div>

                    {/* Corner Elements */}
                    <div className="absolute top-10 left-10 w-20 h-20 border-l-4 border-t-4 border-rose-500/50" />
                    <div className="absolute top-10 right-10 w-20 h-20 border-r-4 border-t-4 border-rose-500/50" />
                    <div className="absolute bottom-10 left-10 w-20 h-20 border-l-4 border-b-4 border-rose-500/50" />
                    <div className="absolute bottom-10 right-10 w-20 h-20 border-r-4 border-b-4 border-rose-500/50" />
                </motion.div>
            )}
        </AnimatePresence>
    );
};
