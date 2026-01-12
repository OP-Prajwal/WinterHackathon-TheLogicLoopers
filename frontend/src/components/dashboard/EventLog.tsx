import React from 'react';
import { AlertCircle, Info, ShieldAlert, History } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { EventData } from '../../services/websocket';
import clsx from 'clsx';

interface EventLogProps {
    events: EventData[];
}

export const EventLog: React.FC<EventLogProps> = ({ events }) => {
    return (
        <div className="glass-panel p-6 overflow-hidden flex flex-col h-full bg-dark-800/40">
            <div className="flex items-center gap-2 mb-4 border-b border-white/5 pb-2">
                <History size={18} className="text-cyan-400" />
                <h3 className="text-lg font-bold text-gray-200">Security Event Log</h3>
            </div>

            <div className="flex-1 overflow-y-auto pr-2 max-h-[300px]" style={{ scrollbarWidth: 'thin', scrollbarColor: '#374151 #1f2937' }}>
                <AnimatePresence mode='popLayout'>
                    {events.length > 0 ? (
                        events.slice().reverse().map((event, index) => (
                            <motion.div
                                key={`${event.batch}-${index}`} // unique key
                                initial={{ opacity: 0, x: -20, height: 0 }}
                                animate={{ opacity: 1, x: 0, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className={clsx(
                                    "mb-3 p-3 rounded-lg border flex items-start gap-3 text-sm relative overflow-hidden",
                                    event.severity === 'danger' && "bg-rose-500/10 border-rose-500/30 text-rose-200",
                                    event.severity === 'warning' && "bg-amber-500/10 border-amber-500/30 text-amber-200",
                                    event.severity === 'info' && "bg-blue-500/10 border-blue-500/30 text-blue-200"
                                )}
                            >
                                <div className="mt-1 shrink-0">
                                    {event.severity === 'danger' && <ShieldAlert size={16} />}
                                    {event.severity === 'warning' && <AlertCircle size={16} />}
                                    {event.severity === 'info' && <Info size={16} />}
                                </div>
                                <div>
                                    <div className="font-medium">{event.message}</div>
                                    <div className="text-xs opacity-60 mt-1 font-mono">Batch ID: {event.batch}</div>
                                </div>
                            </motion.div>
                        ))
                    ) : (
                        <div className="flex flex-col items-center justify-center h-40 text-gray-600">
                            <History size={32} className="mb-2 opacity-50" />
                            <p>No events recorded</p>
                        </div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};
