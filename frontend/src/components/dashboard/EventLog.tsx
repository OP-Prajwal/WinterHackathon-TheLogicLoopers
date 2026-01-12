import React from 'react';
import { AlertCircle, Info, ShieldAlert } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { EventData } from '../../services/websocket';
import classes from './EventLog.module.css';

interface EventLogProps {
    events: EventData[];
}

export const EventLog: React.FC<EventLogProps> = ({ events }) => {
    return (
        <div className={classes.container}>
            <h3 className={classes.heading}>Security Event Log</h3>

            <div className={classes.list}>
                <AnimatePresence>
                    {events.map((event, index) => (
                        <motion.div
                            key={`${event.batch}-${index}`} // unique key
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0 }}
                            className={classes.item}
                            data-severity={event.severity}
                        >
                            <div className={classes.iconWrapper}>
                                {event.severity === 'danger' && <ShieldAlert size={16} />}
                                {event.severity === 'warning' && <AlertCircle size={16} />}
                                {event.severity === 'info' && <Info size={16} />}
                            </div>
                            <div className={classes.content}>
                                <div className={classes.message}>{event.message}</div>
                                <div className={classes.meta}>Batch {event.batch}</div>
                            </div>
                        </motion.div>
                    ))}
                    {events.length === 0 && (
                        <div className={classes.empty}>No events recorded</div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};
