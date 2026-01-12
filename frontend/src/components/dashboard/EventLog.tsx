import React from 'react';
import { Clock, AlertCircle, CheckCircle2, Info } from 'lucide-react';


const events = [
    { id: 1, type: 'warning', message: 'Unauthorized access attempt detected', time: '2 mins ago' },
    { id: 2, type: 'success', message: 'System backup completed successfully', time: '15 mins ago' },
    { id: 3, type: 'info', message: 'User "Admin" updated firewall rules', time: '1 hour ago' },
    { id: 4, type: 'warning', message: 'High CPU usage detected (85%)', time: '2 hours ago' },
    { id: 5, type: 'success', message: 'Database optimization routine finished', time: '4 hours ago' },
];

const EventLog: React.FC = () => {
    return (
        <div className="bg-card text-card-foreground p-6 rounded-lg border border-border shadow-sm h-full">
            <div className="flex items-center justify-between mb-6">
                <h3 className="font-semibold text-lg">Event Log</h3>
                <button className="text-sm text-primary hover:underline">View All</button>
            </div>

            <div className="space-y-4">
                {events.map((event) => (
                    <div key={event.id} className="flex gap-3 pb-4 border-b border-border last:border-0 last:pb-0">
                        <div className="mt-1">
                            {event.type === 'warning' && <AlertCircle size={16} className="text-yellow-500" />}
                            {event.type === 'success' && <CheckCircle2 size={16} className="text-green-500" />}
                            {event.type === 'info' && <Info size={16} className="text-blue-500" />}
                        </div>
                        <div className="flex-1">
                            <p className="text-sm font-medium">{event.message}</p>
                            <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
                                <Clock size={10} />
                                <span>{event.time}</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default EventLog;
