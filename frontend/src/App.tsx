import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';
import { Dashboard } from './pages/Dashboard';
import { RealTimeMetrics } from './pages/RealTimeMetrics';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="events" element={<div className="p-4">Events Page (Placeholder)</div>} />
          <Route path="metrics" element={<RealTimeMetrics />} />
          <Route path="settings" element={<div className="p-4">Settings (Placeholder)</div>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
