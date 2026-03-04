import { Link, useLocation } from 'react-router-dom';
import { Home, ListTodo, Settings as SettingsIcon, Users, Menu, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import api from '../lib/api';

interface QueueStatus {
    running: number;
    queued: number;
    paused: number;
    total_active: number;
}

export function Layout({ children }: { children: React.ReactNode }) {
    const location = useLocation();
    const isHome = location.pathname === '/';
    const isJobs = location.pathname === '/jobs';
    const isSpeakers = location.pathname === '/speakers';
    const isSettings = location.pathname === '/settings';
    const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);
    const [mobileNavOpen, setMobileNavOpen] = useState(false);

    useEffect(() => {
        setMobileNavOpen(false);
    }, [location.pathname]);

    useEffect(() => {
        const fetchQueueStatus = async () => {
            try {
                const res = await api.get<QueueStatus>('/jobs/status');
                setQueueStatus(res.data);
            } catch (e) {
                // Ignore errors - backend might not be running
            }
        };

        fetchQueueStatus();
        const interval = setInterval(fetchQueueStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex h-screen overflow-hidden bg-slate-50">
            {/* Ambient Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 -right-40 w-96 h-96 bg-gradient-to-br from-blue-400/20 to-indigo-500/20 rounded-full blur-3xl" />
                <div className="absolute top-1/2 -left-40 w-80 h-80 bg-gradient-to-br from-indigo-400/15 to-purple-500/15 rounded-full blur-3xl" />
            </div>

            {/* Mobile Top Bar */}
            <div className="md:hidden absolute top-0 left-0 right-0 z-20 px-3 pt-3">
                <div className="glass-sidebar rounded-2xl px-3 py-2.5 flex items-center justify-between border border-white/40 shadow-sm">
                    <Link to="/" className="min-w-0">
                        <div className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 truncate">
                            Chatalogue
                        </div>
                        <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Admin Console</div>
                    </Link>
                    <button
                        type="button"
                        onClick={() => setMobileNavOpen(v => !v)}
                        className="p-2 rounded-xl bg-white/80 border border-slate-200 text-slate-600 hover:text-slate-800 hover:bg-white shadow-sm"
                        aria-label={mobileNavOpen ? 'Close navigation menu' : 'Open navigation menu'}
                        aria-expanded={mobileNavOpen}
                    >
                        {mobileNavOpen ? <X size={18} /> : <Menu size={18} />}
                    </button>
                </div>
            </div>

            {/* Mobile Sidebar Overlay */}
            {mobileNavOpen && (
                <div
                    className="md:hidden absolute inset-0 z-20 bg-slate-900/30 backdrop-blur-[1px]"
                    onClick={() => setMobileNavOpen(false)}
                />
            )}

            {/* Sidebar */}
            <aside
                className={`fixed md:relative top-0 left-0 z-30 md:z-10 w-72 glass-sidebar flex flex-col h-full transition-transform duration-300 md:translate-x-0 ${
                    mobileNavOpen ? 'translate-x-0' : '-translate-x-full'
                }`}
            >
                <div className="md:hidden h-3" />
                <div className="p-8 pb-4">
                    <Link to="/" className="block">
                        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 tracking-tight">
                            Chatalogue
                        </h1>
                        <p className="text-xs text-slate-400 mt-1 uppercase tracking-wider font-semibold">Admin Console</p>
                    </Link>
                </div>

                <nav className="flex-1 px-4 space-y-2 mt-4">
                    <Link
                        to="/"
                        className={`group flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 ${isHome
                            ? 'bg-blue-50/80 text-blue-600 shadow-sm ring-1 ring-blue-100 font-medium'
                            : 'text-slate-500 hover:text-slate-800 hover:bg-white/50'
                            }`}
                    >
                        <Home
                            size={20}
                            className={`transition-colors duration-200 ${isHome ? 'text-blue-500' : 'text-slate-400 group-hover:text-slate-600'}`}
                        />
                        All Channels
                    </Link>

                    <Link
                        to="/jobs"
                        className={`group flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 ${isJobs
                            ? 'bg-blue-50/80 text-blue-600 shadow-sm ring-1 ring-blue-100 font-medium'
                            : 'text-slate-500 hover:text-slate-800 hover:bg-white/50'
                            }`}
                    >
                        <div className="relative">
                            <ListTodo
                                size={20}
                                className={`transition-colors duration-200 ${isJobs ? 'text-blue-500' : 'text-slate-400 group-hover:text-slate-600'}`}
                            />
                            {queueStatus && queueStatus.running > 0 && (
                                <>
                                    <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-green-500" />
                                    <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-green-400 animate-ping" />
                                </>
                            )}
                        </div>
                        <span className="truncate">Job Queue</span>
                        {queueStatus && queueStatus.running > 0 && (
                            <span className="ml-auto px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide rounded-full bg-green-100 text-green-700 border border-green-200">
                                Live
                            </span>
                        )}
                        {queueStatus && queueStatus.total_active > 0 && (
                            <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${queueStatus.running > 0
                                ? 'bg-blue-100 text-blue-700'
                                : 'bg-amber-100 text-amber-700'
                                }`}>
                                {queueStatus.total_active}
                            </span>
                        )}
                    </Link>

                    <Link
                        to="/speakers"
                        className={`group flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 ${isSpeakers
                            ? 'bg-blue-50/80 text-blue-600 shadow-sm ring-1 ring-blue-100 font-medium'
                            : 'text-slate-500 hover:text-slate-800 hover:bg-white/50'
                            }`}
                    >
                        <Users
                            size={20}
                            className={`transition-colors duration-200 ${isSpeakers ? 'text-blue-500' : 'text-slate-400 group-hover:text-slate-600'}`}
                        />
                        Speakers
                    </Link>
                </nav>

                <div className="px-4 pb-2">
                    <Link
                        to="/settings"
                        className={`group flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 ${isSettings
                            ? 'bg-blue-50/80 text-blue-600 shadow-sm ring-1 ring-blue-100 font-medium'
                            : 'text-slate-500 hover:text-slate-800 hover:bg-white/50'
                            }`}
                    >
                        <SettingsIcon
                            size={20}
                            className={`transition-colors duration-200 ${isSettings ? 'text-blue-500' : 'text-slate-400 group-hover:text-slate-600'}`}
                        />
                        Settings
                    </Link>
                </div>
                <div className="p-4 pt-0">
                    <div className="bg-slate-50/50 rounded-xl p-4 text-xs text-slate-400 border border-slate-100/50">
                        <p>System Status: <span className="text-green-500 font-medium">Online</span></p>
                        <p className="mt-1">v0.1.0-alpha</p>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 min-w-0 overflow-y-auto relative z-10 px-3 pb-6 pt-20 md:p-8 md:pt-8">
                {children}
            </main>
        </div >
    );
}
