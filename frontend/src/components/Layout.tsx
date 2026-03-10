import { Link, useLocation } from 'react-router-dom';
import { Home, ListTodo, Settings as SettingsIcon, Users, Menu, X, RefreshCw, Download } from 'lucide-react';
import { useEffect, useState } from 'react';
import api from '../lib/api';

interface QueueStatus {
    running: number;
    queued: number;
    paused: number;
    total_active: number;
}

interface SystemVersionInfo {
    status?: string;
    app_version?: string | null;
    git?: {
        branch?: string | null;
        head_short?: string | null;
        update_available?: boolean;
        behind_count?: number;
        error?: string | null;
    };
}

export function Layout({ children }: { children: React.ReactNode }) {
    const location = useLocation();
    const isHome = location.pathname === '/';
    const isJobs = location.pathname === '/jobs';
    const isSpeakers = location.pathname === '/speakers';
    const isSettings = location.pathname === '/settings';
    const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);
    const [mobileNavOpen, setMobileNavOpen] = useState(false);
    const [backendOnline, setBackendOnline] = useState(false);
    const [versionInfo, setVersionInfo] = useState<SystemVersionInfo | null>(null);
    const [checkingVersion, setCheckingVersion] = useState(false);
    const [updating, setUpdating] = useState(false);
    const [updateMessage, setUpdateMessage] = useState<string | null>(null);

    useEffect(() => {
        setMobileNavOpen(false);
    }, [location.pathname]);

    useEffect(() => {
        const fetchQueueStatus = async () => {
            try {
                const res = await api.get<QueueStatus>('/jobs/status');
                setQueueStatus(res.data);
                setBackendOnline(true);
            } catch (e) {
                // Ignore errors - backend might not be running
                setBackendOnline(false);
            }
        };

        fetchQueueStatus();
        const interval = setInterval(fetchQueueStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        const fetchVersionInfo = async (checkRemote = true) => {
            setCheckingVersion(true);
            try {
                const res = await api.get<SystemVersionInfo>('/system/version', { params: { check_remote: checkRemote } });
                setVersionInfo(res.data);
            } catch {
                // Keep existing version panel info if request fails.
            } finally {
                setCheckingVersion(false);
            }
        };

        fetchVersionInfo(true);
        const interval = setInterval(() => {
            void fetchVersionInfo(true);
        }, 120000);
        return () => clearInterval(interval);
    }, []);

    const handleCheckUpdates = async () => {
        setUpdateMessage(null);
        setCheckingVersion(true);
        try {
            const res = await api.get<SystemVersionInfo>('/system/version', { params: { check_remote: true } });
            setVersionInfo(res.data);
            const available = Boolean(res.data?.git?.update_available);
            setUpdateMessage(available ? 'Update available.' : 'Already up to date.');
        } catch (e: any) {
            setUpdateMessage(String(e?.response?.data?.detail || 'Failed to check updates.'));
        } finally {
            setCheckingVersion(false);
        }
    };

    const handleUpdateAndRestart = async () => {
        if (updating) return;
        setUpdating(true);
        setUpdateMessage('Updating and restarting backend...');
        try {
            await api.post('/system/update');
        } catch (e: any) {
            setUpdating(false);
            setUpdateMessage(String(e?.response?.data?.detail || 'Update failed.'));
            return;
        }

        // Wait for backend to come back and refresh version data.
        for (let i = 0; i < 30; i++) {
            await new Promise((r) => setTimeout(r, 2000));
            try {
                const res = await api.get<SystemVersionInfo>('/system/version', { params: { check_remote: true } });
                setVersionInfo(res.data);
                setUpdateMessage('Updated and restarted.');
                setUpdating(false);
                return;
            } catch {
                // keep polling while backend restarts
            }
        }

        setUpdating(false);
        setUpdateMessage('Restart timed out. Check backend logs and refresh.');
    };

    const versionLabel = versionInfo?.app_version
        || versionInfo?.git?.head_short
        || 'unknown';
    const branchLabel = versionInfo?.git?.branch || 'main';
    const updateAvailable = Boolean(versionInfo?.git?.update_available);
    const behindCount = Number(versionInfo?.git?.behind_count || 0);

    return (
        <div className="flex h-screen overflow-hidden bg-slate-50">
            {/* Ambient Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 -right-40 w-96 h-96 bg-gradient-to-br from-rose-400/20 to-red-500/20 rounded-full blur-3xl" />
                <div className="absolute top-1/2 -left-40 w-80 h-80 bg-gradient-to-br from-rose-300/15 to-pink-500/15 rounded-full blur-3xl" />
            </div>

            {/* Mobile Top Bar */}
            <div className="md:hidden absolute top-0 left-0 right-0 z-20 px-3 pt-3">
                <div className="glass-sidebar rounded-2xl px-3 py-2.5 flex items-center justify-between border border-white/40 shadow-sm">
                    <Link to="/" className="min-w-0 flex items-center gap-2.5">
                        <img src="/chatalogue-logo.svg" alt="Chatalogue logo" className="h-8 w-8 shrink-0 drop-shadow-sm" />
                        <div className="min-w-0">
                            <div className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-rose-600 to-red-700 truncate">
                                Chatalogue
                            </div>
                            <div className="text-[10px] text-slate-400 tracking-wide font-semibold">Dialogue → Data</div>
                        </div>
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
                    <Link to="/" className="flex items-center gap-3">
                        <img src="/chatalogue-logo.svg" alt="Chatalogue logo" className="h-9 w-9 shrink-0 drop-shadow-sm" />
                        <div className="min-w-0">
                            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-rose-600 to-red-700 tracking-tight truncate">
                                Chatalogue
                            </h1>
                            <p className="text-xs text-slate-400 mt-1 tracking-wide font-semibold">Dialogue → Data</p>
                        </div>
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
                        <p>
                            System Status:{' '}
                            <span className={backendOnline ? 'text-green-500 font-medium' : 'text-red-500 font-medium'}>
                                {backendOnline ? 'Online' : 'Offline'}
                            </span>
                        </p>
                        <p className="mt-1">
                            {branchLabel} • {versionLabel}
                        </p>
                        <div className="mt-2 flex items-center gap-2">
                            {updateAvailable ? (
                                <span className="text-amber-600 font-medium">
                                    Update available{behindCount > 0 ? ` (${behindCount})` : ''}
                                </span>
                            ) : (
                                <span className="text-slate-500">{versionInfo?.git?.error ? 'Update check unavailable' : 'Up to date'}</span>
                            )}
                        </div>
                        <div className="mt-2 flex items-center gap-2">
                            <button
                                type="button"
                                onClick={handleCheckUpdates}
                                disabled={checkingVersion || updating}
                                className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-100 disabled:opacity-50"
                                title="Check for updates"
                            >
                                <RefreshCw size={11} className={checkingVersion ? 'animate-spin' : ''} />
                                Check
                            </button>
                            <button
                                type="button"
                                onClick={handleUpdateAndRestart}
                                disabled={updating || checkingVersion || !updateAvailable}
                                className="inline-flex items-center gap-1 rounded-md border border-blue-200 bg-blue-50 px-2 py-1 text-[11px] text-blue-700 hover:bg-blue-100 disabled:opacity-50"
                                title="Pull latest code and restart backend"
                            >
                                <Download size={11} />
                                {updating ? 'Updating...' : 'Update + Restart'}
                            </button>
                        </div>
                        {updateMessage && (
                            <p className="mt-2 text-[11px] text-slate-500">{updateMessage}</p>
                        )}
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
