import { Link, useLocation } from 'react-router-dom';
import { Home, ListTodo, Settings as SettingsIcon, Users, Menu, X, RefreshCw, Download, AlertTriangle } from 'lucide-react';
import { useEffect, useState, useCallback } from 'react';
import api from '../lib/api';

interface QueueStatus {
    running: number;
    queued: number;
    paused: number;
    total_active: number;
}

interface CudaHealth {
    device: string;
    cuda_unhealthy: boolean;
    cuda_unhealthy_reason: string | null;
    cuda_recovery_pending: boolean;
    permanent_cpu_mode: boolean;
    auto_restart_count: number;
    auto_restart_limit: number;
    cuda_fault_count_this_worker: number;
    memory?: {
        free_gb?: number;
        total_gb?: number;
        allocated_gb?: number;
        reserved_gb?: number;
    };
    system_memory?: {
        rss_gb?: number;
        total_gb?: number;
        available_gb?: number;
    };
    component_memory?: Record<string, {
        loaded?: boolean;
        ram_gb?: number;
        vram_gb?: number;
    }>;
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
    const pathname = location.pathname;
    const matchesSection = (prefixes: string[]) => prefixes.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
    const isHome = pathname === '/' || matchesSection(['/channel', '/video']);
    const isJobs = matchesSection(['/jobs']);
    const isSpeakers = matchesSection(['/speakers', '/avatars']);
    const isSettings = matchesSection(['/settings']);
    const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);
    const [mobileNavOpen, setMobileNavOpen] = useState(false);
    const [backendOnline, setBackendOnline] = useState(false);
    const [versionInfo, setVersionInfo] = useState<SystemVersionInfo | null>(null);
    const [checkingVersion, setCheckingVersion] = useState(false);
    const [updating, setUpdating] = useState(false);
    const [updateMessage, setUpdateMessage] = useState<string | null>(null);
    const [cudaHealth, setCudaHealth] = useState<CudaHealth | null>(null);
    const [retryingGpu, setRetryingGpu] = useState(false);

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

    useEffect(() => {
        if (!backendOnline) return;
        const fetchCudaHealth = async () => {
            try {
                const res = await api.get<CudaHealth>('/system/cuda-health');
                setCudaHealth(res.data);
            } catch { /* ignore */ }
        };
        fetchCudaHealth();
        const interval = setInterval(fetchCudaHealth, 15000);
        return () => clearInterval(interval);
    }, [backendOnline]);

    const handleRetryGpu = useCallback(async () => {
        if (retryingGpu) return;
        setRetryingGpu(true);
        try {
            await api.post('/system/cuda-restart-state/reset');
            await api.post('/system/restart');
            // Poll until backend comes back
            for (let i = 0; i < 20; i++) {
                await new Promise((r) => setTimeout(r, 2000));
                try {
                    const res = await api.get<CudaHealth>('/system/cuda-health');
                    setCudaHealth(res.data);
                    break;
                } catch { /* keep polling */ }
            }
        } catch { /* ignore */ } finally {
            setRetryingGpu(false);
        }
    }, [retryingGpu]);

    const cudaBannerSeverity = cudaHealth?.permanent_cpu_mode
        ? 'red'
        : (cudaHealth?.auto_restart_count ?? 0) > 0
            ? 'orange'
            : cudaHealth?.cuda_unhealthy
                ? 'yellow'
                : null;

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

    const memoryComponents = [
        { key: 'parakeet', label: 'Parakeet', color: 'bg-violet-500' },
        { key: 'whisper', label: 'Whisper', color: 'bg-cyan-500' },
        { key: 'pyannote', label: 'Pyannote', color: 'bg-emerald-500' },
        { key: 'avatar_training', label: 'Avatar Trainer', color: 'bg-amber-500' },
    ] as const;

    const renderMemoryBar = (
        label: string,
        totalGb: number | null,
        values: { key: string; label: string; color: string; valueGb: number }[],
        usedGb?: number | null,
        trailing?: string | null,
    ) => {
        if (totalGb == null || totalGb <= 0) return null;
        const componentUsed = values.reduce((sum, item) => sum + Math.max(0, item.valueGb || 0), 0);
        const effectiveUsed = Math.max(0, Math.min(totalGb, Math.max(componentUsed, usedGb ?? 0)));
        const otherUsed = Math.max(0, effectiveUsed - componentUsed);
        const barSegments = [
            ...values,
            ...(otherUsed > 0.05 ? [{ key: 'other', label: 'Other', color: 'bg-slate-300', valueGb: otherUsed }] : []),
        ];
        return (
            <div className="mt-3">
                <div className="flex items-center justify-between text-[11px] text-slate-500">
                    <span className="font-medium text-slate-600">{label}</span>
                    <span>{trailing || `${effectiveUsed.toFixed(1)} / ${totalGb.toFixed(1)} GB`}</span>
                </div>
                <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200">
                    <div className="flex h-full w-full">
                        {barSegments.map((item) => {
                            const widthPct = Math.max(0, Math.min(100, (item.valueGb / totalGb) * 100));
                            if (widthPct <= 0.05) return null;
                            return <div key={item.key} className={`h-full shrink-0 ${item.color}`} style={{ width: `${widthPct}%` }} />;
                        })}
                    </div>
                </div>
            </div>
        );
    };

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
                        {cudaHealth && (() => {
                            const componentMemory = cudaHealth.component_memory || {};
                            const ramTotal = typeof cudaHealth.system_memory?.total_gb === 'number' ? cudaHealth.system_memory.total_gb : null;
                            const vramTotal = typeof cudaHealth.memory?.total_gb === 'number' ? cudaHealth.memory.total_gb : null;
                            const ramUsed = typeof cudaHealth.system_memory?.rss_gb === 'number' ? cudaHealth.system_memory.rss_gb : null;
                            const vramAllocated = typeof cudaHealth.memory?.allocated_gb === 'number' ? cudaHealth.memory.allocated_gb : null;
                            const ramValues = memoryComponents.map((component) => ({
                                ...component,
                                valueGb: Number(componentMemory[component.key]?.ram_gb || 0),
                            }));
                            const vramValues = memoryComponents.map((component) => ({
                                ...component,
                                valueGb: Number(componentMemory[component.key]?.vram_gb || 0),
                            }));
                            return (
                                <>
                                    {renderMemoryBar(
                                        'RAM',
                                        ramTotal,
                                        ramValues,
                                        ramUsed,
                                        ramUsed != null && ramTotal != null ? `${ramUsed.toFixed(1)} / ${ramTotal.toFixed(1)} GB` : null,
                                    )}
                                    {renderMemoryBar(
                                        'VRAM',
                                        vramTotal,
                                        vramValues,
                                        vramAllocated,
                                        vramAllocated != null && vramTotal != null ? `${vramAllocated.toFixed(1)} / ${vramTotal.toFixed(1)} GB` : null,
                                    )}
                                    <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate-500">
                                        {memoryComponents.map((component) => {
                                            const ramGb = Number(componentMemory[component.key]?.ram_gb || 0);
                                            const vramGb = Number(componentMemory[component.key]?.vram_gb || 0);
                                            const loaded = Boolean(componentMemory[component.key]?.loaded) || ramGb > 0 || vramGb > 0;
                                            return (
                                                <span key={component.key} className="inline-flex items-center gap-1">
                                                    <span className={`h-2 w-2 rounded-full ${component.color}`} />
                                                    {component.label}{loaded ? ` ${Math.max(ramGb, vramGb).toFixed(1)} GB` : ' idle'}
                                                </span>
                                            );
                                        })}
                                    </div>
                                </>
                            );
                        })()}
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 min-w-0 overflow-y-auto relative z-10 px-3 pb-6 pt-16 md:p-8 md:pt-8">
                {cudaBannerSeverity && (
                    <div className={`mb-4 rounded-xl border px-4 py-3 flex items-start gap-3 text-sm ${
                        cudaBannerSeverity === 'red'
                            ? 'bg-red-50 border-red-200 text-red-800'
                            : cudaBannerSeverity === 'orange'
                                ? 'bg-orange-50 border-orange-200 text-orange-800'
                                : 'bg-amber-50 border-amber-200 text-amber-800'
                    }`}>
                        <AlertTriangle size={18} className="mt-0.5 shrink-0" />
                        <div className="flex-1 min-w-0">
                            {cudaBannerSeverity === 'red' ? (
                                <p>
                                    <span className="font-semibold">GPU disabled</span> after repeated recovery failures. Processing will use CPU (slower).
                                </p>
                            ) : cudaBannerSeverity === 'orange' ? (
                                <p>
                                    <span className="font-semibold">GPU error detected.</span> Backend auto-restarting (attempt {cudaHealth!.auto_restart_count}/{cudaHealth!.auto_restart_limit})...
                                </p>
                            ) : (
                                <p>
                                    <span className="font-semibold">GPU error detected.</span>{' '}
                                    {cudaHealth?.cuda_recovery_pending ? 'Auto-recovery in progress...' : 'Running on CPU fallback.'}
                                </p>
                            )}
                        </div>
                        {cudaBannerSeverity === 'red' && (
                            <button
                                type="button"
                                onClick={handleRetryGpu}
                                disabled={retryingGpu}
                                className="shrink-0 inline-flex items-center gap-1.5 rounded-lg border border-red-300 bg-white px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
                            >
                                <RefreshCw size={12} className={retryingGpu ? 'animate-spin' : ''} />
                                {retryingGpu ? 'Restarting...' : 'Retry GPU'}
                            </button>
                        )}
                    </div>
                )}
                {children}
            </main>
        </div >
    );
}
