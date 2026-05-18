import { useCallback, useEffect, useState } from 'react';

import api from '../../lib/api';

export interface ExternalShareAuditEntry {
    at: string;
    action: string;
    allowed: boolean;
    reason?: string | null;
    client_ip?: string | null;
    path?: string | null;
}

export interface ExternalShareStatus {
    active: boolean;
    mode: string;
    enable_tunnel: boolean;
    tunnel_provider?: string | null;
    started_at?: string | null;
    expires_at?: string | null;
    frontend_local_url?: string | null;
    api_local_url?: string | null;
    frontend_lan_url?: string | null;
    api_lan_url?: string | null;
    frontend_public_url?: string | null;
    api_public_url?: string | null;
    share_url?: string | null;
    token_required: boolean;
    password_required: boolean;
    ip_allowlist: string[];
    cloudflared_available: boolean;
    audit_log_path?: string | null;
    audit_entries: ExternalShareAuditEntry[];
}

export interface CloudflaredInstallInfo {
    platform: string;
    package_manager: string;
    package_manager_available: boolean;
    download_url: string;
    installed: boolean;
}

export function useExternalShare() {
    const [externalShareStatus, setExternalShareStatus] = useState<ExternalShareStatus | null>(null);
    const [loadingExternalShareStatus, setLoadingExternalShareStatus] = useState(false);
    const [startingExternalShare, setStartingExternalShare] = useState(false);
    const [stoppingExternalShare, setStoppingExternalShare] = useState(false);
    const [copiedExternalShareUrl, setCopiedExternalShareUrl] = useState(false);
    const [externalShareTunnelEnabled, setExternalShareTunnelEnabled] = useState(true);
    const [externalShareFrontendPort, setExternalShareFrontendPort] = useState<number>(() => {
        if (typeof window === 'undefined') return 5173;
        return Number(window.location.port || 5173) || 5173;
    });
    const [externalShareBackendPort, setExternalShareBackendPort] = useState(8011);
    const [externalShareDurationHours, setExternalShareDurationHours] = useState(1);
    const [externalSharePassword, setExternalSharePassword] = useState('');
    const [externalShareIpAllowlist, setExternalShareIpAllowlist] = useState('');
    const [cloudflaredInstallInfo, setCloudflaredInstallInfo] = useState<CloudflaredInstallInfo | null>(null);
    const [installingCloudflared, setInstallingCloudflared] = useState(false);

    const loadExternalShareStatus = useCallback(async (silent = false) => {
        if (!silent) setLoadingExternalShareStatus(true);
        try {
            const res = await api.get<ExternalShareStatus>('/share/status');
            setExternalShareStatus(res.data);
            if (!res.data.active && !res.data.cloudflared_available) {
                setExternalShareTunnelEnabled(false);
            }
        } catch (e) {
            console.error('Failed to load external share status:', e);
            if (!silent) {
                setExternalShareStatus(null);
            }
        } finally {
            if (!silent) setLoadingExternalShareStatus(false);
        }
    }, []);

    const loadCloudflaredInstallInfo = useCallback(async () => {
        try {
            const res = await api.get<CloudflaredInstallInfo>('/system/cloudflared/install-info');
            setCloudflaredInstallInfo(res.data);
        } catch (e) {
            console.error('Failed to load cloudflared install info:', e);
            setCloudflaredInstallInfo(null);
        }
    }, []);

    useEffect(() => {
        void loadExternalShareStatus();
        void loadCloudflaredInstallInfo();
    }, [loadExternalShareStatus, loadCloudflaredInstallInfo]);

    useEffect(() => {
        if (!externalShareStatus?.active) return;
        const timer = window.setInterval(() => {
            void loadExternalShareStatus(true);
        }, 10000);
        return () => window.clearInterval(timer);
    }, [externalShareStatus?.active, loadExternalShareStatus]);

    const handleStartExternalShare = async () => {
        if (externalShareTunnelEnabled && externalShareStatus && !externalShareStatus.cloudflared_available) {
            alert('cloudflared is not installed or not on PATH. Install cloudflared first, or turn off Public tunnel to start a LAN-guarded share session.');
            return;
        }
        setStartingExternalShare(true);
        try {
            const res = await api.post<ExternalShareStatus>('/share/start', {
                enable_tunnel: externalShareTunnelEnabled,
                frontend_port: externalShareFrontendPort,
                backend_port: externalShareBackendPort,
                duration_minutes: externalShareDurationHours * 60,
                password: externalSharePassword,
                ip_allowlist: externalShareIpAllowlist,
            });
            setExternalShareStatus(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to start external share mode');
        } finally {
            setStartingExternalShare(false);
        }
    };

    const handleStopExternalShare = async () => {
        setStoppingExternalShare(true);
        try {
            const res = await api.post<ExternalShareStatus>('/share/stop');
            setExternalShareStatus(res.data);
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to stop external share mode');
        } finally {
            setStoppingExternalShare(false);
        }
    };

    const handleCopyExternalShareUrl = async () => {
        const shareUrl = String(externalShareStatus?.share_url || '').trim();
        if (!shareUrl) return;
        try {
            await navigator.clipboard.writeText(shareUrl);
            setCopiedExternalShareUrl(true);
            window.setTimeout(() => setCopiedExternalShareUrl(false), 1500);
        } catch {
            alert('Failed to copy share URL');
        }
    };

    const handleInstallCloudflared = async () => {
        if (!cloudflaredInstallInfo?.package_manager_available) return;
        const label = cloudflaredInstallInfo.package_manager || 'package manager';
        if (!confirm(`Install cloudflared using ${label}?`)) return;
        setInstallingCloudflared(true);
        try {
            const res = await api.post('/system/cloudflared/install');
            alert(res.data?.status === 'installed'
                ? 'cloudflared installed successfully.'
                : 'cloudflared is already installed.');
            await loadCloudflaredInstallInfo();
            await loadExternalShareStatus();
        } catch (e: any) {
            alert(e?.response?.data?.detail || 'Failed to install cloudflared automatically.');
        } finally {
            setInstallingCloudflared(false);
        }
    };

    return {
        externalShareStatus,
        loadingExternalShareStatus,
        startingExternalShare,
        stoppingExternalShare,
        copiedExternalShareUrl,
        externalShareTunnelEnabled,
        setExternalShareTunnelEnabled,
        externalShareFrontendPort,
        setExternalShareFrontendPort,
        externalShareBackendPort,
        setExternalShareBackendPort,
        externalShareDurationHours,
        setExternalShareDurationHours,
        externalSharePassword,
        setExternalSharePassword,
        externalShareIpAllowlist,
        setExternalShareIpAllowlist,
        cloudflaredInstallInfo,
        installingCloudflared,
        loadExternalShareStatus,
        handleStartExternalShare,
        handleStopExternalShare,
        handleCopyExternalShareUrl,
        handleInstallCloudflared,
    };
}
