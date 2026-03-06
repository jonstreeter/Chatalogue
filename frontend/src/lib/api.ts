import axios from 'axios';

function normalizeLoopbackHost(url: string): string {
    return url.replace('://localhost:', '://127.0.0.1:');
}

function resolveApiBaseUrl(): string {
    const envUrl = (import.meta.env.VITE_API_URL || '').trim();
    if (envUrl) {
        return normalizeLoopbackHost(envUrl);
    }

    if (typeof window !== 'undefined') {
        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
        const host = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
        return `${protocol}//${host}:8011`;
    }

    return 'http://127.0.0.1:8011';
}

export const API_BASE_URL = resolveApiBaseUrl();

export function toApiUrl(pathOrUrl?: string | null): string {
    const value = (pathOrUrl || '').trim();
    if (!value) return '';
    if (/^https?:\/\//i.test(value)) return normalizeLoopbackHost(value);
    return `${API_BASE_URL}${value.startsWith('/') ? '' : '/'}${value}`;
}

const api = axios.create({
    baseURL: API_BASE_URL,
});

export default api;
