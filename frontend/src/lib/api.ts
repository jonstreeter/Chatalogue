import axios from 'axios';

const SHARE_TOKEN_STORAGE_KEY = 'chatalogue_share_token';
const SHARE_PASSWORD_STORAGE_KEY = 'chatalogue_share_password';
const SHARE_API_BASE_STORAGE_KEY = 'chatalogue_share_api_base';

function normalizeLoopbackHost(url: string): string {
    return url.replace('://localhost:', '://127.0.0.1:');
}

function getSearchParam(name: string): string {
    if (typeof window === 'undefined') return '';
    return new URLSearchParams(window.location.search).get(name)?.trim() || '';
}

function persistShareContextFromUrl(): void {
    if (typeof window === 'undefined') return;
    const token = getSearchParam('share_token');
    const password = getSearchParam('share_password');
    const apiBase = getSearchParam('api_base');
    if (token) {
        window.sessionStorage.setItem(SHARE_TOKEN_STORAGE_KEY, token);
    }
    if (password) {
        window.sessionStorage.setItem(SHARE_PASSWORD_STORAGE_KEY, password);
    }
    if (apiBase) {
        window.sessionStorage.setItem(SHARE_API_BASE_STORAGE_KEY, normalizeLoopbackHost(apiBase));
    }
}

function resolveApiBaseUrl(): string {
    const queryUrl = getSearchParam('api_base');
    if (queryUrl) {
        return normalizeLoopbackHost(queryUrl);
    }

    if (typeof window !== 'undefined') {
        const storedToken = window.sessionStorage.getItem(SHARE_TOKEN_STORAGE_KEY)?.trim() || '';
        const storedApiBase = window.sessionStorage.getItem(SHARE_API_BASE_STORAGE_KEY)?.trim() || '';
        if (storedToken && storedApiBase) {
            return normalizeLoopbackHost(storedApiBase);
        }
    }

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

persistShareContextFromUrl();

export function getApiBaseUrl(): string {
    persistShareContextFromUrl();
    return resolveApiBaseUrl();
}

export const API_BASE_URL = getApiBaseUrl();

export function getShareToken(): string {
    if (typeof window === 'undefined') return '';
    return window.sessionStorage.getItem(SHARE_TOKEN_STORAGE_KEY)?.trim() || '';
}

export function getShareApiBase(): string {
    if (typeof window === 'undefined') return '';
    return window.sessionStorage.getItem(SHARE_API_BASE_STORAGE_KEY)?.trim() || '';
}

export function getSharePassword(): string {
    if (typeof window === 'undefined') return '';
    return window.sessionStorage.getItem(SHARE_PASSWORD_STORAGE_KEY)?.trim() || '';
}

export function setSharePassword(password: string): void {
    if (typeof window === 'undefined') return;
    const value = (password || '').trim();
    if (value) {
        window.sessionStorage.setItem(SHARE_PASSWORD_STORAGE_KEY, value);
    } else {
        window.sessionStorage.removeItem(SHARE_PASSWORD_STORAGE_KEY);
    }
}

export function toApiUrl(pathOrUrl?: string | null): string {
    const value = (pathOrUrl || '').trim();
    if (!value) return '';
    if (/^https?:\/\//i.test(value)) return normalizeLoopbackHost(value);
    const baseUrl = getApiBaseUrl();
    return `${baseUrl}${value.startsWith('/') ? '' : '/'}${value}`;
}

const api = axios.create();

api.interceptors.request.use((config) => {
    config.baseURL = getApiBaseUrl();
    const token = getShareToken();
    const password = getSharePassword();
    if (token) {
        config.headers = config.headers || {};
        config.headers['X-Chatalogue-Share-Token'] = token;
    }
    if (password) {
        config.headers = config.headers || {};
        config.headers['X-Chatalogue-Share-Password'] = password;
    }
    return config;
});

export default api;
