import { useRef, useCallback } from 'react';
import axios from 'axios';

/**
 * Creates a guarded fetch function with in-flight dedup and abort-on-force.
 *
 * Returns a stable `fetch(force?)` callback that:
 *  - Skips if a request is already in flight (unless `force` is true)
 *  - Aborts the previous request when forcing
 *  - Calls `onSuccess` with the response data when the fetch succeeds
 *  - Calls `onError` (if provided) on non-canceled failures
 *  - Respects a `mountedRef` so stale callbacks don't update state
 */
export function usePollingFetch<T>(opts: {
    mountedRef: React.MutableRefObject<boolean>;
    request: (signal: AbortSignal) => Promise<{ data: T }>;
    onSuccess: (data: T) => void;
    onError?: (err: unknown) => void;
    onFinally?: () => void;
}) {
    const inFlightRef = useRef(false);
    const abortRef = useRef<AbortController | null>(null);

    const fetch = useCallback(async (force = false) => {
        if (inFlightRef.current && !force) return;
        if (force && abortRef.current) abortRef.current.abort();
        inFlightRef.current = true;
        const controller = new AbortController();
        abortRef.current = controller;
        try {
            const res = await opts.request(controller.signal);
            if (!opts.mountedRef.current) return;
            opts.onSuccess(res.data);
        } catch (e) {
            if (axios.isAxiosError(e) && (e.code === 'ERR_CANCELED' || e.message === 'canceled')) return;
            if (!opts.mountedRef.current) return;
            opts.onError?.(e);
        } finally {
            inFlightRef.current = false;
            opts.onFinally?.();
        }
    }, [opts]);

    return fetch;
}
