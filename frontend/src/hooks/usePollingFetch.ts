import { useRef, useCallback, useEffect } from 'react';
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
    const optsRef = useRef(opts);
    const inFlightRef = useRef(false);
    const abortRef = useRef<AbortController | null>(null);
    const requestSeqRef = useRef(0);

    useEffect(() => {
        optsRef.current = opts;
    }, [opts]);

    const fetch = useCallback(async (force = false) => {
        const currentOpts = optsRef.current;
        if (inFlightRef.current && !force) return;
        if (force && abortRef.current) abortRef.current.abort();
        const requestSeq = requestSeqRef.current + 1;
        requestSeqRef.current = requestSeq;
        inFlightRef.current = true;
        const controller = new AbortController();
        abortRef.current = controller;
        try {
            const res = await currentOpts.request(controller.signal);
            if (!currentOpts.mountedRef.current || requestSeq !== requestSeqRef.current) return;
            currentOpts.onSuccess(res.data);
        } catch (e) {
            if (axios.isAxiosError(e) && (e.code === 'ERR_CANCELED' || e.message === 'canceled')) return;
            if (!currentOpts.mountedRef.current || requestSeq !== requestSeqRef.current) return;
            currentOpts.onError?.(e);
        } finally {
            if (requestSeq === requestSeqRef.current) {
                inFlightRef.current = false;
                abortRef.current = null;
                currentOpts.onFinally?.();
            }
        }
    }, []);

    return fetch;
}
