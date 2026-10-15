import axios from 'axios';

/** Read server error messages without assuming that every thrown value is Axios. */
export function getErrorDetail(error: unknown): string | undefined {
    if (!axios.isAxiosError<{ detail?: unknown; error?: unknown }>(error)) return undefined;
    const detail = error.response?.data?.detail ?? error.response?.data?.error;
    return typeof detail === 'string' ? detail : undefined;
}

export function getErrorStatus(error: unknown): number | undefined {
    return axios.isAxiosError(error) ? error.response?.status : undefined;
}

export function getErrorMessage(error: unknown): string | undefined {
    return error instanceof Error ? error.message : undefined;
}
