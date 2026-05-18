import { useEffect, useMemo } from 'react';
import type { TranscriptSegment } from '../../types';
import { useTranscriptStore } from '../../store/useTranscriptStore';

type UseSearchResultScrollOptions = {
    segments: TranscriptSegment[];
};

export function useSearchResultScroll({ segments }: UseSearchResultScrollOptions) {
    const searchQuery = useTranscriptStore((s) => s.searchQuery);
    const searchMatchIndex = useTranscriptStore((s) => s.searchMatchIndex);
    const setSearchMatchIndex = useTranscriptStore((s) => s.setSearchMatchIndex);

    const searchLower = searchQuery.toLowerCase().trim();
    const filteredSegments = useMemo(
        () => (searchLower ? segments.filter((seg) => seg.text.toLowerCase().includes(searchLower)) : segments),
        [segments, searchLower],
    );

    useEffect(() => {
        if (searchLower && filteredSegments.length > 0 && searchMatchIndex < filteredSegments.length) {
            const seg = filteredSegments[searchMatchIndex];
            const el = document.getElementById(`seg-${seg.id}`);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, [filteredSegments, searchLower, searchMatchIndex]);

    useEffect(() => {
        setSearchMatchIndex(0);
    }, [searchQuery, setSearchMatchIndex]);

    return {
        searchLower,
        filteredSegments,
        totalMatches: filteredSegments.length,
    };
}
