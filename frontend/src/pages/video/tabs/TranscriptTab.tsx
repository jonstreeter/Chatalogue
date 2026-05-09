import type { ReactNode } from 'react';

type Props = {
    renderSidebar: () => ReactNode;
    renderFunnyMomentsOverlay: () => ReactNode;
};

export function TranscriptTab({ renderSidebar, renderFunnyMomentsOverlay }: Props) {
    return (
        <>
            {renderSidebar()}
            {renderFunnyMomentsOverlay()}
        </>
    );
}
