import type { ReactNode } from 'react';

type Props = {
    renderSidebar: () => ReactNode;
    renderFunnyMomentsOverlay: () => ReactNode;
};

export function TranscriptTab({ renderSidebar, renderFunnyMomentsOverlay }: Props) {
    return (
        <>
            <TranscriptSidebarContent renderSidebar={renderSidebar} />
            <TranscriptFunnyMomentsOverlay renderFunnyMomentsOverlay={renderFunnyMomentsOverlay} />
        </>
    );
}

function TranscriptSidebarContent({ renderSidebar }: Pick<Props, 'renderSidebar'>) {
    return <>{renderSidebar()}</>;
}

function TranscriptFunnyMomentsOverlay({ renderFunnyMomentsOverlay }: Pick<Props, 'renderFunnyMomentsOverlay'>) {
    return <>{renderFunnyMomentsOverlay()}</>;
}
