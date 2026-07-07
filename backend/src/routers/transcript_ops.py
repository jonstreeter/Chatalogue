"""Transcript optimization endpoints: quality, runs, gold windows, evaluation,
optimization campaigns, repair, diarization rebuild, retranscription."""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db.database import (
    Channel,
    Job,
    TranscriptEvaluationResult,
    TranscriptEvaluationReview,
    TranscriptGoldWindow,
    TranscriptOptimizationCampaign,
    TranscriptOptimizationCampaignItem,
    TranscriptQualitySnapshot,
    TranscriptRun,
    TranscriptSegment,
    Video,
)
from ..deps import get_ingestion_service, get_session
from ..video_utils import (
    _enqueue_unique_job,
    _queue_diarization_rebuild_job,
    _queue_full_retranscription_job,
)
from ..job_utils import PIPELINE_ACTIVE_STATUSES
from ..schemas import (
    TranscriptDiarizationBenchmarkRequest,
    TranscriptDiarizationConfigBenchmarkRead,
    TranscriptDiarizationRebuildBulkQueueRequest,
    TranscriptDiarizationRebuildBulkQueueResponse,
    TranscriptDiarizationRebuildQueueRequest,
    TranscriptDiarizationRebuildQueueResponse,
    TranscriptEvaluationBatchResponse,
    TranscriptEvaluationResultRead,
    TranscriptEvaluationReviewRead,
    TranscriptEvaluationReviewRequest,
    TranscriptEvaluationSummaryRead,
    TranscriptGoldWindowRead,
    TranscriptGoldWindowUpsertRequest,
    TranscriptOptimizationCampaignCreateRequest,
    TranscriptOptimizationCampaignDeleteResponse,
    TranscriptOptimizationCampaignExecuteResponse,
    TranscriptOptimizationCampaignItemRead,
    TranscriptOptimizationCampaignRead,
    TranscriptOptimizeDryRunRequest,
    TranscriptOptimizeDryRunResponse,
    TranscriptQualityRead,
    TranscriptQualitySnapshotRead,
    TranscriptRepairBulkQueueRequest,
    TranscriptRepairBulkQueueResponse,
    TranscriptRepairQueueRequest,
    TranscriptRepairQueueResponse,
    TranscriptRestoreResponse,
    TranscriptRetranscriptionBulkQueueRequest,
    TranscriptRetranscriptionBulkQueueResponse,
    TranscriptRetranscriptionQueueRequest,
    TranscriptRetranscriptionQueueResponse,
    TranscriptRollbackOptionRead,
    TranscriptRunRead,
)
from ..services import speaker_queries as spk_q

router = APIRouter()


def _main():
    """Transitional accessor for helpers that still live in main.py."""
    from .. import main

    return main


@router.get("/videos/{video_id}/transcript-quality", response_model=TranscriptQualityRead)
def get_transcript_quality(
    video_id: int,
    persist_snapshot: bool = False,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    try:
        return get_ingestion_service().evaluate_transcript_quality(
            session,
            video_id,
            source="api",
            persist_snapshot=bool(persist_snapshot),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/videos/{video_id}/transcript-quality/snapshots", response_model=List[TranscriptQualitySnapshotRead])
def list_transcript_quality_snapshots(video_id: int, session: Session = Depends(get_session)):
    snapshots = session.exec(
        select(TranscriptQualitySnapshot)
        .where(TranscriptQualitySnapshot.video_id == video_id)
        .order_by(TranscriptQualitySnapshot.created_at.desc(), TranscriptQualitySnapshot.id.desc())
    ).all()
    items: list[TranscriptQualitySnapshotRead] = []
    for snapshot in snapshots:
        try:
            metrics = json.loads(snapshot.metrics_json or "{}")
        except Exception:
            metrics = {}
        try:
            reasons = json.loads(snapshot.reasons_json or "[]")
        except Exception:
            reasons = []
        items.append(
            TranscriptQualitySnapshotRead(
                id=int(snapshot.id),
                video_id=int(snapshot.video_id),
                run_id=snapshot.run_id,
                source=str(snapshot.source or "manual"),
                quality_profile=str(snapshot.quality_profile or "unknown"),
                recommended_tier=str(snapshot.recommended_tier or "none"),
                score=float(snapshot.score or 0.0),
                metrics=metrics if isinstance(metrics, dict) else {},
                reasons=reasons if isinstance(reasons, list) else [],
                created_at=snapshot.created_at,
            )
        )
    return items


@router.get("/videos/{video_id}/transcript-runs", response_model=List[TranscriptRunRead])
def list_transcript_runs(video_id: int, session: Session = Depends(get_session)):
    runs = session.exec(
        select(TranscriptRun)
        .where(TranscriptRun.video_id == video_id)
        .order_by(TranscriptRun.created_at.desc(), TranscriptRun.id.desc())
    ).all()
    items: list[TranscriptRunRead] = []
    for run in runs:
        try:
            metrics_before = json.loads(run.metrics_before_json or "{}") if run.metrics_before_json else None
        except Exception:
            metrics_before = None
        try:
            metrics_after = json.loads(run.metrics_after_json or "{}") if run.metrics_after_json else None
        except Exception:
            metrics_after = None
        try:
            artifact_refs = json.loads(run.artifact_refs_json or "{}") if run.artifact_refs_json else None
        except Exception:
            artifact_refs = None
        try:
            model_provenance = json.loads(run.model_provenance_json or "{}") if run.model_provenance_json else None
        except Exception:
            model_provenance = None
        items.append(
            TranscriptRunRead(
                id=int(run.id),
                video_id=int(run.video_id),
                input_run_id=run.input_run_id,
                mode=str(run.mode or "baseline"),
                pipeline_version=str(run.pipeline_version or "baseline-v1"),
                status=str(run.status or "completed"),
                quality_profile=run.quality_profile,
                recommended_tier=run.recommended_tier,
                started_at=run.started_at,
                completed_at=run.completed_at,
                metrics_before=metrics_before if isinstance(metrics_before, dict) else None,
                metrics_after=metrics_after if isinstance(metrics_after, dict) else None,
                artifact_refs=artifact_refs if isinstance(artifact_refs, dict) else None,
                rollback_state=run.rollback_state,
                model_provenance=model_provenance if isinstance(model_provenance, dict) else None,
                note=run.note,
                created_at=run.created_at,
            )
        )
    return items


@router.get("/videos/{video_id}/transcript-rollback-options", response_model=List[TranscriptRollbackOptionRead])
def list_transcript_rollback_options(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    return [
        TranscriptRollbackOptionRead(**item)
        for item in get_ingestion_service().list_transcript_rollback_options(session, video_id)
    ]


@router.post("/videos/{video_id}/transcript-runs/{run_id}/restore", response_model=TranscriptRestoreResponse)
def restore_transcript_run(
    video_id: int,
    run_id: int,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video_id,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=409, detail=f"Video has an active job {active_job.id} ({active_job.status})")
    try:
        result = get_ingestion_service().restore_transcript_from_run(session, video_id, run_id, source="api_restore")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptRestoreResponse(**result)


def _serialize_transcript_gold_window(window: TranscriptGoldWindow) -> TranscriptGoldWindowRead:
    try:
        entities = json.loads(window.entities_json or "[]") if window.entities_json else []
    except Exception:
        entities = []
    return TranscriptGoldWindowRead(
        id=int(window.id),
        video_id=int(window.video_id),
        label=str(window.label or "window"),
        quality_profile=window.quality_profile,
        language=window.language,
        start_time=float(window.start_time),
        end_time=float(window.end_time),
        reference_text=str(window.reference_text or ""),
        entities=entities if isinstance(entities, list) else [],
        notes=window.notes,
        active=bool(window.active),
        created_at=window.created_at,
        updated_at=window.updated_at,
    )


def _serialize_transcript_evaluation_result(result: TranscriptEvaluationResult) -> TranscriptEvaluationResultRead:
    try:
        metrics = json.loads(result.metrics_json or "{}") if result.metrics_json else {}
    except Exception:
        metrics = {}
    return TranscriptEvaluationResultRead(
        id=int(result.id),
        gold_window_id=int(result.gold_window_id),
        video_id=int(result.video_id),
        run_id=result.run_id,
        source=str(result.source or "manual"),
        candidate_text=str(result.candidate_text or ""),
        reference_text=str(result.reference_text or ""),
        wer=float(result.wer or 0.0),
        cer=float(result.cer or 0.0),
        entity_accuracy=float(result.entity_accuracy) if result.entity_accuracy is not None else None,
        matched_entity_count=int(result.matched_entity_count or 0),
        total_entity_count=int(result.total_entity_count or 0),
        segment_count=int(result.segment_count or 0),
        unknown_speaker_rate=float(result.unknown_speaker_rate or 0.0),
        punctuation_density_delta=float(result.punctuation_density_delta or 0.0),
        metrics=metrics if isinstance(metrics, dict) else {},
        created_at=result.created_at,
    )


def _serialize_transcript_evaluation_review(review: TranscriptEvaluationReview) -> TranscriptEvaluationReviewRead:
    try:
        tags = json.loads(review.tags_json or "[]") if review.tags_json else []
    except Exception:
        tags = []
    return TranscriptEvaluationReviewRead(
        id=int(review.id),
        evaluation_result_id=int(review.evaluation_result_id),
        reviewer=review.reviewer,
        verdict=str(review.verdict or "same"),
        tags=tags if isinstance(tags, list) else [],
        notes=review.notes,
        created_at=review.created_at,
    )


def _serialize_transcript_rollback_option(run: TranscriptRun) -> TranscriptRollbackOptionRead:
    rollback_state = str(run.rollback_state or "").strip() or None
    rollback_available = False
    if rollback_state:
        try:
            rollback_available = Path(rollback_state).exists()
        except Exception:
            rollback_available = False
    return TranscriptRollbackOptionRead(
        run_id=int(run.id),
        video_id=int(run.video_id),
        mode=str(run.mode or "unknown"),
        pipeline_version=str(run.pipeline_version or ""),
        note=run.note,
        created_at=run.created_at,
        rollback_available=rollback_available,
        rollback_state=rollback_state,
    )


def _serialize_transcript_campaign(campaign: TranscriptOptimizationCampaign) -> TranscriptOptimizationCampaignRead:
    try:
        tiers = json.loads(campaign.tiers_json or "[]") if campaign.tiers_json else []
    except Exception:
        tiers = []
    return TranscriptOptimizationCampaignRead(
        id=int(campaign.id),
        channel_id=campaign.channel_id,
        scope=str(campaign.scope or "global"),
        status=str(campaign.status or "draft"),
        tiers=tiers if isinstance(tiers, list) else [],
        limit=int(campaign.limit or 0),
        force_non_eligible=bool(campaign.force_non_eligible),
        queued_jobs=int(campaign.queued_jobs or 0),
        skipped_active=int(campaign.skipped_active or 0),
        skipped_no_segments=int(campaign.skipped_no_segments or 0),
        skipped_not_eligible=int(campaign.skipped_not_eligible or 0),
        skipped_other=int(campaign.skipped_other or 0),
        note=campaign.note,
        created_at=campaign.created_at,
        updated_at=campaign.updated_at,
    )


def _serialize_transcript_campaign_item(item: TranscriptOptimizationCampaignItem) -> TranscriptOptimizationCampaignItemRead:
    return TranscriptOptimizationCampaignItemRead(
        id=int(item.id),
        campaign_id=int(item.campaign_id),
        video_id=int(item.video_id),
        recommended_tier=str(item.recommended_tier or "none"),
        action_tier=str(item.action_tier or "none"),
        quality_score=float(item.quality_score or 0.0),
        reason=item.reason,
        status=str(item.status or "pending"),
        job_id=item.job_id,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.get("/videos/{video_id}/transcript-gold-windows", response_model=List[TranscriptGoldWindowRead])
def list_transcript_gold_windows(video_id: int, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    windows = session.exec(
        select(TranscriptGoldWindow)
        .where(TranscriptGoldWindow.video_id == video_id)
        .order_by(TranscriptGoldWindow.start_time, TranscriptGoldWindow.id)
    ).all()
    return [_serialize_transcript_gold_window(item) for item in windows]


@router.post("/videos/{video_id}/transcript-gold-windows", response_model=TranscriptGoldWindowRead)
def create_transcript_gold_window(
    video_id: int,
    body: TranscriptGoldWindowUpsertRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    try:
        window = get_ingestion_service().upsert_transcript_gold_window(
            session,
            video_id,
            label=body.label,
            quality_profile=body.quality_profile,
            language=body.language,
            start_time=body.start_time,
            end_time=body.end_time,
            reference_text=body.reference_text,
            entities=body.entities,
            notes=body.notes,
            active=body.active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_transcript_gold_window(window)


@router.put("/transcript-gold-windows/{window_id}", response_model=TranscriptGoldWindowRead)
def update_transcript_gold_window(
    window_id: int,
    body: TranscriptGoldWindowUpsertRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    existing = session.get(TranscriptGoldWindow, window_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Gold window not found")
    try:
        window = get_ingestion_service().upsert_transcript_gold_window(
            session,
            int(existing.video_id),
            window_id=int(window_id),
            label=body.label,
            quality_profile=body.quality_profile,
            language=body.language,
            start_time=body.start_time,
            end_time=body.end_time,
            reference_text=body.reference_text,
            entities=body.entities,
            notes=body.notes,
            active=body.active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_transcript_gold_window(window)


@router.post("/videos/{video_id}/transcript-evaluation", response_model=TranscriptEvaluationBatchResponse)
def evaluate_transcript_against_gold_windows(
    video_id: int,
    run_id: Optional[int] = None,
    active_only: bool = True,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        result = get_ingestion_service().evaluate_transcript_gold_windows(
            session,
            video_id,
            run_id=run_id,
            source="manual_api",
            active_only=bool(active_only),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptEvaluationBatchResponse(
        video_id=int(result["video_id"]),
        run_id=result.get("run_id"),
        total_windows=int(result["total_windows"]),
        average_wer=float(result["average_wer"]),
        average_cer=float(result["average_cer"]),
        average_entity_accuracy=float(result["average_entity_accuracy"]) if result.get("average_entity_accuracy") is not None else None,
        average_unknown_speaker_rate=float(result["average_unknown_speaker_rate"]),
        items=[_serialize_transcript_evaluation_result(item) for item in result["items"]],
    )


@router.get("/videos/{video_id}/transcript-evaluation-results", response_model=List[TranscriptEvaluationResultRead])
def list_transcript_evaluation_results(video_id: int, run_id: Optional[int] = None, session: Session = Depends(get_session)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    query = select(TranscriptEvaluationResult).where(TranscriptEvaluationResult.video_id == video_id)
    if run_id is not None:
        query = query.where(TranscriptEvaluationResult.run_id == run_id)
    results = session.exec(
        query.order_by(TranscriptEvaluationResult.created_at.desc(), TranscriptEvaluationResult.id.desc())
    ).all()
    return [_serialize_transcript_evaluation_result(item) for item in results]


@router.post("/transcript-evaluation-results/{result_id}/review", response_model=TranscriptEvaluationReviewRead)
def create_transcript_evaluation_review(
    result_id: int,
    body: TranscriptEvaluationReviewRequest,
    session: Session = Depends(get_session),
):
    result = session.get(TranscriptEvaluationResult, result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    review = TranscriptEvaluationReview(
        evaluation_result_id=int(result_id),
        reviewer=str(body.reviewer or "").strip() or None,
        verdict=str(body.verdict),
        tags_json=json.dumps(list(body.tags or []), ensure_ascii=False),
        notes=str(body.notes or "").strip() or None,
    )
    session.add(review)
    session.commit()
    session.refresh(review)
    return _serialize_transcript_evaluation_review(review)


@router.get("/transcript-evaluation-results/{result_id}/reviews", response_model=List[TranscriptEvaluationReviewRead])
def list_transcript_evaluation_reviews(result_id: int, session: Session = Depends(get_session)):
    result = session.get(TranscriptEvaluationResult, result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    reviews = session.exec(
        select(TranscriptEvaluationReview)
        .where(TranscriptEvaluationReview.evaluation_result_id == result_id)
        .order_by(TranscriptEvaluationReview.created_at.desc(), TranscriptEvaluationReview.id.desc())
    ).all()
    return [_serialize_transcript_evaluation_review(item) for item in reviews]


@router.get("/transcript-evaluation/summary", response_model=TranscriptEvaluationSummaryRead)
def get_transcript_evaluation_summary(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    if channel_id is not None:
        channel = session.get(Channel, channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
    return TranscriptEvaluationSummaryRead(**get_ingestion_service().summarize_transcript_evaluation(session, channel_id=channel_id))


@router.get("/transcript-evaluation/diarization-config-summary", response_model=List[TranscriptDiarizationConfigBenchmarkRead])
def get_transcript_diarization_config_summary(channel_id: Optional[int] = None, session: Session = Depends(get_session)):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    if channel_id is not None:
        channel = session.get(Channel, channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
    return [
        TranscriptDiarizationConfigBenchmarkRead(**item)
        for item in get_ingestion_service().summarize_diarization_benchmarks(session, channel_id=channel_id)
    ]


@router.post("/transcripts/optimize/dry-run", response_model=TranscriptOptimizeDryRunResponse)
def transcript_optimize_dry_run(
    request: TranscriptOptimizeDryRunRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    result = get_ingestion_service().transcript_optimization_dry_run(
        session,
        channel_id=request.channel_id,
        video_id=request.video_id,
        limit=request.limit,
        persist_snapshots=bool(request.persist_snapshots),
    )
    return TranscriptOptimizeDryRunResponse(**result)


@router.get("/transcript-optimization-campaigns", response_model=List[TranscriptOptimizationCampaignRead])
def list_transcript_optimization_campaigns(
    channel_id: Optional[int] = None,
    limit: int = 30,
    session: Session = Depends(get_session),
):
    query = select(TranscriptOptimizationCampaign)
    if channel_id is not None:
        query = query.where(TranscriptOptimizationCampaign.channel_id == channel_id)
    campaigns = session.exec(
        query.order_by(TranscriptOptimizationCampaign.created_at.desc(), TranscriptOptimizationCampaign.id.desc()).limit(max(1, min(int(limit), 200)))
    ).all()
    return [_serialize_transcript_campaign(item) for item in campaigns]


@router.post("/transcript-optimization-campaigns", response_model=TranscriptOptimizationCampaignRead)
def create_transcript_optimization_campaign(
    request: TranscriptOptimizationCampaignCreateRequest,
    session: Session = Depends(get_session),
):
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    if request.channel_id is not None:
        channel = session.get(Channel, request.channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
    campaign = get_ingestion_service().create_transcript_optimization_campaign(
        session,
        channel_id=request.channel_id,
        limit=request.limit,
        tiers=list(request.tiers or []),
        force_non_eligible=bool(request.force_non_eligible),
        note=request.note,
    )
    return _serialize_transcript_campaign(campaign)


@router.get("/transcript-optimization-campaigns/{campaign_id}/items", response_model=List[TranscriptOptimizationCampaignItemRead])
def list_transcript_optimization_campaign_items(campaign_id: int, session: Session = Depends(get_session)):
    campaign = session.get(TranscriptOptimizationCampaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    items = session.exec(
        select(TranscriptOptimizationCampaignItem)
        .where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id)
        .order_by(TranscriptOptimizationCampaignItem.quality_score.asc(), TranscriptOptimizationCampaignItem.id.asc())
    ).all()
    return [_serialize_transcript_campaign_item(item) for item in items]


@router.post("/transcript-optimization-campaigns/{campaign_id}/execute", response_model=TranscriptOptimizationCampaignExecuteResponse)
def execute_transcript_optimization_campaign(campaign_id: int, session: Session = Depends(get_session)):
    campaign = session.get(TranscriptOptimizationCampaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    items = session.exec(
        select(TranscriptOptimizationCampaignItem)
        .where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id)
        .order_by(TranscriptOptimizationCampaignItem.quality_score.asc(), TranscriptOptimizationCampaignItem.id.asc())
    ).all()

    queued_jobs = 0
    skipped_active = 0
    skipped_no_segments = 0
    skipped_not_eligible = 0
    skipped_other = 0

    for item in items:
        video = session.get(Video, item.video_id)
        if not video:
            item.status = "skipped_missing_video"
            skipped_other += 1
            session.add(item)
            continue
        active_job = session.exec(
            select(Job.id).where(
                Job.video_id == item.video_id,
                Job.status.in_(PIPELINE_ACTIVE_STATUSES),
            )
        ).first()
        if active_job:
            item.status = "skipped_active"
            skipped_active += 1
            session.add(item)
            continue

        action_tier = str(item.action_tier or item.recommended_tier or "none")
        if action_tier not in {"low_risk_repair", "diarization_rebuild", "full_retranscription"}:
            item.status = "skipped_not_supported"
            skipped_other += 1
            session.add(item)
            continue
        if not bool(campaign.force_non_eligible) and str(item.recommended_tier or "none") != action_tier:
            item.status = "skipped_not_eligible"
            skipped_not_eligible += 1
            session.add(item)
            continue
        seg_exists = session.exec(select(TranscriptSegment.id).where(TranscriptSegment.video_id == item.video_id).limit(1)).first()
        if not seg_exists:
            item.status = "skipped_no_segments"
            skipped_no_segments += 1
            session.add(item)
            continue
        try:
            if action_tier == "low_risk_repair":
                job = _enqueue_unique_job(
                    session,
                    video_id=int(item.video_id),
                    job_type="transcript_repair",
                    payload={
                        "save_files": True,
                        "force": bool(campaign.force_non_eligible),
                        "note": str(campaign.note or "").strip() or None,
                        "queued_from": f"campaign:{campaign_id}",
                    },
                )
            elif action_tier == "diarization_rebuild":
                job, _, _, _ = _queue_diarization_rebuild_job(
                    session,
                    video=video,
                    force=bool(campaign.force_non_eligible),
                    note=campaign.note,
                    queued_from=f"campaign:{campaign_id}",
                )
            else:
                job, _, _, _ = _queue_full_retranscription_job(
                    session,
                    video=video,
                    force=bool(campaign.force_non_eligible),
                    note=campaign.note,
                    queued_from=f"campaign:{campaign_id}",
                )
            item.job_id = int(job.id)
            item.status = "queued"
            item.updated_at = datetime.now()
            session.add(item)
            queued_jobs += 1
        except HTTPException as exc:
            detail_text = str(exc.detail or "")
            if exc.status_code in {400, 409} and "active job" in detail_text.lower():
                item.status = "skipped_active"
                skipped_active += 1
            elif "no raw transcript" in detail_text.lower() or "no segments" in detail_text.lower():
                item.status = "skipped_no_segments"
                skipped_no_segments += 1
            elif "not '" in detail_text.lower() or "not '" in detail_text:
                item.status = "skipped_not_eligible"
                skipped_not_eligible += 1
            else:
                item.status = "failed_queue"
                skipped_other += 1
            item.updated_at = datetime.now()
            session.add(item)

    campaign.status = "queued"
    campaign.queued_jobs = queued_jobs
    campaign.skipped_active = skipped_active
    campaign.skipped_no_segments = skipped_no_segments
    campaign.skipped_not_eligible = skipped_not_eligible
    campaign.skipped_other = skipped_other
    campaign.updated_at = datetime.now()
    session.add(campaign)
    session.commit()
    return TranscriptOptimizationCampaignExecuteResponse(
        campaign_id=int(campaign_id),
        status=str(campaign.status or "queued"),
        queued_jobs=queued_jobs,
        skipped_active=skipped_active,
        skipped_no_segments=skipped_no_segments,
        skipped_not_eligible=skipped_not_eligible,
        skipped_other=skipped_other,
    )


@router.delete("/transcript-optimization-campaigns/{campaign_id}", response_model=TranscriptOptimizationCampaignDeleteResponse)
def delete_transcript_optimization_campaign(campaign_id: int, session: Session = Depends(get_session)):
    from sqlalchemy import delete as sa_delete

    campaign = session.get(TranscriptOptimizationCampaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    items = session.exec(
        select(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id)
    ).all()

    detached_job_refs = 0
    for item in items:
        if item.job_id is not None:
            item.job_id = None
            detached_job_refs += 1
            session.add(item)
    session.flush()
    session.exec(sa_delete(TranscriptOptimizationCampaignItem).where(TranscriptOptimizationCampaignItem.campaign_id == campaign_id))
    session.delete(campaign)
    session.commit()

    return TranscriptOptimizationCampaignDeleteResponse(
        campaign_id=int(campaign_id),
        deleted_items=int(len(items)),
        detached_job_refs=int(detached_job_refs),
        status="deleted",
    )


@router.post("/videos/{video_id}/transcript-repair", response_model=TranscriptRepairQueueResponse)
def queue_transcript_repair(
    video_id: int,
    request: TranscriptRepairQueueRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    active_job = session.exec(
        select(Job).where(
            Job.video_id == video.id,
            Job.status.in_(PIPELINE_ACTIVE_STATUSES),
        )
    ).first()
    if active_job:
        raise HTTPException(status_code=400, detail=f"Video has an active job {active_job.id} ({active_job.status})")

    quality = get_ingestion_service().evaluate_transcript_quality(session, video_id, source="queue_gate", persist_snapshot=False)
    if not request.force and str(quality.get("recommended_tier") or "") != "low_risk_repair":
        raise HTTPException(
            status_code=409,
            detail=f"Video is currently classified as '{quality.get('recommended_tier') or 'none'}', not 'low_risk_repair'. Use force=true to queue anyway.",
        )

    job = _enqueue_unique_job(
        session,
        video_id=video_id,
        job_type="transcript_repair",
        payload={
            "save_files": bool(request.save_files),
            "force": bool(request.force),
            "note": str(request.note or "").strip() or None,
            "queued_from": "video",
        },
    )
    return TranscriptRepairQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@router.post("/channels/{channel_id}/transcript-repair/queue", response_model=TranscriptRepairBulkQueueResponse)
def queue_channel_transcript_repairs(
    channel_id: int,
    request: TranscriptRepairBulkQueueRequest,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    if get_ingestion_service() is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id, Video.processed == True)  # noqa: E712
        .order_by(Video.published_at.desc(), Video.id.desc())
        .limit(int(request.limit))
    ).all()

    queued_jobs: list[TranscriptRepairQueueResponse] = []
    skipped_active = 0
    skipped_no_segments = 0
    skipped_not_low_risk = 0

    for video in videos:
        active_job = session.exec(
            select(Job.id).where(
                Job.video_id == video.id,
                Job.status.in_(PIPELINE_ACTIVE_STATUSES),
            )
        ).first()
        if active_job:
            skipped_active += 1
            continue

        seg_exists = session.exec(
            select(TranscriptSegment.id).where(TranscriptSegment.video_id == video.id).limit(1)
        ).first()
        if not seg_exists:
            skipped_no_segments += 1
            continue

        quality = get_ingestion_service().evaluate_transcript_quality(session, int(video.id), source="bulk_queue_gate", persist_snapshot=False)
        if not request.force_non_eligible and str(quality.get("recommended_tier") or "") != "low_risk_repair":
            skipped_not_low_risk += 1
            continue

        job = _enqueue_unique_job(
            session,
            video_id=int(video.id),
            job_type="transcript_repair",
            payload={
                "save_files": bool(request.save_files),
                "force": bool(request.force_non_eligible),
                "note": str(request.note or "").strip() or None,
                "queued_from": "channel",
                "channel_id": int(channel_id),
            },
        )
        queued_jobs.append(
            TranscriptRepairQueueResponse(
                job_id=int(job.id),
                video_id=int(video.id),
                status="queued",
                recommended_tier=str(quality.get("recommended_tier") or "none"),
                quality_score=float(quality.get("quality_score") or 0.0),
                queued=True,
            )
        )

    return TranscriptRepairBulkQueueResponse(
        channel_id=int(channel_id),
        queued=len(queued_jobs),
        skipped_active=skipped_active,
        skipped_no_segments=skipped_no_segments,
        skipped_not_low_risk=skipped_not_low_risk,
        jobs=queued_jobs,
    )


@router.post("/videos/{video_id}/transcript-diarization-rebuild", response_model=TranscriptDiarizationRebuildQueueResponse)
def queue_transcript_diarization_rebuild(
    video_id: int,
    request: TranscriptDiarizationRebuildQueueRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    job, quality, _, _ = _queue_diarization_rebuild_job(
        session,
        video=video,
        force=bool(request.force),
        note=request.note,
        queued_from="video_rebuild",
    )
    spk_q._invalidate_speaker_query_caches()
    return TranscriptDiarizationRebuildQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@router.post("/videos/{video_id}/transcript-diarization-benchmark", response_model=TranscriptDiarizationRebuildQueueResponse)
def queue_transcript_diarization_benchmark(
    video_id: int,
    request: TranscriptDiarizationBenchmarkRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    job, quality, _, _ = _queue_diarization_rebuild_job(
        session,
        video=video,
        force=bool(request.force),
        note=request.note,
        queued_from="benchmark",
        optimization_target="diarization_benchmark",
        diarization_sensitivity_override=request.diarization_sensitivity,
        speaker_match_threshold_override=request.speaker_match_threshold,
    )
    spk_q._invalidate_speaker_query_caches()
    return TranscriptDiarizationRebuildQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@router.post("/channels/{channel_id}/transcript-diarization-rebuild/queue", response_model=TranscriptDiarizationRebuildBulkQueueResponse)
def queue_channel_transcript_diarization_rebuilds(
    channel_id: int,
    request: TranscriptDiarizationRebuildBulkQueueRequest,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id)
        .order_by(Video.published_at.desc(), Video.id.desc())
        .limit(int(request.limit))
    ).all()

    jobs: list[TranscriptDiarizationRebuildQueueResponse] = []
    skipped_active = 0
    skipped_no_raw_transcript = 0
    skipped_not_diarization_rebuild = 0
    skipped_unprocessed = 0
    skipped_muted = 0

    for video in videos:
        if bool(video.muted):
            skipped_muted += 1
            continue
        if not bool(video.processed):
            skipped_unprocessed += 1
            continue
        try:
            job, quality, _, _ = _queue_diarization_rebuild_job(
                session,
                video=video,
                force=bool(request.force_non_eligible),
                note=request.note,
                queued_from="channel_rebuild",
            )
            jobs.append(
                TranscriptDiarizationRebuildQueueResponse(
                    job_id=int(job.id),
                    video_id=int(video.id),
                    status="queued",
                    recommended_tier=str(quality.get("recommended_tier") or "none"),
                    quality_score=float(quality.get("quality_score") or 0.0),
                    queued=True,
                )
            )
        except HTTPException as exc:
            detail_text = str(exc.detail or "")
            if exc.status_code == 400 and "active job" in detail_text.lower():
                skipped_active += 1
            elif exc.status_code == 400 and "no raw transcript" in detail_text.lower():
                skipped_no_raw_transcript += 1
            elif exc.status_code == 409 and "diarization_rebuild" in detail_text:
                skipped_not_diarization_rebuild += 1
            else:
                raise

    spk_q._invalidate_speaker_query_caches()
    return TranscriptDiarizationRebuildBulkQueueResponse(
        channel_id=int(channel_id),
        queued=len(jobs),
        skipped_active=skipped_active,
        skipped_no_raw_transcript=skipped_no_raw_transcript,
        skipped_not_diarization_rebuild=skipped_not_diarization_rebuild,
        skipped_unprocessed=skipped_unprocessed,
        skipped_muted=skipped_muted,
        jobs=jobs,
    )


@router.post("/videos/{video_id}/transcript-retranscribe", response_model=TranscriptRetranscriptionQueueResponse)
def queue_transcript_retranscription(
    video_id: int,
    request: TranscriptRetranscriptionQueueRequest,
    session: Session = Depends(get_session),
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    job, quality, _, _ = _queue_full_retranscription_job(
        session,
        video=video,
        force=bool(request.force),
        note=request.note,
        queued_from="video_retranscription",
    )
    spk_q._invalidate_speaker_query_caches()
    return TranscriptRetranscriptionQueueResponse(
        job_id=int(job.id),
        video_id=int(video_id),
        status="queued",
        recommended_tier=str(quality.get("recommended_tier") or "none"),
        quality_score=float(quality.get("quality_score") or 0.0),
        queued=True,
    )


@router.post("/channels/{channel_id}/transcript-retranscribe/queue", response_model=TranscriptRetranscriptionBulkQueueResponse)
def queue_channel_transcript_retranscriptions(
    channel_id: int,
    request: TranscriptRetranscriptionBulkQueueRequest,
    session: Session = Depends(get_session),
):
    channel = session.get(Channel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    videos = session.exec(
        select(Video)
        .where(Video.channel_id == channel_id)
        .order_by(Video.published_at.desc(), Video.id.desc())
        .limit(int(request.limit))
    ).all()

    jobs: list[TranscriptRetranscriptionQueueResponse] = []
    skipped_active = 0
    skipped_not_full_retranscription = 0
    skipped_unprocessed = 0
    skipped_muted = 0

    for video in videos:
        if bool(video.muted):
            skipped_muted += 1
            continue
        if not bool(video.processed):
            skipped_unprocessed += 1
            continue
        try:
            job, quality, _, _ = _queue_full_retranscription_job(
                session,
                video=video,
                force=bool(request.force_non_eligible),
                note=request.note,
                queued_from="channel_retranscription",
            )
            jobs.append(
                TranscriptRetranscriptionQueueResponse(
                    job_id=int(job.id),
                    video_id=int(video.id),
                    status="queued",
                    recommended_tier=str(quality.get("recommended_tier") or "none"),
                    quality_score=float(quality.get("quality_score") or 0.0),
                    queued=True,
                )
            )
        except HTTPException as exc:
            detail_text = str(exc.detail or "")
            if exc.status_code == 400 and "active job" in detail_text.lower():
                skipped_active += 1
            elif exc.status_code == 409 and "full_retranscription" in detail_text:
                skipped_not_full_retranscription += 1
            else:
                raise

    spk_q._invalidate_speaker_query_caches()
    return TranscriptRetranscriptionBulkQueueResponse(
        channel_id=int(channel_id),
        queued=len(jobs),
        skipped_active=skipped_active,
        skipped_not_full_retranscription=skipped_not_full_retranscription,
        skipped_unprocessed=skipped_unprocessed,
        skipped_muted=skipped_muted,
        jobs=jobs,
    )
