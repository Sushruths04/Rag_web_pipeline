from rag_gt.pipeline.gt_pipeline import (
    _candidate_submit_limit,
    _deadline_from_minutes,
    _is_semantic_duplicate_embedding,
    _normalize_question_for_dedup,
    _tf_sfg_stage_deadline,
    _target_question_count,
)


def test_requested_questions_get_review_buffer():
    cfg = {
        "question_generation": {
            "output_buffer_multiplier": 1.25,
            "candidate_submit_multiplier": 4,
        }
    }

    target = _target_question_count(60, cfg)

    assert target == 75
    assert _candidate_submit_limit(target, cfg) == 300


def test_question_target_never_goes_below_request():
    cfg = {"question_generation": {"output_buffer_multiplier": 0.5}}

    assert _target_question_count(60, cfg) == 60


def test_qrsg_candidate_multiplier_overrides_base_when_enabled():
    cfg = {
        "question_generation": {"candidate_submit_multiplier": 2},
        "v13_qrsg": {"enabled": True, "candidate_submit_multiplier": 5},
    }

    assert _candidate_submit_limit(10, cfg) == 50


def test_semantic_duplicate_embedding_threshold():
    seen = [[1.0, 0.0], [0.0, 1.0]]

    assert _is_semantic_duplicate_embedding([0.93, 0.1], seen, 0.92)
    assert not _is_semantic_duplicate_embedding([0.2, 0.1], seen, 0.92)


def test_deadline_from_minutes_disabled_by_default():
    assert _deadline_from_minutes(100.0, None) is None
    assert _deadline_from_minutes(100.0, 0) is None
    assert _deadline_from_minutes(100.0, 2) == 220.0


def test_tf_sfg_stage_deadline_reserves_question_generation_time():
    deadline = 3600.0
    assert _tf_sfg_stage_deadline(
        deadline,
        0.0,
        max_wall_fraction=0.35,
        reserve_generation_minutes=20.0,
    ) == 1260.0
    assert _tf_sfg_stage_deadline(
        600.0,
        0.0,
        max_wall_fraction=0.35,
        reserve_generation_minutes=20.0,
    ) == 210.0


def test_question_dedup_normalizes_unicode_dash_variants():
    q1 = "What role does the policy play in determining which state–action pairs are visited?"
    q2 = "What role does the policy play in determining which state-action pairs are visited?"

    assert _normalize_question_for_dedup(q1) == _normalize_question_for_dedup(q2)
