from rag_gt.cli.generate import _build_parser


def test_generate_cli_accepts_pair_budget_override():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--input_dir",
            "data/in",
            "--output",
            "data/out.jsonl",
            "--v16",
            "--pair_budget",
            "50",
        ]
    )
    assert args.pair_budget == 50


def test_generate_cli_accepts_paid_probe_safety_controls():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--input_dir",
            "data/in",
            "--output",
            "data/out.jsonl",
            "--v16",
            "--max_live_api_calls",
            "12",
            "--disable_v16_singlehop_fallback",
            "--disable_v16_twins",
        ]
    )
    assert args.max_live_api_calls == 12
    assert args.disable_v16_singlehop_fallback is True
    assert args.disable_v16_twins is True
