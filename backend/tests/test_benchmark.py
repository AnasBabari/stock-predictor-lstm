from benchmark import build_parser


def test_benchmark_parser_builds_reproducible_defaults():
    args = build_parser().parse_args([])
    assert args.ticker == "AAPL"
    assert args.lookback == 60
    assert args.horizons == (1, 5, 20)
    assert args.target_type == "log_return"
    assert "price" in args.feature_sets
