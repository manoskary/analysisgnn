from analysisgnn.train.train_analysisgnn import get_parser


def test_seed_flag_defaults_to_zero():
    parser = get_parser()
    args = parser.parse_args([])
    assert args.seed == 0


def test_seed_flag_accepts_override():
    parser = get_parser()
    args = parser.parse_args(["--seed", "4"])
    assert args.seed == 4
