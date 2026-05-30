from grader.demo.golden_loader import golden_predict_demo_path, load_golden_predict_demo


def test_main_golden_has_examples() -> None:
    data = load_golden_predict_demo()
    examples = data.get("examples")
    assert isinstance(examples, list)
    assert len(examples) >= 2


def test_golden_path_exists() -> None:
    assert golden_predict_demo_path().is_file()
