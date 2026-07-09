from lm_eval.tasks.bigtom import utils as bigtom_utils


def _payload(raw_idx, condition, acc, pair_kind="belief"):
    return {
        "raw_idx": raw_idx,
        "variable": "forward_belief",
        "init_belief": 0,
        "pair_kind": pair_kind,
        "condition": condition,
        "acc": float(acc),
    }


def test_tb_fb_aggregator_truth_table():
    assert bigtom_utils.agg_tb_fb_acc(
        [_payload(0, "true_belief", 1), _payload(0, "false_belief", 1)]
    ) == 1.0
    assert bigtom_utils.agg_tb_fb_acc(
        [_payload(0, "true_belief", 0), _payload(0, "false_belief", 1)]
    ) == 0.0
    assert bigtom_utils.agg_tb_fb_acc(
        [_payload(0, "true_belief", 1), _payload(0, "false_belief", 0)]
    ) == 0.0
    assert bigtom_utils.agg_tb_fb_acc(
        [_payload(0, "true_belief", 0), _payload(0, "false_belief", 0)]
    ) == 0.0


def test_tb_fb_aggregator_pairs_by_raw_idx_not_order():
    items = [
        _payload(0, "true_belief", 1),
        _payload(1, "false_belief", 1),
        _payload(1, "true_belief", 0),
        _payload(0, "false_belief", 0),
    ]

    assert bigtom_utils.agg_tb_fb_acc(items) == 0.0


def test_load_paired_row_counts_and_pair_metadata():
    dataset = bigtom_utils.load_paired(
        variable="forward_belief", init_belief=0, pair_kind="belief"
    )["train"]

    assert len(dataset) == 400
    true_rows = [row for row in dataset if row["condition"] == "true_belief"]
    false_rows = [row for row in dataset if row["condition"] == "false_belief"]
    assert len(true_rows) == 200
    assert len(false_rows) == 200
    assert {row["raw_idx"] for row in true_rows} == set(range(200))
    assert {row["raw_idx"] for row in false_rows} == set(range(200))
    assert {row["pair_kind"] for row in dataset} == {"belief"}


def test_load_variable_preserves_old_cell_counts():
    dataset = bigtom_utils.load_variable(variable="forward_belief")["train"]

    assert len(dataset) == 1600
    for init_belief in (0, 1):
        for condition in (
            "true_belief",
            "false_belief",
            "true_control",
            "false_control",
        ):
            rows = [
                row
                for row in dataset
                if row["init_belief"] == init_belief and row["condition"] == condition
            ]
            assert len(rows) == 200
            assert {row["raw_idx"] for row in rows} == set(range(200))


def test_marginal_processor_remains_scalar_acc_and_error_rate_only():
    doc = bigtom_utils.load(
        variable="forward_belief", init_belief=0, condition="true_belief"
    )["train"][0]
    response = f"Answer: {doc['gold_letter']}){doc['true_text']}"

    assert bigtom_utils.process_results_vanilla(doc, [response]) == {
        "acc": 1.0,
        "error_rate": 0.0,
    }


def test_variable_processor_emits_overall_cell_and_paired_metrics():
    doc = bigtom_utils.load_variable(variable="forward_belief")["train"][0]
    response = f"Answer: {doc['gold_letter']}){doc['true_text']}"

    metrics = bigtom_utils.process_results_variable_vanilla(doc, [response])

    assert metrics["acc"] == 1.0
    assert metrics["error_rate"] == 0.0
    assert metrics["acc_0_true_belief"] == 1.0
    assert metrics["error_rate_0_true_belief"] == 0.0
    assert metrics["tb_fb_acc_0_belief"] == {
        "raw_idx": 0,
        "variable": "forward_belief",
        "init_belief": 0,
        "pair_kind": "belief",
        "condition": "true_belief",
        "acc": 1.0,
    }


def test_chat_1shot_cot_current_parser_can_disagree_with_source_style_full_response():
    doc = {
        "gold_letter": "a",
        "other_letter": "b",
        "true_text": "the correct final answer",
        "wrong_text": "the tempting earlier answer",
    }
    response = (
        "Thought: a) the correct final answer seems plausible at first.\n"
        "Answer: b) the tempting earlier answer"
    )

    assert bigtom_utils.process_results_cot(doc, [response]) == {
        "acc": 0.0,
        "error_rate": 0.0,
    }
    assert (
        bigtom_utils._grade(
            bigtom_utils._strip_thinking(response),
            doc["gold_letter"],
            doc["other_letter"],
            doc["true_text"],
            doc["wrong_text"],
        )
        == "correct"
    )
