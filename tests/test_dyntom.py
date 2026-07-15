import json

from lm_eval.tasks.dyntom import utils as dyntom_utils


def test_parse_answers_final_json_overrides_echoed_example():
    gen = (
        'For example: {"type_d_how_1":"a"}\n'
        'Final answer: {"type_d_how_1":"c","type_a_what_2":"D"}'
    )

    assert dyntom_utils._parse_answers(gen) == {
        "type_d_how_1": "c",
        "type_a_what_2": "d",
    }


def test_parse_answers_cot_prose_plus_final_json():
    gen = (
        "I compare the events first, then answer.\n"
        '```json\n{"type_a_what_1":"B"}\n```'
    )

    assert dyntom_utils._parse_answers(gen) == {"type_a_what_1": "b"}


def test_parse_answers_repeated_bare_pairs_later_wins():
    gen = "type_c_how_3: a\nAfter checking again, type_c_how_3: d"

    assert dyntom_utils._parse_answers(gen) == {"type_c_how_3": "d"}


def test_process_results_scores_missing_answers_wrong():
    doc = {
        "meta": json.dumps(
            [
                {
                    "id": "type_a_what_1",
                    "gold": "b",
                    "family": "type_a_what",
                    "dim": "u",
                    "subject": "belief",
                }
            ]
        )
    }

    assert dyntom_utils.process_results(doc, ["no parseable answer here"]) == {
        "acc": 0.0,
        "acc_core": 0.0,
        "acc_belief_u": 0.0,
    }
