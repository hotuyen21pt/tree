import json

import pytest

from tree_cot_rt.data import Instance, Quad, read_jsonl
from tree_cot_rt.reasoning import (
    TREE_PATHS,
    chain_is_valid,
    majority_vote,
    parse_quads,
    path_constraints,
    reasoning_prompt,
    typed_instances,
)
from tree_cot_rt.rewards import consistency, group_advantages, score_chain


def make_instance(*, aspect=None, opinion="noisy") -> Instance:
    return Instance(
        "row-1",
        "The room was noisy.",
        (Quad(aspect, "FACILITY", opinion, "negative"),),
        aspect is None,
        opinion is None,
    )


IMPLICIT_ASPECT_CHAIN = json.dumps(
    {
        "reasoning": (
            "Step 1: the sentence explicitly states the opinion \"noisy\". "
            "Step 2: the implicit aspect is inferred because the noise is "
            "reported about the room itself. "
            "Step 3: the sentiment is negative. "
            "Step 4: the category is FACILITY."
        ),
        "quadruples": [
            {
                "aspect_term": None,
                "aspect_category": "FACILITY",
                "opinion_term": "noisy",
                "sentiment": "negative",
            }
        ],
    }
)


def test_quad_types_use_the_papers_four_names():
    assert set(TREE_PATHS) == {"EAEO", "EAIO", "IAEO", "IAIO"}


def test_jsonl_loader_preserves_implicit_flags(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "one",
                "text": "Noisy.",
                "quads": [
                    {
                        "aspect_term": None,
                        "aspect_category": "FACILITY",
                        "opinion_term": "Noisy",
                        "sentiment": "negative",
                        "aspect_implicit": True,
                    }
                ],
            }
        ) + "\n",
        encoding="utf-8",
    )
    row = read_jsonl(path)[0]
    assert row.quad_type == "IAEO"
    assert row.quads[0].aspect_term is None


def test_mixed_quad_types_are_split_per_quad():
    row = Instance(
        "mixed",
        "Room was clean.",
        (
            Quad("Room", "FACILITY", "clean", "positive"),
            Quad(None, "EXPERIENCE", "clean", "positive"),
            Quad("Room", "FACILITY", None, "positive"),
        ),
        True,
        True,
    )
    groups = {typed.quad_type: typed.quads for typed in typed_instances(row)}
    assert set(groups) == {"EAEO", "IAEO", "EAIO"}
    assert all(len(quads) == 1 for quads in groups.values())


def test_type_specific_prompt_follows_table_1_step_order():
    prompt = reasoning_prompt("Noisy.", "IAEO", ["FACILITY"])
    assert "aspect is implicit; opinion is explicit" in prompt
    assert "Allowed aspect categories" in prompt
    path = prompt.split("Reasoning path: ")[1].splitlines()[0]
    # IAEO is O -> infer A -> sentiment, category: the opinion must come first
    # and the sentiment must not be derived before the aspect is inferred.
    assert path.index("explicit opinion") < path.index("implicit aspect")
    assert path.index("implicit aspect") < path.index("sentiment")

    implicit_opinion = reasoning_prompt("The screen broke.", "EAIO")
    opinion_path = implicit_opinion.split("Reasoning path: ")[1].splitlines()[0]
    assert opinion_path.index("explicit aspect") < opinion_path.index("implicit opinion")
    assert opinion_path.index("implicit opinion") < opinion_path.index("sentiment")

    with pytest.raises(ValueError):
        reasoning_prompt("text", "invalid")


def test_type_agnostic_prompt_is_used_for_inference():
    prompt = reasoning_prompt("Noisy.", None, ["FACILITY"])
    assert "Tree type:" not in prompt
    for name in TREE_PATHS:
        assert f"- {name}:" in prompt


def test_parser_accepts_json_in_code_fence_and_rejects_invalid_quad():
    fenced = "```json\n" + IMPLICIT_ASPECT_CHAIN + "\n```"
    assert parse_quads(fenced)[0]["aspect_term"] is None
    assert chain_is_valid(fenced, "IAEO")
    assert not chain_is_valid(fenced, "EAEO")
    assert parse_quads('{"quadruples":[{"aspect_term":"room"}]}') is None


def test_validity_is_graded_not_all_or_nothing():
    # Partial step labelling fails coverage but keeps order and consistency.
    partial = json.dumps(
        {
            "reasoning": "Step 1: identify the opinion. Step 3: classify sentiment.",
            "quadruples": [
                {
                    "aspect_term": None,
                    "aspect_category": "FACILITY",
                    "opinion_term": "noisy",
                    "sentiment": "negative",
                }
            ],
        }
    )
    assert path_constraints(partial, "IAEO") == [False, True, True]
    assert path_constraints(IMPLICIT_ASPECT_CHAIN, "IAEO") == [True, True, True]
    assert path_constraints("not json at all", "IAEO") == [False, False, False]


def test_consistency_gives_partial_credit_per_element():
    gold = [
        {
            "aspect_term": "room",
            "aspect_category": "FACILITY",
            "opinion_term": "noisy",
            "sentiment": "negative",
        }
    ]
    three_of_four = [{**gold[0], "sentiment": "positive"}]
    nothing_right = [
        {
            "aspect_term": "lobby",
            "aspect_category": "FOOD",
            "opinion_term": "great",
            "sentiment": "positive",
        }
    ]
    assert consistency(gold, gold) == 1.0
    assert consistency(three_of_four, gold) == pytest.approx(0.75)
    assert consistency(nothing_right, gold) == 0.0
    # A spurious extra quad costs reward even when one quad is perfect.
    assert consistency(gold + nothing_right, gold) == pytest.approx(0.5)


def test_reward_components_follow_the_paper_definitions():
    row = make_instance(aspect=None)
    scores = score_chain(IMPLICIT_ASPECT_CHAIN, row, templates=["a totally different chain"])
    assert scores["consistency"] == 1.0
    assert scores["validity"] == 1.0
    assert scores["implicit"] == 1.0
    # R_impl(o) = 0 for EAEO, which has nothing implicit to justify.
    explicit = make_instance(aspect="room")
    explicit_chain = json.dumps(
        {
            "reasoning": "Step 1: a. Step 2: b. Step 3: c. Step 4: d.",
            "quadruples": [explicit.quads[0].as_dict()],
        }
    )
    assert score_chain(explicit_chain, explicit)["implicit"] == 0.0
    assert sum(group_advantages([1.0, 2.0, 3.0])) == pytest.approx(0.0)
    assert group_advantages([1.0, 1.0]) == [0.0, 0.0]


def test_implicit_reward_needs_sentence_grounded_evidence():
    row = make_instance(aspect=None)
    ungrounded = json.dumps(
        {
            "reasoning": (
                "Step 1: x. Step 2: the implicit aspect is inferred because of "
                "general knowledge about hotels. Step 3: y. Step 4: z."
            ),
            "quadruples": [row.quads[0].as_dict()],
        }
    )
    assert score_chain(ungrounded, row)["implicit"] < 1.0
    assert score_chain(IMPLICIT_ASPECT_CHAIN, row)["implicit"] == 1.0


def test_diversity_measures_distance_from_the_step_one_templates():
    row = make_instance(aspect=None)
    cloned = score_chain(IMPLICIT_ASPECT_CHAIN, row, templates=[IMPLICIT_ASPECT_CHAIN])
    distinct = score_chain(IMPLICIT_ASPECT_CHAIN, row, templates=["nothing alike here"])
    assert cloned["diversity"] == 1.0  # identical template is excluded, nothing left
    assert distinct["diversity"] > 0.5
    near_duplicate = score_chain(
        IMPLICIT_ASPECT_CHAIN,
        row,
        templates=[IMPLICIT_ASPECT_CHAIN.replace("negative", "neutral")],
    )
    assert near_duplicate["diversity"] < 0.2


def test_majority_vote_uses_majority_or_plurality():
    first = '{"quadruples":[{"aspect_term":"room","aspect_category":"FACILITY","opinion_term":"clean","sentiment":"positive"}]}'
    second = '{"quadruples":[{"aspect_term":"room","aspect_category":"FACILITY","opinion_term":"dirty","sentiment":"negative"}]}'
    assert majority_vote([first, first, second])[0]["sentiment"] == "positive"
    assert majority_vote([first, second]) == []
    assert majority_vote([first, "not JSON"])[0]["sentiment"] == "positive"
    wrong_type = '{"quadruples":[{"aspect_term":null,"aspect_category":"FACILITY","opinion_term":"clean","sentiment":"positive"}]}'
    assert majority_vote([first, wrong_type], expected_type="EAEO")[0]["sentiment"] == "positive"
