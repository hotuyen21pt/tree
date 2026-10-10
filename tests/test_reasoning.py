import json

import pytest

from tree_cot_rt.data import Instance, Quad, read_jsonl
from tree_cot_rt.reasoning import (
    chain_is_valid,
    majority_vote,
    parse_quads,
    reasoning_prompt,
    typed_instances,
)
from tree_cot_rt.rewards import group_advantages, score_chain


def make_instance(*, aspect=None, opinion="noisy") -> Instance:
    return Instance(
        "row-1",
        "The room was noisy.",
        (Quad(aspect, "FACILITY", opinion, "negative"),),
        aspect is None,
        opinion is None,
    )


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
    assert row.quad_type == "IE"
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
    assert set(groups) == {"EE", "EA", "IE"}
    assert all(len(quads) == 1 for quads in groups.values())


def test_prompt_selects_type_specific_tree():
    prompt = reasoning_prompt("Noisy.", "IA", ["FACILITY"])
    assert "aspect is implicit; opinion is implicit" in prompt
    assert "Allowed aspect categories" in prompt
    with pytest.raises(ValueError):
        reasoning_prompt("text", "invalid")


def test_parser_accepts_json_in_code_fence_and_rejects_invalid_quad():
    valid = '```json\n{"reasoning":"Step 1: identify the opinion. Step 2: infer the aspect. Step 3: classify sentiment. Step 4: assign category because the room is noisy.","quadruples":[{"aspect_term":null,"aspect_category":"FACILITY","opinion_term":"noisy","sentiment":"negative"}]}\n```'
    assert parse_quads(valid)[0]["aspect_term"] is None
    assert chain_is_valid(valid, "IE")
    assert not chain_is_valid(valid, "EE")
    assert parse_quads('{"quadruples":[{"aspect_term":"room"}]}') is None


def test_reward_scores_alignment_validity_and_group_normalization():
    row = make_instance(aspect=None)
    chain = json.dumps(
        {
            "reasoning": (
                "Step 1: identify the opinion. Step 2: infer the aspect. "
                "Step 3: classify sentiment. Step 4: assign category because "
                "the sentence reports noise."
            ),
            "quadruples": [row.quads[0].as_dict()],
        }
    )
    rewards = score_chain(chain, row, [chain, chain + " "])
    assert rewards["consistency"] == 1.0
    assert rewards["validity"] == 1.0
    assert rewards["implicit"] == 1.0
    assert sum(group_advantages([1.0, 2.0, 3.0])) == pytest.approx(0.0)
    assert group_advantages([1.0, 1.0]) == [0.0, 0.0]


def test_majority_vote_uses_majority_or_plurality():
    first = '{"quadruples":[{"aspect_term":"room","aspect_category":"FACILITY","opinion_term":"clean","sentiment":"positive"}]}'
    second = '{"quadruples":[{"aspect_term":"room","aspect_category":"FACILITY","opinion_term":"dirty","sentiment":"negative"}]}'
    assert majority_vote([first, first, second])[0]["sentiment"] == "positive"
    assert majority_vote([first, second]) == []
    assert majority_vote([first, "not JSON"])[0]["sentiment"] == "positive"
    wrong_type = '{"quadruples":[{"aspect_term":null,"aspect_category":"FACILITY","opinion_term":"clean","sentiment":"positive"}]}'
    assert majority_vote([first, wrong_type], expected_type="EE")[0]["sentiment"] == "positive"
