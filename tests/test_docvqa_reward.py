import unittest

from rewards.docvqa_grpo_reward import compute_reward, reward_func
from rewards.easyr1_docvqa_reward import compute_score as easyr1_compute_score


class DocVQARewardTest(unittest.TestCase):
    def test_exact_and_grounded_reward(self):
        result = compute_reward(
            "<think>I found the total 1450.00 in the invoice.</think><answer>1450.00</answer>",
            ["1450.00"],
        )
        self.assertEqual(result.reward, 2.0)
        self.assertEqual(result.accuracy_reward, 1.5)
        self.assertEqual(result.grounding_reward, 0.5)

    def test_wrong_decimal_gets_no_accuracy_reward(self):
        result = compute_reward(
            "<think>I found 1450.00.</think><answer>145.00</answer>",
            ["1450.00"],
        )
        self.assertEqual(result.accuracy_reward, 0.0)
        self.assertEqual(result.grounding_reward, 0.5)

    def test_missing_answer_block_gets_no_accuracy_reward(self):
        result = compute_reward("<think>1450.00</think>1450.00", ["1450.00"])
        self.assertEqual(result.accuracy_reward, 0.0)
        self.assertFalse(result.has_single_answer_block)

    def test_short_grounding_answer_is_skipped(self):
        result = compute_reward("<think>A appears here.</think><answer>A</answer>", ["A"])
        self.assertEqual(result.accuracy_reward, 1.5)
        self.assertEqual(result.grounding_reward, 0.0)

    def test_reward_func_batch(self):
        rewards = reward_func(
            [
                "<think>San Diego</think><answer>San Diego</answer>",
                "<think>San Diego</think><answer>Los Angeles</answer>",
            ],
            answers=[["San Diego"], ["San Diego"]],
        )
        self.assertEqual(rewards, [2.0, 0.5])

    def test_easyr1_reward_adapter(self):
        scores = easyr1_compute_score(
            [
                {
                    "response": "<think>The answer San Diego is visible.</think><answer>San Diego</answer>",
                    "ground_truth": ["San Diego"],
                }
            ]
        )
        self.assertEqual(scores[0]["overall"], 2.0)
        self.assertEqual(scores[0]["accuracy"], 1.5)
        self.assertEqual(scores[0]["grounding"], 0.5)


if __name__ == "__main__":
    unittest.main()
