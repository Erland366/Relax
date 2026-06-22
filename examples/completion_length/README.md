# Completion Length Reward Example

This is the smallest built-in reward task for checking whether Relax can learn a
simple preference. The reward is:

```text
reward = -len(completion_tokens)
```

Use `--rm-type completion_length` to enable it. The reward reads
`Sample.response_length`, which is the generated completion token count recorded
by rollout. If the policy emits only EOS, the completion length is 1 and the
reward is `-1`; any longer completion receives a lower reward.

## Dataset

`prompts.jsonl` contains repeated message-format prompts. The `label` field is
present only to satisfy launch scripts that expect one; the reward ignores it.
`prompts_end_response.jsonl` is an alternate short instruction prompt for
sampling probes where the baseline prompt still generates to the response cap.

## Launch Snippet

Use these task-level arguments in a normal training launch script:

```bash
--prompt-data examples/completion_length/prompts.jsonl
--input-key prompt
--label-key label
--apply-chat-template
--rm-type completion_length
```

Do not pass `--reward-key` for this task because the reward is a scalar.

When W&B is enabled, watch `rollout/reward/mean` and `rollout/response_len/mean`.
Learning the task means `rollout/reward/mean` moves upward toward `-1` while
`rollout/response_len/mean` moves downward toward 1.
