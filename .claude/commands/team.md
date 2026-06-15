---
description: Run the alpha_lab cross-functional team (research-lead, quant-skeptic, engineer, execution-safety) over a task or the current diff, then synthesize.
argument-hint: <task, or empty to review the current diff>
---

Run the alpha_lab product team on: $ARGUMENTS

1. **Scout inline first** to scope it — list the files / find the diff (`git --no-pager diff`) / note which legs are touched. Don't fan out blind.
2. **Call the Workflow tool** with the saved workflow:
   `Workflow({ name: 'product-team', args: { task: "<the task, or 'review the current diff'>", focus: "<optional sharpening>" } })`.
   It fans the four role-agents in `.claude/agents/` over the task in parallel (each through its own lens) and returns a synthesis (blockers + go/no-go + next steps).
3. **Read the synthesis, then stay in the loop.** For anything beyond review (design / build): implement the agreed change *yourself* (the execution leg is safety-sensitive — don't let parallel agents write to it), then re-run the team to adversarially verify the diff. Iterate until blockers are cleared.
4. **Scale to the task:** a small change → a single lens pass; a big audit or feature → multiple rounds (design → build → verify), like a real sprint.

Guardrails: the team costs tokens and is opt-in — do NOT fan out for trivial or conversational work; just do those solo. The value is decorrelated adversarial review, not persona role-play.
