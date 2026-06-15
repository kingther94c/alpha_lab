export const meta = {
  name: 'product-team',
  description: 'Cross-functional pass over a task or diff: research-lead, quant-skeptic, engineer, execution-safety each analyze from their own lens (.claude/agents/), then a lead synthesizes blockers + a go/no-go.',
  phases: [
    { title: 'Lenses', detail: 'four project-tuned roles analyze in parallel' },
    { title: 'Synthesize', detail: 'merge into blockers + go/no-go + next steps' },
  ],
}

// args (from `/team` or the Workflow call): { task: string, focus?: string }
const task = (args && args.task) || 'Review the current working-tree change (run `git --no-pager diff`).'
const focus = (args && args.focus) ? `\nFOCUS: ${args.focus}` : ''

const LENS = {
  type: 'object', additionalProperties: false,
  required: ['role', 'summary', 'findings', 'verdict'],
  properties: {
    role: { type: 'string' },
    summary: { type: 'string', description: '2-3 sentences from this lens (or "not applicable" + why)' },
    findings: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'point', 'evidence', 'suggestion'],
        properties: {
          severity: { type: 'string', enum: ['blocker', 'high', 'medium', 'low'] },
          point: { type: 'string' },
          evidence: { type: 'string', description: 'file:line or concrete observation' },
          suggestion: { type: 'string' },
        },
      },
    },
    verdict: { type: 'string', description: "this lens's bottom line" },
  },
}

const ROLES = [
  { at: 'research-lead', lens: 'Is the question/goal well-framed, the economic mechanism sound, success criteria + holdout discipline respected? Give a go/no-go.' },
  { at: 'quant-skeptic', lens: 'Hunt leakage/lookahead, cost-of-cash, full-sample stats, survivorship. Is any number believable? Run scan_leakage.py if code paths are in scope.' },
  { at: 'engineer', lens: "Placement (right leg/submodule), don't-overengineer, small/typed/tested, two-leg decoupling, feasibility." },
  { at: 'execution-safety', lens: 'ONLY if the task touches src/quant_bot_manager: kill-switch integrity, never-auto-live gating, demo default, secrets/raw-data hygiene. Otherwise say "not applicable".' },
]

phase('Lenses')
const reviews = (await parallel(ROLES.map(r => () =>
  agent(`TASK:\n${task}${focus}\n\nAnalyze ONLY through your role's lens: ${r.lens}`,
    { agentType: r.at, label: `lens:${r.at}`, phase: 'Lenses', schema: LENS }),
))).filter(Boolean)

phase('Synthesize')
const synthesis = await agent(
  `You are the integrating lead. Merge these four role reviews into: (1) the deduped blocker + high findings, ranked; (2) ONE blunt go/no-go call; (3) an ordered next-step list. Drop "not applicable" lenses. Be concise.\n\nREVIEWS (JSON):\n${JSON.stringify(reviews)}`,
  { agentType: 'research-lead', label: 'synthesize', phase: 'Synthesize' })

return { reviews, synthesis }
