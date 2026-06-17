// Controlled A/B/C: build the SAME neutral feature three ways to measure whether team
// mode (and per-role tuning) earns its cost in alpha_lab. See REPORT.md for the design,
// the grading rubric, and the (rate-limit-blocked) first run.
//
// Run after the session limit resets:
//   Workflow({ scriptPath: "<abs path to this file>" })
// then grade each arm's experiments/team_eval/arm_*/perf.py against a known-answer equity
// series (excess-of-rf? √365 on daily marks? de-fauceted via all_strategy_equity?).

export const meta = {
  name: 'team-eval-experiment',
  description: 'Controlled A/B/C experiment: build the SAME neutral feature (a paper-bot performance summary) three ways — solo (no review), the project-tuned 4-role team, and a generic builder+reviewer team — in isolated dirs, to measure empirically whether team mode and role-tuning earn their token cost in alpha_lab.',
  phases: [
    { title: 'ArmA-solo', detail: 'one generic agent builds it end-to-end, no reviewer' },
    { title: 'ArmB-tuned-team', detail: 'engineer builds; quant-skeptic + execution-safety + research-lead review; engineer revises' },
    { title: 'ArmC-generic-team', detail: 'generic builder + one generic reviewer + revise' },
  ],
}

const PY = 'D:\\conda\\envs\\py313\\python.exe'

function spec(arm) {
  return [
    'FEATURE — bot performance summary (execution leg).',
    'In src/quant_bot_manager, add a performance summary for a paper-trading bot. The per-bot store',
    '(quant_bot_manager.core.store.Store) already records an equity time series. Given a bot, read its',
    'equity history and compute four headline metrics: annualized return, annualized volatility,',
    'Sharpe ratio, and maximum drawdown.',
    '',
    'Deliver:',
    '  1. A small, typed metric CORE: a pure function over an equity series (unit-testable without a live bot).',
    '  2. A thin store-backed wrapper that produces the summary for a bot.',
    '  3. A test, using the store tmp_path fixture pattern: Store("t", path=tmp_path/"bot.db"); .append_equity(ts,total,fut,spot).',
    '',
    'Constraints:',
    '  - Write ALL files ONLY under experiments/team_eval/' + arm + '/ . Module = perf.py, test = test_perf.py.',
    '    Do NOT edit, create, or delete anything outside that directory. No __init__.py; the test sits beside',
    '    perf.py so it can `from perf import ...`.',
    '  - Ship it the way it SHOULD ship in THIS repo (its conventions are in CLAUDE.md / AGENTS.md): small,',
    '    typed, pure where practical; reuse existing store helpers; do not overengineer.',
    '  - Interpreter ' + PY + ' . Before declaring done, run and make green:',
    '      & ' + PY + ' -m pytest experiments/team_eval/' + arm + '/ -q',
    '      & ' + PY + ' -m ruff check experiments/team_eval/' + arm + '/',
    '  - Other teams are solving the SAME task in sibling experiments/team_eval/* dirs. Do NOT read them.',
    '    Solve it independently.',
    '',
    'Report at the end: the function signature(s) you shipped; the key assumptions/preconditions you made;',
    'and ONE sentence on how you would judge whether the resulting Sharpe is "good".',
  ].join('\n')
}

function reviewPrompt(arm) {
  return [
    'A teammate implemented a bot performance summary at experiments/team_eval/' + arm + '/perf.py',
    '(with test_perf.py beside it). It computes annualized return, volatility, Sharpe, and max drawdown',
    'for a paper-trading crypto bot from its stored equity path.',
    'READ those two files (and any store/runner code you need), then analyze ONLY through your role lens.',
    'Cite file:line. Mark anything that would make a shipped number wrong or a gate weak as a blocker.',
  ].join('\n')
}

function genericReviewPrompt(arm) {
  return [
    'Review the code at experiments/team_eval/' + arm + '/perf.py and its test test_perf.py.',
    'It computes annualized return, annualized volatility, Sharpe ratio, and maximum drawdown for a',
    'trading bot from an equity series. Read both files. Check correctness, bugs, edge cases, numeric',
    'soundness, code quality, and whether the test actually exercises the logic. List concrete issues',
    'with file:line and a severity, and give a bottom-line verdict.',
  ].join('\n')
}

function revisePrompt(arm, reviews) {
  return [
    'Reviewers analyzed your implementation at experiments/team_eval/' + arm + '/ . Their findings (JSON):',
    JSON.stringify(reviews),
    'Address every blocker and high finding by editing experiments/team_eval/' + arm + '/perf.py and',
    'test_perf.py (stay inside that dir). Then re-run pytest + ruff (interpreter ' + PY + ') and make them',
    'green. Report what you changed and why.',
  ].join('\n')
}

const BUILD = {
  type: 'object', additionalProperties: false,
  required: ['arm', 'signature', 'assumptions', 'sharpe_good', 'tests_pass'],
  properties: {
    arm: { type: 'string' },
    signature: { type: 'string', description: 'the main function signature(s) you shipped' },
    assumptions: { type: 'string', description: 'key assumptions / preconditions you made' },
    sharpe_good: { type: 'string', description: 'one sentence: how you judge whether the Sharpe is good' },
    tests_pass: { type: 'boolean' },
    notes: { type: 'string', description: 'anything else worth knowing' },
  },
}

const LENS = {
  type: 'object', additionalProperties: false,
  required: ['role', 'summary', 'findings', 'verdict'],
  properties: {
    role: { type: 'string' },
    summary: { type: 'string' },
    findings: {
      type: 'array', items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'point', 'evidence', 'suggestion'],
        properties: {
          severity: { type: 'string', enum: ['blocker', 'high', 'medium', 'low'] },
          point: { type: 'string' },
          evidence: { type: 'string' },
          suggestion: { type: 'string' },
        },
      },
    },
    verdict: { type: 'string' },
  },
}

// --- Arm A: solo, no review ---------------------------------------------
const armA = () => agent(
  spec('arm_a') + '\n\nBuild this end-to-end by YOURSELF. No reviewer will check your work, so ship it correct.',
  { agentType: 'general-purpose', label: 'A:solo-build', phase: 'ArmA-solo', schema: BUILD })

// --- Arm B: the project-tuned 4-role team -------------------------------
const armB = async () => {
  const build = await agent(
    spec('arm_b') + '\n\nYou are the implementer; the team will adversarially review your diff after, so build it right.',
    { agentType: 'engineer', label: 'B:build', phase: 'ArmB-tuned-team', schema: BUILD })
  const reviews = (await parallel([
    () => agent(reviewPrompt('arm_b'), { agentType: 'quant-skeptic', label: 'B:skeptic', phase: 'ArmB-tuned-team', schema: LENS }),
    () => agent(reviewPrompt('arm_b'), { agentType: 'execution-safety', label: 'B:safety', phase: 'ArmB-tuned-team', schema: LENS }),
    () => agent(reviewPrompt('arm_b'), { agentType: 'research-lead', label: 'B:lead', phase: 'ArmB-tuned-team', schema: LENS }),
  ])).filter(Boolean)
  const revise = await agent(revisePrompt('arm_b', reviews),
    { agentType: 'engineer', label: 'B:revise', phase: 'ArmB-tuned-team', schema: BUILD })
  return { build, reviews, revise }
}

// --- Arm C: a DIFFERENT team definition — generic builder + generic reviewer ---
const armC = async () => {
  const build = await agent(spec('arm_c'),
    { agentType: 'general-purpose', label: 'C:build', phase: 'ArmC-generic-team', schema: BUILD })
  const review = await agent(genericReviewPrompt('arm_c'),
    { agentType: 'general-purpose', label: 'C:review', phase: 'ArmC-generic-team', schema: LENS })
  const revise = await agent(revisePrompt('arm_c', [review]),
    { agentType: 'general-purpose', label: 'C:revise', phase: 'ArmC-generic-team', schema: BUILD })
  return { build, review, revise }
}

const arms = await parallel([armA, armB, armC])
return { A: arms[0], B: arms[1], C: arms[2] }
