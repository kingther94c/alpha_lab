# docs/

Durable project knowledge for alpha_lab. Read order for a new contributor
(or agent):

1. [Architecture](architecture/alpha_lab_architecture.md) — where things
   live and why.
2. [Research artifact contracts](contracts/research_artifacts.md) — shapes
   and dtypes for the things that flow between notebooks and `src/`.
3. [Strategy research notebook template](../notebooks/_templates/strategy_research_template.md) —
   the outline every new study should start from.
4. [Decision record template](research_decisions/template.md) — how to
   close a study with `accept` / `accept_monitoring` / `needs_revision` /
   `reject` / `park`.
5. [Notebook → package backlog](backlog/notebook_to_package_backlog.md) —
   running list of patterns to lift from notebooks into `src/alpha_lab/`.
6. [Roadmap](ROADMAP.md) — milestone plan for `src/` capability growth.

`docs/` is for knowledge that should outlive any single notebook or commit.
Status updates and short-lived notes belong in commit messages, not here.
