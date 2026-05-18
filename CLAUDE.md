# frugal-ml

## gstack

Use `/browse` from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills:
- `/office-hours` — office hours facilitation
- `/plan-ceo-review` — CEO review planning
- `/plan-eng-review` — engineering review planning
- `/plan-design-review` — design review planning
- `/design-consultation` — design consultation
- `/design-shotgun` — rapid design exploration
- `/design-html` — HTML design generation
- `/review` — code review
- `/ship` — ship a feature
- `/land-and-deploy` — land and deploy changes
- `/canary` — canary deployment
- `/benchmark` — benchmarking
- `/browse` — headless browser for web browsing and QA
- `/connect-chrome` — connect to Chrome
- `/qa` — QA testing
- `/qa-only` — QA without implementation
- `/design-review` — design review
- `/setup-browser-cookies` — configure browser cookies
- `/setup-deploy` — configure deployment
- `/setup-gbrain` — configure gbrain
- `/retro` — retrospective
- `/investigate` — investigation and debugging
- `/document-release` — release documentation
- `/document-generate` — documentation generation
- `/codex` — codex operations
- `/cso` — CSO operations
- `/autoplan` — automated planning
- `/plan-devex-review` — developer experience review planning
- `/devex-review` — developer experience review
- `/careful` — careful mode for risky changes
- `/freeze` — freeze deployments
- `/guard` — guard mode
- `/unfreeze` — unfreeze deployments
- `/gstack-upgrade` — upgrade gstack
- `/learn` — learning and documentation

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
