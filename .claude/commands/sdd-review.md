Review the current implementation against:
- CLAUDE.md
- .claude/specs/001-rosbot-navigation.md
- the latest approved milestone

Check for:
1. Use of forbidden Supervisor API.
2. Hardcoded maze-specific behavior.
3. Missing sensor/device initialization.
4. Unsafe behavior around obstacles or green ground.
5. Code complexity or poor modularity.
6. Whether acceptance criteria are met.

Do not implement changes unless explicitly asked.

Return:
- Pass