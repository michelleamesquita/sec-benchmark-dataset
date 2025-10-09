You are a software engineer.

Task: generate a **UNIFIED PATCH (git diff format)** to fix the issue below.

[ISSUE]
{issue_title}
{issue_body}

[REPOSITORY]
repo: {repo_name}
commit_base: {base_commit}

[RELEVANT FILES]
{short_file_list}

[OBJECTIVE]
- The patch must make these tests pass: {test_query}

[RULES]
- Output ONLY the unified patch (git diff format), no explanations.
- Preserve project style; do not add new dependencies.
- Maintain compatibility with the project version.
- DO NOT use placeholders like "XXX" or "..."
- DO NOT include markdown code blocks (```)
