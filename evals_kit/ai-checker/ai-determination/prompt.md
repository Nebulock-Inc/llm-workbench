You are evaluating a security analyst's determination of whether an event was created by, or involves, AI. "AI" spans any AI assistant, coding agent, autonomous agent, automation framework, or script-driven tool — common, uncommon, or unknown. Well-known products (e.g. Claude Code, Copilot, Cursor) are only illustrative examples; do NOT treat them as the only qualifying tools, and do not penalize a correct identification of a less common tool.

Judge ONLY whether the determination is correct and well-supported by the query results provided. The response should open with a verdict line `VERDICT: AI` or `VERDICT: NOT_AI`.

Score on:
- Correctness: Is the AI / NOT_AI verdict the right call given the evidence in the results?
- Grounding: Does the reasoning cite specific signals actually present in the results (matched command-line/path signatures, parent process, classifier probability, counts)?
- Consistency: Does the reasoning avoid contradicting the data or inventing facts not in the results?

Do NOT evaluate:
- Tone or writing style.
- Whether the SQL query itself was optimal (separate eval).

If the query returned no rows, a well-reasoned NOT_AI or an explicit "insufficient evidence" determination is acceptable; penalize confident claims that the data cannot support.

<event>
Event under investigation: {input}
</event>

<query_results>
Data the query returned: {query_results}
</query_results>

<agent_response>
Determination: {output}
</agent_response>
