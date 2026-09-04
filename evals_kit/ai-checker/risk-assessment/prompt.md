You are evaluating a security analyst's opinion on whether an event is malicious or suspicious. The response should contain a risk line `RISK: BENIGN`, `RISK: SUSPICIOUS`, or `RISK: MALICIOUS` (this is separate from the AI determination on the VERDICT line).

Judge ONLY whether the risk opinion is well-calibrated and supported by the query results provided.

Score on:
- Calibration: Is the chosen level (benign / suspicious / malicious) appropriate for the evidence? MALICIOUS should require indicators of genuine malicious intent or impact (credential theft, data exfiltration, defense evasion); SUSPICIOUS fits anomalous-but-unproven activity; BENIGN fits normal activity. Penalize both over-statement and under-statement.
- Grounding: Does the reasoning cite specific signals actually present in the results?
- Consistency: Does the reasoning avoid contradicting the data or inventing facts not in the results?

Do NOT evaluate:
- Tone or writing style.
- The AI/NOT_AI determination (separate eval) or the SQL query quality (separate eval).

If the query returned no rows, a well-reasoned BENIGN or an explicit "insufficient evidence" opinion is acceptable; penalize confident risk claims the data cannot support.

<event>
Event under investigation: {input}
</event>

<query_results>
Data the query returned: {query_results}
</query_results>

<agent_response>
Determination: {output}
</agent_response>
