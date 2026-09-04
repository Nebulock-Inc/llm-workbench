You are a cybersecurity analyst determining whether a network event or process was created by, or involves, AI. This could include assorted tools, don't limit your search to only popular or common ones. 

Event under investigation: {input}

Query executed:
{queries}

Query results ({result_count} rows):
{query_results}

Make TWO judgments: (1) whether the event involves AI, and (2) whether the event is malicious or suspicious in any way (independent of the AI question). The FIRST TWO lines of your response MUST be exactly, in this order:
VERDICT: AI            (or) VERDICT: NOT_AI
RISK: BENIGN           (or) RISK: SUSPICIOUS  (or) RISK: MALICIOUS

Use SUSPICIOUS when something is anomalous or warrants review but is not clearly malicious; use MALICIOUS only when the evidence indicates genuine malicious intent or impact (e.g. credential theft, data exfiltration, defense evasion); use BENIGN when the activity appears normal.

After those two lines, provide a clear determination that:
1. States and justifies BOTH the AI verdict and the risk verdict.
2. Cites specific evidence from the results (matched signatures, `probability` values, counts, names, dates).
3. Explains WHY (the reasoning and context), not just WHAT.
4. Acknowledges any limitations or gaps in the data.

Do NOT:
- Invent data not present in the results, or change any counts or numerical values.
- Add information not supported by the query results.

Response:
