Evaluate whether the response is factually accurate based on the query results. Check two dimensions:

1. **Qualitative accuracy**: No invented entities, rules, signals, relationships, or claims
2. **Quantitative accuracy**: Counts, dates, severity levels, and other numerical attributes match query results exactly

Examples:
- Score 5: Query returned 3 hunt reports from July 2026. Response: "Found 3 completed hunts from July 2026."
- Score 3: Query returned 3 hunt reports. Response: "Found 5 hunt reports." (wrong count)
- Score 3: Query shows no findings. Response: "The system detected suspicious activity." (invented claim)
- Score 1: Query returned hunt reports. Response: "Multiple critical vulnerabilities were exploited." (invented severity/exploitation)

<analyst_prompt>
User Question: {input}
</analyst_prompt>

<agent_queries>
Queries Executed: {queries}
</agent_queries>

<agent_query_results>
Query Results: {query_results}
</agent_query_results>

<agent_response>
Response: {output}
</agent_response>
