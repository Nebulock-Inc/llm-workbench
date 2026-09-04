Evaluate whether the response provides adequate explanation and reasoning derived from query results. 

This is NOT about factual accuracy (that's a separate eval). This checks:
- Does it explain WHY, not just WHAT?
- Does it cite specific sources (hunt reports, rules, findings)?
- Does it acknowledge limitations or gaps?
- Does the reader understand the reasoning?

Examples:
- Score 5: "Found 3 hunt reports. The 2026-08-18 VNC hunt found 48 listening endpoints but no compromises. The 2026-04-22 RAE hunt cleared all AppleEventsD activity. Neither addressed screensharingd process monitoring, leaving that gap unresolved."
- Score 3: "Found 3 hunt reports about macOS." (lacks detail and reasoning)
- Score 1: "Here are the results." (no explanation at all)

<agent_query_results>
Query Results: {query_results}
</agent_query_results>

<agent_response>
Response: {output} 
</agent_response>
