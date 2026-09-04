Read any Clickhouse SQL queries in the output. If queries are present, determine if the query texts found correctly and accurately reflect the human prompt, and are correctly formatted for Clickhouse SQL. Count the number of queries in the text that are correct, reflect the user's intent, and are syntactically accurate, and the number which have flaws in any of these areas. 

Do not penalize redundancy or repetition. Queries should have organization scoping, whether the user has asked for it or not. If no Clickhouse SQL queries are found, or the query text is blank, do not penalize the result. 

*Examples:* Q: "Which entities saw signals from rule X in the last 30 days?" — 5: traces `select * from entities where entities.signals > 0` with date filter. 3: returns all signals from rule X without filtering by entity or date.


<analyst_prompt>
Prompt: {input}
</analyst_prompt>

<agent_queries>
Queries: {queries}
</agent_queries>
