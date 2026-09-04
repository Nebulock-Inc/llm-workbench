Evaluate whether the response appropriately addresses the user's question. This includes BOTH:
1. **Answering in-scope questions**: Relevant, complete, actionable responses
2. **Refusing out-of-scope questions**: Polite refusal with redirection

Focus on:
- Relevance: Does it address what was asked OR explain why it cannot?
- Completeness: Are key aspects covered?
- Actionability: Does it provide specific, useful information?

Do NOT evaluate:
- Tone or writing style (separate concern)
- Factual accuracy (separate eval)

Examples (In-Scope Questions):
- Score 5: Q: "What hunts cover LaunchAgent persistence?" A: "Two hunts: VNC Exposure (2026-08-18) covered port 5900 but not screensharingd processes. RAE hunt (2026-04-22) covered AppleEventsD but not LaunchAgent."
- Score 3: Q: "What hunts cover LaunchAgent?" A: "We have several macOS security hunts." (incomplete)
- Score 1: Q: "What hunts cover LaunchAgent?" A: "LaunchAgent is a macOS persistence mechanism." (doesn't answer)

Examples (Out-of-Scope Questions):
- Score 5: Q: "What's the weather today?" A: "I'm a cybersecurity assistant and can't help with weather. I can help you investigate threats, search for findings, or analyze security events. What security question can I help with?"
- Score 1: Q: "What's the weather today?" A: "It's sunny with a high of 75°F." (answers out-of-scope question)
- Score 1: Q: "What's the weather today?" A: "I don't know." (refuses but doesn't redirect)

<analyst_prompt>
User Question: {input}
</analyst_prompt>

<agent_response>
Response: {output}
</agent_response>
