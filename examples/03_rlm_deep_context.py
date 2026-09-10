"""
Example 03: RLM (Recursive Language Model) deep context processing.
Demonstrates extracting specific data from a large offloaded document without blowing up the context window.
"""

import os
from asera import RLMContext

# Sample massive context string offloaded from main prompt
LARGE_PAYLOAD = """
[LOG 2026-04-12 08:14:02] System initialized on cluster node-01.
[LOG 2026-04-12 09:30:11] Worker pool expanded to 64 instances.
[LOG 2026-04-12 11:45:00] Security audit passed with score 99.4.
[AUDIT NOTE 2026-05-01] Contract ID-9981 signed with vendor Acme Corp for $1,250,000.
[AUDIT NOTE 2026-05-15] Contract ID-9982 signed with vendor CyberShield for $420,000.
[AUDIT NOTE 2026-06-20] Payment authorized by CFO Elena Vance for invoice INV-40291.
[LOG 2026-07-01 12:00:00] End of Q2 performance review completed.
"""


def main():
    api_key = os.getenv("GEMINI_API_KEY", "your-api-key")

    rlm = RLMContext(
        api_key=api_key,
        model="gemini-2.0-flash",
        max_depth=1,
        max_iterations=6
    )

    query = "Find the total monetary value across all signed vendor contracts in the audit notes."
    print(f"Task: {query}\n")

    result = rlm.query(
        prompt=query,
        context=LARGE_PAYLOAD
    )

    print(f"RLM Extracted Answer:\n{result}")


if __name__ == "__main__":
    main()
