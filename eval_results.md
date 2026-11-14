# Canon eval

15 questions (6 code, 3 finance, 2 medical, 2 local, 2 general), then the same 15 reworded. Run 2026-11-14 14:16 PST.

| mode | accuracy | avg calls | avg tokens the agent must read | avg latency | memory hit rate |
|---|---|---|---|---|---|
| Web-only (tavily) | 93% (14/15) | 1.0 | 2,690 | 3.8s | — |
| Telem default (all 7) | 87% (13/15) | 7.0 | 19,556 | 4.2s | — |
| Canon, first pass | 100% (15/15) | 3.0 | 44 | 7.4s | 0% |
| Canon, second pass (reworded) | 100% (15/15) | 0.5 | 44 | 0.4s | 80% |

Tokens: web-only and Telem default = the result text the agent reads; Canon = the answer, quote, URL and date it returns (0 calls on a memory hit). Telem default accuracy uses its `answer` field (tavily is the only answer provider in the default set).

## Per question

| # | category | question | truth | web-only (tavily) | Canon pass 1 | Canon pass 2 (reworded) |
|---|---|---|---|---|---|---|
| 1 | code | Latest version of pytorch | 2.15.0 | ✅ The most recent release listed is PyTorch 2.15.0, announced on 28 Oct  | ✅ torch 2.15.0 (released 2026-10-28) | ✅ torch 2.15.0 (released 2026-10-28) *(memory)* |
| 2 | code | Latest version of next.js | 16.4.6 | ✅ The latest released version of Next.js is 16.4.6. | ✅ next 16.4.6 (released 2026-11-09) | ✅ next 16.4.6 (released 2026-11-09) *(memory)* |
| 3 | code | Latest pydantic version | 2.14.3 | ❌ The most recent release shown is v2.13.4 (2026‑05‑06). | ✅ pydantic 2.14.3 (released 2026-11-05) | ✅ pydantic 2.14.3 (released 2026-11-05) *(memory)* |
| 4 | code | Latest version of langchain | 1.5.0 | ✅ The most recent release of the Python langchain package is version 1.5 | ✅ langchain 1.5.0 (released 2026-11-11) | ✅ langchain 1.5.0 (released 2026-11-11) |
| 5 | code | Latest version of react | 19.3.1 | ✅ The React documentation lists the latest major release as 19.3, while  | ✅ react 19.3.1 (released 2026-10-14) | ✅ react 19.3.1 (released 2026-10-14) |
| 6 | code | Latest version of fastapi | 0.163.0 | ✅ According to the PyPI project page, the latest version is 0.163.0; the | ✅ fastapi 0.163.0 (released 2026-11-14) | ✅ fastapi 0.163.0 (released 2026-11-14) |
| 7 | finance | What was Tesla's revenue in Q3 2026? | 28[.,]0[45] / 28,?046 | ✅ Tesla reported revenue of roughly $28 billion in the third quarter of  | ✅ $28.05 billion | ✅ $28.05 billion *(memory)* |
| 8 | finance | What was Tesla's revenue in Q2 2026? | 28[.,]2[34] / 28,?236 | ✅ Tesla’s revenue in Q2 2026 was $28.24 billion. | ✅ $28.24 billion | ✅ $28.24 billion *(memory)* |
| 9 | finance | What was Tesla's revenue in Q1 2026? | 22[.,]3[89] / 22,?387 | ✅ Tesla reported Q1 2026 revenue of roughly $22.39 billion. | ✅ $22.387 billion (Total revenues, Q1 2026, $22,387 million) | ✅ $22.387 billion (Total revenues, Q1 2026, $22,387 million) *(memory)* |
| 10 | medical | Does semaglutide reduce cardiovascular events in obesity? | \byes\b / reduc / lower | ✅ Yes. In patients with overweight or obesity (BMI ≥ 27 kg/m²) and estab | ✅ Yes, HR 0.80 (6.5% vs 8.0%) for the primary MACE composite | ✅ Yes, HR 0.80 (6.5% vs 8.0%) for the primary MACE composite *(memory)* |
| 11 | medical | Does dapagliflozin reduce hospitalization for heart failure? | \byes\b / reduc / lower | ✅ Yes. Across multiple trials dapagliflozin consistently lowered the rat | ✅ Yes; composite of worsening heart failure (hospitalization or urgent I | ✅ Yes; composite of worsening heart failure (hospitalization or urgent I *(memory)* |
| 12 | local | What is the address of the Computer History Museum in Mountain View? | 1401 | ✅ The Computer History Museum is located at 1401 N. Shoreline Blvd., Mou | ✅ 1401 N Shoreline Blvd, Mountain View, CA 94043 | ✅ 1401 N Shoreline Blvd, Mountain View, CA 94043 *(memory)* |
| 13 | local | Where is the Exploratorium in San Francisco? | pier 15 / pier 17 | ✅ The Exploratorium is located at Pier 15 on San Francisco’s Embarcadero | ✅ Pier 15 Embarcadero at Green St, San Francisco, CA 94111 | ✅ Pier 15 Embarcadero at Green St, San Francisco, CA 94111 *(memory)* |
| 14 | general | When was the ISS launched? | 1998 | ✅ The International Space Station’s assembly began with the launch of th | ✅ November 20, 1998 | ✅ November 20, 1998 *(memory)* |
| 15 | general | Who wrote the novel Middlemarch? | george eliot / mary ann evans | ✅ Middlemarch was written by George Eliot (the pen name of Mary Ann Evan | ✅ George Eliot (Mary Ann Evans) | ✅ George Eliot (Mary Ann Evans) *(memory)* |
