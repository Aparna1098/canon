"""The demo script: one entry per scene, narration plus what the browser does while it plays.

Each scene starts its voice clip and its actions at the same moment. The scene ends when both are finished, so a slow
live answer just holds the frame (with the progress dots moving) instead of drifting out of sync.
Narration avoids live values (versions, revenue) that can change between runs; the page shows them.
"""


async def intro(d):
    await d.scroll_top()
    await d.move_to("h1", dx=-260, steps=40)
    await d.wait(2.5)
    await d.move_to("#tiles", steps=35)


async def hero(d):
    await d.highlight("h1", 3.5)
    await d.wait(4)
    await d.highlight("#tiles", 4)


async def pydantic_ask(d):
    await d.scroll_to("#ask", offset=-40)
    await d.click("#compare")
    await d.wait(0.6)
    await d.ask("Latest pydantic version")
    await d.wait_answer()


async def pydantic_result(d):
    await d.scroll_to("#run", offset=-20)
    await d.highlight("#result article:nth-child(1)", 4)
    await d.wait(4.2)
    await d.highlight("#result article:nth-child(2)", 6)


async def tesla_ask(d):
    await d.scroll_to("#ask", offset=-40)
    await d.ask("What was Tesla's revenue in Q3 2026?")
    await d.wait_answer()


async def tesla_result(d):
    await d.scroll_to("#run", offset=-20)
    await d.highlight("#result article:nth-child(2) pre", 6)


async def memory_ask(d):
    await d.scroll_to("#ask", offset=-40)
    await d.ask("How much revenue did Tesla report for Q3 2026?")
    await d.wait_canon(expect_memory=True)


async def memory_result(d):
    await d.scroll_to("#run", offset=-20)
    await d.highlight("#result article:nth-child(2) span.rounded-full", 5)
    await d.wait_answer()


async def memory_table(d):
    await d.scroll_to("#memory", offset=-20)
    await d.highlight("#facts", 6)


async def langchain(d):
    await d.scroll_to("#ask", offset=-40)
    await d.ask("What's new in the latest langchain release?")
    await d.wait_answer()
    await d.scroll_to("#run", offset=-20)


async def how(d):
    await d.scroll_to("#how", offset=-90)
    await d.highlight("#how", 6)
    await d.wait(6)
    await d.highlight("#sources", 6)


async def evaluation(d):
    await d.scroll_to("#eval", offset=-160)
    await d.highlight("#eval tr:nth-child(2)", 4)
    await d.wait(4.5)
    await d.highlight("#eval tr:nth-child(3)", 5)


async def outro(d):
    await d.scroll_top()
    await d.move_to("h1", steps=40)


# (id, chapter title for YouTube or None to continue the previous chapter, narration, action)
SCENES = [
    ("intro", "The problem",
     "Every AI agent that needs a fact searches the web, reads thousands of tokens, and hopes the page isn't "
     "stale. Then the next agent does it all over again.", intro),
    ("hero", "What Canon is",
     "This is Canon. Telem finds the sources. Canon keeps the facts. Canon answers from the official source, with an "
     "exact quote checked against the document, then remembers the fact, so the next agent pays zero searches for it.",
     hero),
    ("pydantic_ask", "Live demo: code",
     "Let's compare it with a plain web-only agent, and ask for the latest version of pydantic. Canon sees a code "
     "question, so it skips the open web and goes straight to the PyPI registry.", pydantic_ask),
    ("pydantic_result", None,
     "The web-only agent answers from whatever page it found, and pages lag the registry. Canon's answer comes from "
     "PyPI itself: the version, the release date, the exact field it read, and only a few dozen tokens.",
     pydantic_result),
    ("tesla_ask", "Live demo: finance",
     "Now finance: Tesla's revenue for Q3 2026. Canon routes this to SEC EDGAR through Telem, fetches the actual "
     "filing, and has Claude Haiku extract the number with a supporting quote.", tesla_ask),
    ("tesla_result", None,
     "The quote is verified. Canon checks that it really appears in the filing, so the answer can't be "
     "hallucinated. Only verified facts are saved to memory, each with an expiry.", tesla_result),
    ("memory_ask", "Shared memory",
     "Now a second agent asks the same thing, in different words.", memory_ask),
    ("memory_result", None,
     "Straight from memory. Zero calls, in milliseconds, while the web-only agent is still searching. Canon matches "
     "the meaning, the company and the quarter, so a different quarter would never get this answer.", memory_result),
    ("memory_table", None,
     "The memory panel shows every stored fact, its official source, when it expires, and how often it's been "
     "reused.", memory_table),
    ("langchain", "Live demo: release notes",
     "It handles more than numbers. For what's new in the latest langchain release, Canon reads the GitHub release "
     "notes for that exact version.", langchain),
    ("how", "How it works",
     "Five steps: check memory, pick the official source, fetch it through Telem with one targeted provider instead "
     "of all seven, verify the quote, and remember it. Code goes to PyPI and npm, finance to SEC EDGAR, medical to "
     "PubMed, and local questions to Google Maps.", how),
    ("evaluation", "Results",
     "On fifteen questions with known answers, Canon was right on all fifteen, and agents read about forty-four "
     "tokens instead of nearly twenty thousand. Reworded repeats came back from memory eighty percent of the time.",
     evaluation),
    ("outro", None,
     "Canon. Telem finds the sources. Canon keeps the facts.", outro),
]
