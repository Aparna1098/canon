"""Drive the live demo page scene by scene and record it: demo/out/raw.webm + demo/out/timeline.json.

Needs the API running (.venv/bin/uvicorn api:app --port 8000) and demo/out/audio/manifest.json from tts.py.
Memory is reset first so the reworded Tesla question is a real memory hit, as in scripts/record_replay.py.
"""
import asyncio
import json
import os
import shutil
import sys
import time

import httpx
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
from scenes import SCENES  # noqa: E402

BASE = os.environ.get("CANON_URL", "http://localhost:8000")
W, H = 1920, 1080
PAD = 0.45  # silence after each clip

CURSOR_JS = """
window.addEventListener('DOMContentLoaded', () => {
  const st = document.createElement('style');
  st.textContent = `
    #__cur{position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;width:28px;height:28px;
           transform:translate(960px,540px);filter:drop-shadow(0 2px 4px rgba(0,0,0,.6))}
    .__rip{position:fixed;z-index:2147483646;pointer-events:none;width:44px;height:44px;margin:-22px 0 0 -22px;
           border-radius:50%;border:3px solid #34d399;animation:__rip .6s ease-out forwards}
    @keyframes __rip{from{transform:scale(.3);opacity:1}to{transform:scale(1.5);opacity:0}}
    .__hl{position:absolute;z-index:2147483645;pointer-events:none;border:3px solid #34d399;border-radius:16px;
          box-shadow:0 0 0 6px rgba(52,211,153,.18),0 0 40px rgba(52,211,153,.25);
          transition:opacity .4s;opacity:0}`;
  document.head.appendChild(st);
  const c = document.createElement('div'); c.id = '__cur';
  c.innerHTML = '<svg viewBox="0 0 24 24" width="28" height="28"><path d="M4 2l15 11.5-6.6 1.1 3.8 7.3-3 1.5-3.8-7.4L4 20.5z" fill="#fff" stroke="#111" stroke-width="1.3" stroke-linejoin="round"/></svg>';
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.transform = `translate(${e.clientX - 4}px,${e.clientY - 2}px)`; }, true);
  document.addEventListener('mousedown', e => {
    const r = document.createElement('div'); r.className = '__rip';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
    document.body.appendChild(r); setTimeout(() => r.remove(), 700);
  }, true);
  window.__hl = (sel, ms) => {
    const el = document.querySelector(sel); if (!el) return false;
    const b = el.getBoundingClientRect(), h = document.createElement('div'); h.className = '__hl';
    Object.assign(h.style, {left: (b.left + scrollX - 10) + 'px', top: (b.top + scrollY - 10) + 'px',
                            width: (b.width + 20) + 'px', height: (b.height + 20) + 'px'});
    document.body.appendChild(h); requestAnimationFrame(() => h.style.opacity = 1);
    setTimeout(() => { h.style.opacity = 0; setTimeout(() => h.remove(), 450); }, ms);
    return true;
  };
});
"""


class Director:
    def __init__(self, page, t0):
        self.page, self.mouse, self.t0 = page, page.mouse, t0
        self.x, self.y = W / 2, H / 2
        self.waits = []  # (start, end) of waits on the live API, so the build can cut the silent parts

    def now(self):
        return time.perf_counter() - self.t0

    async def wait(self, s):
        await asyncio.sleep(s)

    async def _center(self, sel):
        el = self.page.locator(sel).first
        await el.scroll_into_view_if_needed()
        b = await el.bounding_box()
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

    async def move_to(self, sel, dx=0, dy=0, steps=25):
        x, y = await self._center(sel)
        self.x, self.y = x + dx, y + dy
        await self.mouse.move(self.x, self.y, steps=steps)

    async def click(self, sel):
        await self.move_to(sel)
        await asyncio.sleep(0.25)
        await self.mouse.click(self.x, self.y)

    async def ask(self, q):
        await self.click("#q")
        await self.page.keyboard.press("Meta+A")
        await self.page.keyboard.press("Backspace")
        await self.page.keyboard.type(q, delay=45)
        await asyncio.sleep(0.4)
        await self.click("#ask-btn")

    async def _scroll(self, top):
        await self.page.evaluate("t => window.scrollTo({top: Math.max(0, t), behavior: 'smooth'})", top)
        await asyncio.sleep(1.1)

    async def scroll_top(self):
        await self._scroll(0)

    async def scroll_to(self, sel, offset=0):
        top = await self.page.evaluate(
            "s => document.querySelector(s).getBoundingClientRect().top + scrollY", sel)
        await self._scroll(top + offset)

    async def highlight(self, sel, seconds):
        if not await self.page.evaluate("([s, ms]) => window.__hl(s, ms)", [sel, int(seconds * 1000)]):
            print(f"  warn: nothing to highlight for {sel}")

    async def wait_canon(self, expect_memory=False, timeout=180):
        """Until Canon's card shows a verified answer (the web-only card may still be loading)."""
        start = self.now()
        await self.page.wait_for_function(
            """() => { const t = document.getElementById('result').innerText;
                       return /VERIFIED|FROM MEMORY|UNVERIFIED|LESS PRECISE/.test(t)
                              || !document.getElementById('error').classList.contains('hidden'); }""",
            timeout=timeout * 1000)
        self.waits.append((round(start, 3), round(self.now(), 3)))
        text = await self.page.inner_text("#result")
        err = await self.page.inner_text("#error")
        if err.strip() or "UNVERIFIED" in text:
            raise RuntimeError(f"Canon did not return a verified answer: {err or text[:200]}")
        if expect_memory and "FROM MEMORY" not in text:
            raise RuntimeError("expected a memory hit for the reworded question, got a fresh lookup")

    async def wait_answer(self, timeout=180):
        """Until the whole run (Canon and the web-only agent) is finished."""
        await self.wait_canon(timeout=timeout)
        start = self.now()
        await self.page.wait_for_function("() => !document.getElementById('ask-btn').disabled", timeout=timeout * 1000)
        self.waits.append((round(start, 3), round(self.now(), 3)))
        await asyncio.sleep(0.5)


async def main():
    clips = json.load(open(os.path.join(OUT, "audio", "manifest.json")))["clips"]
    httpx.post(f"{BASE}/memory/reset", timeout=10).raise_for_status()
    vid_dir = os.path.join(OUT, "video_tmp")
    shutil.rmtree(vid_dir, ignore_errors=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        ctx = await browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                        record_video_dir=vid_dir, record_video_size={"width": W, "height": H},
                                        color_scheme="dark")
        await ctx.add_init_script(CURSOR_JS)
        page = await ctx.new_page()
        t0 = time.perf_counter()
        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_function("() => document.getElementById('tiles').children.length > 0")
        await asyncio.sleep(1.0)  # let fonts settle
        d = Director(page, t0)
        await page.mouse.move(W / 2, H / 2)
        timeline = []
        for sid, _, _, action in SCENES:
            dur = clips[sid]["duration"]
            start = time.perf_counter() - t0
            print(f"scene {sid:16s} at {start:6.1f}s (voice {dur:4.1f}s)")
            await asyncio.gather(action(d), asyncio.sleep(dur + PAD))
            timeline.append({"id": sid, "start": round(start, 3), "voice_end": round(start + dur + PAD, 3),
                             "end": round(time.perf_counter() - t0, 3)})
        await asyncio.sleep(1.5)
        video = await page.video.path()
        await ctx.close()
        await browser.close()
    shutil.move(video, os.path.join(OUT, "raw.webm"))
    shutil.rmtree(vid_dir, ignore_errors=True)
    json.dump({"scenes": timeline, "waits": d.waits, "end": round(time.perf_counter() - t0, 3)},
              open(os.path.join(OUT, "timeline.json"), "w"), indent=1)
    print(f"record: {timeline[-1]['end']:.0f}s → demo/out/raw.webm")


if __name__ == "__main__":
    asyncio.run(main())
