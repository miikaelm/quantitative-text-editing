"""
Render HTML/CSS documents to images using a headless browser (Playwright).

Produces deterministic, pixel-perfect renders at a fixed resolution.
"""

import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image
from playwright.async_api import async_playwright, Browser


@dataclass
class RenderConfig:
    """Configuration for deterministic rendering."""
    width: int = 1024
    height: int = 1024
    device_scale_factor: float = 1.0  # keep at 1 to avoid DPI scaling artifacts
    # Downscale rendered images to this size (e.g. 512 for 512x512 training).
    # None means no downscaling.
    downscale_to: int | None = 512
    # Disable animations/transitions for determinism
    disable_animations: bool = True
    # Optional: force a specific font to avoid system font differences
    default_font: str | None = None


@dataclass
class RenderResult:
    """Result of rendering an HTML document."""
    image_path: Path
    width: int
    height: int
    html_path: Path | None = None
    errors: list[str] = field(default_factory=list)


class Renderer:
    """Headless browser renderer for HTML->image conversion."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        self._playwright = None
        self._browser: Browser | None = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            args=[
                "--disable-gpu",
                "--disable-lcd-text",        # disable subpixel rendering
                "--disable-font-subpixel-positioning",
                "--font-render-hinting=none", # consistent text rendering
            ]
        )
        return self

    async def __aexit__(self, *exc):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _new_page(self):
        """Create a new page with deterministic settings."""
        context = await self._browser.new_context(
            viewport={
                "width": self.config.width,
                "height": self.config.height,
            },
            device_scale_factor=self.config.device_scale_factor,
            # Disable animations globally
            reduced_motion="reduce" if self.config.disable_animations else "no-preference",
        )
        page = await context.new_page()

        if self.config.disable_animations:
            await page.add_style_tag(content="""
                *, *::before, *::after {
                    animation-duration: 0s !important;
                    animation-delay: 0s !important;
                    transition-duration: 0s !important;
                    transition-delay: 0s !important;
                }
            """)

        if self.config.default_font:
            await page.add_style_tag(content=f"""
                * {{ font-family: '{self.config.default_font}', sans-serif !important; }}
            """)

        # Collect console errors
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        return page, context, errors

    def _downscale(self, image_path: Path) -> None:
        """Downscale an image in-place using Lanczos resampling."""
        size = self.config.downscale_to
        if size is None:
            return
        img = Image.open(image_path)
        if img.size == (size, size):
            return
        img = img.resize((size, size), Image.LANCZOS)
        img.save(image_path)

    async def render_html_string(
        self, html: str, output_path: Path, full_page: bool = False
    ) -> RenderResult:
        """Render an HTML string to a PNG image."""
        page, context, errors = await self._new_page()
        try:
            await page.set_content(html, wait_until="networkidle")
            # Wait for fonts to load
            await page.evaluate("() => document.fonts.ready")

            await page.screenshot(
                path=str(output_path),
                full_page=full_page,
                type="png",
            )

            self._downscale(output_path)

            final_size = self.config.downscale_to or self.config.width
            return RenderResult(
                image_path=output_path,
                width=final_size,
                height=final_size,
                errors=errors,
            )
        finally:
            await context.close()

    async def render_html_file(
        self, html_path: Path, output_path: Path, full_page: bool = False
    ) -> RenderResult:
        """Render an HTML file to a PNG image."""
        html = html_path.read_text(encoding="utf-8")
        result = await self.render_html_string(html, output_path, full_page)
        result.html_path = html_path
        return result

    async def render_pair(
        self,
        source_html: str,
        target_html: str,
        output_dir: Path,
        pair_id: str,
    ) -> tuple[RenderResult, RenderResult]:
        """Render a source/target HTML pair — the core operation for the pipeline."""
        output_dir.mkdir(parents=True, exist_ok=True)

        source_result = await self.render_html_string(
            source_html, output_dir / f"{pair_id}_source.png"
        )
        target_result = await self.render_html_string(
            target_html, output_dir / f"{pair_id}_target.png"
        )
        return source_result, target_result


def render_pair_sync(
    source_html: str,
    target_html: str,
    output_dir: str | Path,
    pair_id: str,
    config: RenderConfig | None = None,
) -> tuple[RenderResult, RenderResult]:
    """Synchronous wrapper for render_pair."""

    async def _run():
        async with Renderer(config) as renderer:
            return await renderer.render_pair(
                source_html, target_html, Path(output_dir), pair_id
            )

    return asyncio.run(_run())


if __name__ == "__main__":
    source = """
    <!DOCTYPE html>
    <html><body style="margin:0; background:#f5f5f5; display:flex;
        justify-content:center; align-items:center; height:100vh;">
        <h1 style="font-size:72px; color:#333333; font-family:Arial,sans-serif;">
            MINIMALIST
        </h1>
    </body></html>
    """

    target = """
    <!DOCTYPE html>
    <html><body style="margin:0; background:#f5f5f5; display:flex;
        justify-content:center; align-items:center; height:100vh;">
        <h1 style="font-size:72px; color:#3B82F6; font-family:Arial,sans-serif;">
            MINIMALIST
        </h1>
    </body></html>
    """

    src_result, tgt_result = render_pair_sync(
        source, target, output_dir="./test_renders", pair_id="color_test_001"
    )
    print(f"Source: {src_result.image_path} ({src_result.width}x{src_result.height})")
    print(f"Target: {tgt_result.image_path} ({tgt_result.width}x{tgt_result.height})")
    if src_result.errors or tgt_result.errors:
        print(f"Errors: {src_result.errors + tgt_result.errors}")