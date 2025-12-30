"""Web page fetching and processing utilities."""

import base64
import io
import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import aiohttp
import markdownify
from PIL import Image

logger = logging.getLogger(__name__)

# URL regex pattern - matches http/https URLs
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"\'\)\]]+",
    re.IGNORECASE,
)


@dataclass
class ImageData:
    """Image extracted from webpage."""

    src: str  # Original URL
    alt: str
    base64_data: str  # Base64-encoded image data
    mime_type: str  # e.g., "image/png"
    width: int
    height: int


@dataclass
class WebPageResult:
    """Result of fetching a webpage."""

    url: str
    title: str
    markdown_content: str
    images: list[ImageData] = field(default_factory=list)
    fetch_time_ms: float = 0.0
    truncated: bool = False
    error: str | None = None


class ImageExtractingConverter(markdownify.MarkdownConverter):
    """Custom markdown converter that extracts image references with context."""

    def __init__(self, image_refs: list[dict], **kwargs):
        """Initialize with a list to collect image references.

        Args:
            image_refs: List to append image info dicts to
            **kwargs: Additional args for MarkdownConverter
        """
        super().__init__(**kwargs)
        self._image_refs = image_refs

    def convert_img(self, el, text, parent_tags):
        """Extract image with context and return placeholder."""
        src = el.get("src", "")
        alt = el.get("alt", "")

        if src:
            # Collect context about the image's location
            ref = {
                "src": src,
                "alt": alt,
                "parent_tags": set(parent_tags),  # Tags this image is nested within
                "in_figure": False,
                "in_header": False,
                "in_link_to_home": False,
                "aria_hidden": el.get("aria-hidden") == "true",
                "role_presentation": el.get("role") == "presentation",
            }

            # Check immediate ancestors for semantic context
            parent = el.parent
            ancestors_checked = 0
            while parent and ancestors_checked < 5:
                parent_name = parent.name if parent.name else ""
                parent_class = " ".join(parent.get("class", []))
                parent_id = parent.get("id", "")

                if parent_name == "figure":
                    ref["in_figure"] = True
                if parent_name in ("header", "nav", "footer", "aside"):
                    ref["in_header"] = True
                if parent_name == "a":
                    href = parent.get("href", "")
                    if href in ("/", "#", "") or href.endswith("/index.html"):
                        ref["in_link_to_home"] = True

                # Check for junk classes/ids
                combined = f"{parent_class} {parent_id}".lower()
                junk_patterns = (
                    "logo", "banner", "nav", "header", "footer",
                    "icon", "avatar", "badge",
                )
                if any(x in combined for x in junk_patterns):
                    ref["in_header"] = True

                parent = parent.parent
                ancestors_checked += 1

            self._image_refs.append(ref)

            # Return a placeholder that will be meaningful in context
            alt_text = alt if alt else "image"
            return f"[Image: {alt_text}]"

        return ""


class WebFetcher:
    """Fetches and processes web pages."""

    MAX_CONTENT_CHARS = 50_000  # Truncation limit for markdown
    MAX_IMAGES = 5  # Maximum images to include
    MAX_IMAGE_DIMENSION = 1024  # Resize images to fit within this
    MIN_IMAGE_DIMENSION = 80  # Skip tiny images (likely icons)
    TIMEOUT_SECONDS = 10
    MAX_REDIRECTS = 5

    # Common user agent to avoid blocks
    USER_AGENT = (
        "Mozilla/5.0 (compatible; BickerBot/1.0; +https://github.com/your-repo)"
    )

    # Patterns in src/alt that suggest junk images
    JUNK_PATTERNS = (
        "logo", "banner", "icon", "avatar", "badge", "button",
        "sprite", "spacer", "pixel", "tracking", "analytics",
        "ads", "advertisement", "loading", "spinner",
    )

    def __init__(self):
        """Initialize the web fetcher."""
        self._session: aiohttp.ClientSession | None = None

    def _score_image_ref(self, ref: dict) -> int:
        """Score an image reference for prioritization.

        Positive scores = good content images
        Negative scores = junk/decorative images (will be filtered out)
        Zero = neutral

        Args:
            ref: Image reference dict with context

        Returns:
            Integer score
        """
        score = 0
        src_lower = ref["src"].lower()
        alt_lower = ref.get("alt", "").lower()

        # === Negative signals (junk) ===

        # In header/nav/footer/aside - almost certainly decorative
        if ref.get("in_header"):
            score -= 50

        # Logo link to homepage
        if ref.get("in_link_to_home"):
            score -= 50

        # Explicitly decorative
        if ref.get("aria_hidden") or ref.get("role_presentation"):
            score -= 30

        # Junk patterns in src or alt
        for pattern in self.JUNK_PATTERNS:
            if pattern in src_lower or pattern in alt_lower:
                score -= 20
                break

        # Data URIs for tiny inline images (often math fallbacks, emojis)
        if ref["src"].startswith("data:") and len(ref["src"]) < 500:
            score -= 10

        # === Positive signals (content) ===

        # Inside <figure> - intentional content image
        if ref.get("in_figure"):
            score += 30

        # Inside article/main (check parent_tags)
        parent_tags = ref.get("parent_tags", set())
        if "article" in parent_tags or "main" in parent_tags:
            score += 20

        # Has descriptive alt text (more than just "image")
        alt = ref.get("alt", "")
        if len(alt) > 15:
            score += 15
        elif len(alt) > 5:
            score += 5

        # Path suggests content (common patterns for blog/article images)
        content_paths = (
            "/post/", "/article/", "/content/",
            "/assets/", "/images/", "/figures/",
        )
        if any(x in src_lower for x in content_paths):
            score += 10

        return score

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.TIMEOUT_SECONDS)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.USER_AGENT},
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch(self, url: str, include_images: bool = True) -> WebPageResult:
        """Fetch and process a webpage.

        Args:
            url: The URL to fetch
            include_images: Whether to fetch and include images

        Returns:
            WebPageResult with processed content
        """
        start_time = time.monotonic()

        # Validate URL
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return WebPageResult(
                url=url,
                title="",
                markdown_content="",
                error=f"Invalid URL scheme: {parsed.scheme}. Only http/https supported.",
            )

        try:
            session = await self._get_session()

            async with session.get(
                url,
                allow_redirects=True,
                max_redirects=self.MAX_REDIRECTS,
            ) as response:
                if response.status != 200:
                    return WebPageResult(
                        url=url,
                        title="",
                        markdown_content="",
                        error=f"HTTP {response.status}: {response.reason}",
                        fetch_time_ms=(time.monotonic() - start_time) * 1000,
                    )

                content_type = response.headers.get("Content-Type", "")

                # Handle direct image URLs
                if content_type.startswith("image/"):
                    image_data = await response.read()
                    image_result = self._process_image(image_data, url, "")
                    if image_result:
                        return WebPageResult(
                            url=url,
                            title="",
                            markdown_content="[Direct image link]",
                            images=[image_result],
                            fetch_time_ms=(time.monotonic() - start_time) * 1000,
                        )
                    else:
                        return WebPageResult(
                            url=url,
                            title="",
                            markdown_content="",
                            error="Failed to process image",
                            fetch_time_ms=(time.monotonic() - start_time) * 1000,
                        )

                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    return WebPageResult(
                        url=url,
                        title="",
                        markdown_content="",
                        error=f"Not an HTML page: {content_type}",
                        fetch_time_ms=(time.monotonic() - start_time) * 1000,
                    )

                html = await response.text()

        except aiohttp.ClientError as e:
            return WebPageResult(
                url=url,
                title="",
                markdown_content="",
                error=f"Failed to fetch: {e}",
                fetch_time_ms=(time.monotonic() - start_time) * 1000,
            )
        except TimeoutError:
            return WebPageResult(
                url=url,
                title="",
                markdown_content="",
                error=f"Timeout after {self.TIMEOUT_SECONDS}s",
                fetch_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Convert HTML to markdown and extract image refs
        markdown, image_refs, title = self._html_to_markdown(html, url)

        # Truncate if needed
        truncated = False
        if len(markdown) > self.MAX_CONTENT_CHARS:
            markdown = self._smart_truncate(markdown, self.MAX_CONTENT_CHARS)
            truncated = True

        # Fetch images if requested
        images: list[ImageData] = []
        if include_images and image_refs:
            # Score and filter images
            scored_refs = [(ref, self._score_image_ref(ref)) for ref in image_refs]

            # Filter out negative scores and sort by score descending
            positive_refs = [(ref, score) for ref, score in scored_refs if score >= 0]
            positive_refs.sort(key=lambda x: x[1], reverse=True)

            # Take top N
            top_refs = [ref for ref, score in positive_refs[:self.MAX_IMAGES]]

            logger.debug(
                f"IMAGE_SCORING: {len(image_refs)} total, "
                f"{len(positive_refs)} positive, taking {len(top_refs)}"
            )

            images = await self._fetch_images(top_refs, url)

        fetch_time_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            f"WEB_FETCH: url={url} chars={len(markdown)} images={len(images)} "
            f"truncated={truncated} time_ms={fetch_time_ms:.1f}"
        )

        return WebPageResult(
            url=url,
            title=title,
            markdown_content=markdown,
            images=images,
            fetch_time_ms=fetch_time_ms,
            truncated=truncated,
        )

    def _html_to_markdown(
        self, html: str, base_url: str
    ) -> tuple[str, list[dict], str]:
        """Convert HTML to markdown and extract image references.

        Args:
            html: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            Tuple of (markdown_text, image_refs, title)
        """
        from bs4 import BeautifulSoup

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove script, style, and other non-content elements
        for element in soup.find_all(["script", "style", "noscript", "iframe", "nav", "footer"]):
            element.decompose()

        # Try to find main content
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.find("body")
            or soup
        )

        # Convert to markdown with image extraction
        image_refs: list[dict] = []
        converter = ImageExtractingConverter(
            image_refs,
            heading_style="ATX",
            bullets="-",
            strip=["a"],  # Remove links to keep text cleaner
        )

        markdown = converter.convert(str(main_content))

        # Clean up excessive whitespace
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        markdown = markdown.strip()

        # Resolve relative image URLs
        for ref in image_refs:
            ref["src"] = urljoin(base_url, ref["src"])

        return markdown, image_refs, title

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """Truncate text at paragraph boundary if possible.

        Args:
            text: Text to truncate
            max_chars: Maximum characters

        Returns:
            Truncated text with indicator
        """
        if len(text) <= max_chars:
            return text

        # Find last paragraph break before limit
        truncate_at = text.rfind("\n\n", 0, max_chars - 50)

        if truncate_at < max_chars // 2:
            # No good break point, just cut
            truncate_at = max_chars - 50

        return text[:truncate_at].strip() + "\n\n[Content truncated...]"

    async def _fetch_images(
        self, image_refs: list[dict], base_url: str
    ) -> list[ImageData]:
        """Fetch and process images.

        Args:
            image_refs: List of image reference dicts with 'src' and 'alt'
            base_url: Base URL for resolving relative paths

        Returns:
            List of processed ImageData objects
        """
        images: list[ImageData] = []
        session = await self._get_session()

        for ref in image_refs:
            try:
                image_data = await self._fetch_single_image(session, ref, base_url)
                if image_data:
                    images.append(image_data)
            except Exception as e:
                logger.debug(f"Failed to fetch image {ref['src']}: {e}")
                continue

        return images

    async def _fetch_single_image(
        self,
        session: aiohttp.ClientSession,
        ref: dict,
        base_url: str,
    ) -> ImageData | None:
        """Fetch and process a single image.

        Args:
            session: aiohttp session
            ref: Image reference dict
            base_url: Base URL for resolving relative paths

        Returns:
            ImageData or None if failed
        """
        src = ref["src"]
        alt = ref.get("alt", "")

        # Handle data URIs
        if src.startswith("data:"):
            return self._parse_data_uri(src, alt)

        # Resolve relative URL
        url = urljoin(base_url, src)

        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None

                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    return None

                data = await response.read()

        except Exception:
            return None

        # Process and possibly resize the image
        return self._process_image(data, url, alt)

    def _parse_data_uri(self, data_uri: str, alt: str) -> ImageData | None:
        """Parse a data URI and return ImageData.

        Args:
            data_uri: The data URI string
            alt: Alt text for the image

        Returns:
            ImageData or None if parsing fails
        """
        try:
            # Format: data:image/png;base64,<data>
            _header, b64_data = data_uri.split(",", 1)

            # Decode and process
            raw_data = base64.b64decode(b64_data)
            return self._process_image(raw_data, data_uri[:50], alt)

        except Exception:
            return None

    def _process_image(
        self, data: bytes, src: str, alt: str
    ) -> ImageData | None:
        """Process image data: resize if needed, convert to base64.

        Args:
            data: Raw image bytes
            src: Source URL/identifier
            alt: Alt text

        Returns:
            ImageData or None if processing fails or image is too small
        """
        try:
            img = Image.open(io.BytesIO(data))

            # Convert to RGB if needed (handles RGBA, P mode, etc.)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            width, height = img.size

            # Skip tiny images (icons, badges, spacers)
            if width < self.MIN_IMAGE_DIMENSION and height < self.MIN_IMAGE_DIMENSION:
                logger.debug(f"Skipping tiny image {width}x{height}: {src[:50]}")
                return None

            # Resize if larger than max dimension
            if width > self.MAX_IMAGE_DIMENSION or height > self.MAX_IMAGE_DIMENSION:
                img = self._resize_image(img, self.MAX_IMAGE_DIMENSION)
                width, height = img.size

            # Encode to JPEG for consistency and size
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return ImageData(
                src=src,
                alt=alt,
                base64_data=b64_data,
                mime_type="image/jpeg",
                width=width,
                height=height,
            )

        except Exception as e:
            logger.debug(f"Failed to process image: {e}")
            return None

    def _resize_image(self, img: Image.Image, max_dim: int) -> Image.Image:
        """Resize image to fit within max_dim while preserving aspect ratio.

        Args:
            img: PIL Image
            max_dim: Maximum dimension (width or height)

        Returns:
            Resized PIL Image
        """
        width, height = img.size

        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def extract_urls_from_text(self, text: str) -> list[str]:
        """Extract URLs from text.

        Args:
            text: Text to search for URLs

        Returns:
            List of found URLs (deduplicated, order preserved)
        """
        matches = URL_PATTERN.findall(text)

        # Deduplicate while preserving order
        seen = set()
        urls = []
        for url in matches:
            # Clean up trailing punctuation that might have been captured
            url = url.rstrip(".,;:!?)")

            if url not in seen:
                seen.add(url)
                urls.append(url)

        return urls
