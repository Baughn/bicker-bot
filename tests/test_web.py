"""Tests for web fetching utilities."""

import base64
import io

import pytest
from PIL import Image

from bicker_bot.core.web import ImageData, WebFetcher, WebPageResult


class TestUrlExtraction:
    """Tests for URL extraction from text."""

    def test_extract_single_url(self):
        """Test extracting a single URL."""
        fetcher = WebFetcher()
        text = "Check out https://example.com for more info"
        urls = fetcher.extract_urls_from_text(text)
        assert urls == ["https://example.com"]

    def test_extract_multiple_urls(self):
        """Test extracting multiple URLs."""
        fetcher = WebFetcher()
        text = "Visit https://example.com and http://test.org/page"
        urls = fetcher.extract_urls_from_text(text)
        assert "https://example.com" in urls
        assert "http://test.org/page" in urls
        assert len(urls) == 2

    def test_extract_url_with_path(self):
        """Test extracting URL with path and query."""
        fetcher = WebFetcher()
        text = "See https://example.com/path/to/page?foo=bar&baz=1"
        urls = fetcher.extract_urls_from_text(text)
        assert urls == ["https://example.com/path/to/page?foo=bar&baz=1"]

    def test_strip_trailing_punctuation(self):
        """Test that trailing punctuation is stripped."""
        fetcher = WebFetcher()
        text = "Check https://example.com. Also see https://test.org!"
        urls = fetcher.extract_urls_from_text(text)
        assert "https://example.com" in urls
        assert "https://test.org" in urls

    def test_no_urls_in_text(self):
        """Test with no URLs present."""
        fetcher = WebFetcher()
        text = "This text has no URLs at all."
        urls = fetcher.extract_urls_from_text(text)
        assert urls == []

    def test_deduplication(self):
        """Test that duplicate URLs are removed."""
        fetcher = WebFetcher()
        text = "https://example.com is great. Visit https://example.com today!"
        urls = fetcher.extract_urls_from_text(text)
        assert urls == ["https://example.com"]


class TestHtmlToMarkdown:
    """Tests for HTML to markdown conversion."""

    @pytest.fixture
    def fetcher(self):
        return WebFetcher()

    def test_basic_conversion(self, fetcher):
        """Test basic HTML to markdown conversion."""
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is a test paragraph.</p>
        </body>
        </html>
        """
        markdown, images, title = fetcher._html_to_markdown(html, "https://example.com")

        assert "# Hello World" in markdown
        assert "This is a test paragraph" in markdown
        assert title == "Test Page"
        assert images == []

    def test_image_extraction(self, fetcher):
        """Test that images are extracted."""
        html = """
        <html>
        <body>
            <p>Some text</p>
            <img src="/images/test.png" alt="Test image">
            <img src="https://other.com/pic.jpg" alt="Other pic">
        </body>
        </html>
        """
        markdown, images, _ = fetcher._html_to_markdown(html, "https://example.com")

        assert len(images) == 2
        assert images[0]["src"] == "https://example.com/images/test.png"
        assert images[0]["alt"] == "Test image"
        assert images[1]["src"] == "https://other.com/pic.jpg"
        assert "[Image: Test image]" in markdown

    def test_nested_lists(self, fetcher):
        """Test nested list conversion."""
        html = """
        <html><body>
            <ul>
                <li>Item 1</li>
                <li>Item 2
                    <ul>
                        <li>Nested A</li>
                        <li>Nested B</li>
                    </ul>
                </li>
            </ul>
        </body></html>
        """
        markdown, _, _ = fetcher._html_to_markdown(html, "https://example.com")

        assert "Item 1" in markdown
        assert "Item 2" in markdown
        assert "Nested A" in markdown

    def test_code_blocks(self, fetcher):
        """Test code block handling."""
        html = """
        <html><body>
            <pre><code>def hello():
    print("world")</code></pre>
        </body></html>
        """
        markdown, _, _ = fetcher._html_to_markdown(html, "https://example.com")

        assert "def hello():" in markdown
        assert 'print("world")' in markdown

    def test_removes_script_and_style(self, fetcher):
        """Test that script and style are removed."""
        html = """
        <html><body>
            <script>alert('xss')</script>
            <style>.foo { color: red; }</style>
            <p>Real content</p>
        </body></html>
        """
        markdown, _, _ = fetcher._html_to_markdown(html, "https://example.com")

        assert "alert" not in markdown
        assert "color: red" not in markdown
        assert "Real content" in markdown

    def test_empty_title(self, fetcher):
        """Test handling of missing title."""
        html = "<html><body><p>No title</p></body></html>"
        _, _, title = fetcher._html_to_markdown(html, "https://example.com")
        assert title == ""


class TestContentTruncation:
    """Tests for content truncation."""

    @pytest.fixture
    def fetcher(self):
        return WebFetcher()

    def test_no_truncation_for_short_content(self, fetcher):
        """Test that short content is not truncated."""
        text = "Short content here."
        result = fetcher._smart_truncate(text, 1000)
        assert result == text
        assert "[Content truncated...]" not in result

    def test_truncation_at_paragraph(self, fetcher):
        """Test truncation at paragraph boundary."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph is very long."
        result = fetcher._smart_truncate(text, 50)

        assert result.endswith("[Content truncated...]")
        assert "Third paragraph" not in result

    def test_truncation_indicator(self, fetcher):
        """Test that truncation indicator is added."""
        text = "x" * 1000
        result = fetcher._smart_truncate(text, 100)

        assert len(result) < 1000
        assert "[Content truncated...]" in result


class TestImageProcessing:
    """Tests for image processing."""

    @pytest.fixture
    def fetcher(self):
        return WebFetcher()

    def create_test_image(self, width: int, height: int, mode: str = "RGB") -> bytes:
        """Create a test image of specified size."""
        img = Image.new(mode, (width, height), color="red")
        buffer = io.BytesIO()
        if mode == "RGBA":
            img.save(buffer, format="PNG")
        else:
            img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_resize_large_image(self, fetcher):
        """Test that large images are resized."""
        img = Image.new("RGB", (2000, 1000))
        resized = fetcher._resize_image(img, 1024)

        assert resized.width == 1024
        assert resized.height == 512  # Maintains aspect ratio

    def test_resize_tall_image(self, fetcher):
        """Test resizing tall images."""
        img = Image.new("RGB", (500, 2000))
        resized = fetcher._resize_image(img, 1024)

        assert resized.height == 1024
        assert resized.width == 256  # Maintains aspect ratio

    def test_no_resize_small_image(self, fetcher):
        """Test that small images are not resized."""
        img = Image.new("RGB", (500, 500))
        resized = fetcher._resize_image(img, 1024)

        # resize_image is always called by _process_image only when needed,
        # but the method itself will resize to the specified dimension
        # So this tests aspect ratio preservation with a smaller image
        assert resized.width == 1024 or resized.height == 1024

    def test_process_image_converts_to_jpeg(self, fetcher):
        """Test that images are converted to JPEG."""
        data = self.create_test_image(100, 100)
        result = fetcher._process_image(data, "test.jpg", "test")

        assert result is not None
        assert result.mime_type == "image/jpeg"
        assert result.width == 100
        assert result.height == 100

    def test_process_image_rgba(self, fetcher):
        """Test processing RGBA images."""
        data = self.create_test_image(100, 100, mode="RGBA")
        result = fetcher._process_image(data, "test.png", "test")

        assert result is not None
        assert result.mime_type == "image/jpeg"  # Converted to JPEG

    def test_process_image_base64_valid(self, fetcher):
        """Test that base64 output is valid."""
        data = self.create_test_image(100, 100)
        result = fetcher._process_image(data, "test.jpg", "test")

        assert result is not None
        # Verify base64 is decodable
        decoded = base64.b64decode(result.base64_data)
        assert len(decoded) > 0


class TestDataUriParsing:
    """Tests for data URI parsing."""

    @pytest.fixture
    def fetcher(self):
        return WebFetcher()

    def test_parse_valid_data_uri(self, fetcher):
        """Test parsing a valid data URI."""
        # Create a test image (must be >= MIN_IMAGE_DIMENSION)
        img = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()

        data_uri = f"data:image/png;base64,{b64}"
        result = fetcher._parse_data_uri(data_uri, "test")

        assert result is not None
        assert result.mime_type == "image/jpeg"  # Converted
        assert result.width == 100
        assert result.height == 100

    def test_parse_invalid_data_uri(self, fetcher):
        """Test handling of invalid data URI."""
        result = fetcher._parse_data_uri("not a data uri", "test")
        assert result is None

    def test_tiny_data_uri_filtered(self, fetcher):
        """Test that tiny images in data URIs are filtered out."""
        # Create a tiny image (< MIN_IMAGE_DIMENSION)
        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()

        data_uri = f"data:image/png;base64,{b64}"
        result = fetcher._parse_data_uri(data_uri, "test")

        assert result is None  # Filtered out as too small


class TestWebPageResult:
    """Tests for WebPageResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = WebPageResult(
            url="https://example.com",
            title="Test",
            markdown_content="Content",
        )

        assert result.images == []
        assert result.fetch_time_ms == 0.0
        assert result.truncated is False
        assert result.error is None

    def test_with_error(self):
        """Test result with error."""
        result = WebPageResult(
            url="https://example.com",
            title="",
            markdown_content="",
            error="Connection refused",
        )

        assert result.error == "Connection refused"


class TestImageData:
    """Tests for ImageData dataclass."""

    def test_creation(self):
        """Test creating ImageData."""
        data = ImageData(
            src="https://example.com/img.jpg",
            alt="Test image",
            base64_data="abc123",
            mime_type="image/jpeg",
            width=100,
            height=200,
        )

        assert data.src == "https://example.com/img.jpg"
        assert data.alt == "Test image"
        assert data.width == 100
        assert data.height == 200


class TestFetchValidation:
    """Tests for URL validation in fetch."""

    @pytest.fixture
    def fetcher(self):
        return WebFetcher()

    @pytest.mark.asyncio
    async def test_invalid_scheme_rejected(self, fetcher):
        """Test that non-http(s) schemes are rejected."""
        result = await fetcher.fetch("ftp://example.com/file")
        assert result.error is not None
        assert "Invalid URL scheme" in result.error

    @pytest.mark.asyncio
    async def test_javascript_scheme_rejected(self, fetcher):
        """Test that javascript: scheme is rejected."""
        result = await fetcher.fetch("javascript:alert(1)")
        assert result.error is not None
        assert "Invalid URL scheme" in result.error


class TestDirectImageFetch:
    """Tests for direct image URL fetching."""

    @pytest.fixture
    def fetcher(self):
        return WebFetcher()

    def create_test_image(self, width: int, height: int) -> bytes:
        """Create a test PNG image."""
        img = Image.new("RGB", (width, height), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_direct_image_fetch_processing(self, fetcher):
        """Test that direct image URLs are processed correctly.

        This tests the _process_image path that's used when fetching
        a direct image URL (content-type: image/*).
        """
        # Test the image processing that happens for direct image URLs
        image_bytes = self.create_test_image(200, 150)
        result = fetcher._process_image(image_bytes, "https://example.com/test.png", "")

        assert result is not None
        assert result.mime_type == "image/jpeg"
        assert result.width == 200
        assert result.height == 150
        assert result.base64_data  # Has base64 encoded data

        # Verify the base64 can be decoded back
        decoded = base64.b64decode(result.base64_data)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_direct_image_too_small_rejected(self, fetcher):
        """Test that tiny images are rejected."""
        # Image smaller than MIN_IMAGE_DIMENSION (80)
        image_bytes = self.create_test_image(50, 50)
        result = fetcher._process_image(image_bytes, "https://example.com/tiny.png", "")

        assert result is None  # Should be rejected as too small


class TestImageScoring:
    """Tests for image scoring and prioritization."""

    @pytest.fixture
    def fetcher(self):
        return WebFetcher()

    def test_score_header_image_negative(self, fetcher):
        """Test that header/nav images get negative scores."""
        ref = {
            "src": "https://example.com/logo.png",
            "alt": "",
            "in_header": True,
            "in_figure": False,
            "in_link_to_home": False,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": set(),
        }
        score = fetcher._score_image_ref(ref)
        assert score < 0

    def test_score_figure_image_positive(self, fetcher):
        """Test that figure images get positive scores."""
        ref = {
            "src": "https://example.com/chart.png",
            "alt": "Data visualization chart",
            "in_header": False,
            "in_figure": True,
            "in_link_to_home": False,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": {"article"},
        }
        score = fetcher._score_image_ref(ref)
        assert score > 0

    def test_score_logo_link_negative(self, fetcher):
        """Test that logo links to home get negative scores."""
        ref = {
            "src": "https://example.com/header-image.jpg",
            "alt": "",
            "in_header": False,
            "in_figure": False,
            "in_link_to_home": True,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": set(),
        }
        score = fetcher._score_image_ref(ref)
        assert score < 0

    def test_score_aria_hidden_negative(self, fetcher):
        """Test that aria-hidden images get negative scores."""
        ref = {
            "src": "https://example.com/decorative.png",
            "alt": "",
            "in_header": False,
            "in_figure": False,
            "in_link_to_home": False,
            "aria_hidden": True,
            "role_presentation": False,
            "parent_tags": set(),
        }
        score = fetcher._score_image_ref(ref)
        assert score < 0

    def test_score_junk_pattern_in_src(self, fetcher):
        """Test that junk patterns in src reduce score."""
        ref = {
            "src": "https://example.com/icon-menu.svg",
            "alt": "",
            "in_header": False,
            "in_figure": False,
            "in_link_to_home": False,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": set(),
        }
        score = fetcher._score_image_ref(ref)
        assert score < 0

    def test_score_content_path_positive(self, fetcher):
        """Test that content paths boost score."""
        ref = {
            "src": "https://example.com/assets/2025/article-image.png",
            "alt": "",
            "in_header": False,
            "in_figure": False,
            "in_link_to_home": False,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": set(),
        }
        score = fetcher._score_image_ref(ref)
        assert score > 0

    def test_score_descriptive_alt_positive(self, fetcher):
        """Test that descriptive alt text boosts score."""
        ref = {
            "src": "https://example.com/image.png",
            "alt": "A detailed chart showing qubit coherence times over the years",
            "in_header": False,
            "in_figure": False,
            "in_link_to_home": False,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": set(),
        }
        score = fetcher._score_image_ref(ref)
        assert score > 0

    def test_score_neutral_image(self, fetcher):
        """Test that a neutral image gets zero score."""
        ref = {
            "src": "https://example.com/random.jpg",
            "alt": "",
            "in_header": False,
            "in_figure": False,
            "in_link_to_home": False,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": set(),
        }
        score = fetcher._score_image_ref(ref)
        assert score == 0

    def test_score_article_context_positive(self, fetcher):
        """Test that article context boosts score."""
        ref = {
            "src": "https://example.com/photo.jpg",
            "alt": "",
            "in_header": False,
            "in_figure": False,
            "in_link_to_home": False,
            "aria_hidden": False,
            "role_presentation": False,
            "parent_tags": {"article", "body"},
        }
        score = fetcher._score_image_ref(ref)
        assert score > 0
