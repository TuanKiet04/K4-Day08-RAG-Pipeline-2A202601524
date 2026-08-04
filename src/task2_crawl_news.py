"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    # Ví dụ (trang công khai Tiki):
    "https://hotro.tiki.vn/knowledge-base/post/850-dieu-khoan-su-dung",
    "https://hotro.tiki.vn/knowledge-base/post/778-chinh-sach-giai-quyet-khieu-nai",
    "https://hotro.tiki.vn/knowledge-base/post/802-chinh-sach-doi-tra-san-pham",
    "https://hotro.tiki.vn/knowledge-base/post/838-cac-hinh-thuc-giao-hang-tai-tiki",
    "https://hotro.tiki.vn/knowledge-base/post/1135",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from playwright.async_api import async_playwright

    # TODO: Implement crawling logic using pure playwright to avoid anti-bot
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            
            # Đợi tải trang và thử lấy data tối đa 5 lần (mỗi lần 3s) để tránh lỗi mạng chậm
            text = ""
            for _ in range(5):
                await page.wait_for_timeout(3000) # Đợi JS render
                text = await page.evaluate('document.body.innerText')
                if text and len(text.strip()) > 200: # Nếu nội dung đủ dài thì dừng đợi
                    break

            title = await page.title()
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            title = "Unknown"
            text = ""
        finally:
            await browser.close()
        
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": text,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
