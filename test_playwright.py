import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        await page.goto('https://hotro.tiki.vn/knowledge-base/post/1135')
        await page.wait_for_timeout(3000)
        text = await page.evaluate('document.body.innerText')
        with open('test_output.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
