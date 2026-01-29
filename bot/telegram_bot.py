"""Telegram bot для поиска фотографий через Smart Photo Indexing API."""

import os
import logging
from io import BytesIO

import httpx
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://api:8000")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TOP_K = int(os.getenv("TOP_K", "3"))
# Форматы для поиска (только основные фото-форматы)
BOT_FORMATS = os.getenv("BOT_FORMATS", "jpg,jpeg,heic,heif,nef").split(",")


async def fetch_image(client: httpx.AsyncClient, image_id: str) -> bytes | None:
    """Получить полноразмерное изображение из API."""
    resp = await client.get(f"{API_URL}/image/{image_id}/full", timeout=60)
    if resp.status_code == 200:
        return resp.content
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Поиск фотографий:\n"
        "- Отправьте текст — поиск по описанию\n"
        "- Отправьте фото — поиск похожих\n\n"
        f"Выдаю {TOP_K} лучших результатов."
    )


async def search_by_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск по текстовому запросу."""
    query = update.message.text.strip()
    if not query:
        return

    await update.message.reply_text(f"Ищу: «{query}»...")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_URL}/search/text",
            json={
                "query": query,
                "top_k": TOP_K,
                "similarity_threshold": 0.1,
                "formats": BOT_FORMATS,
            },
            timeout=60,
        )

        if resp.status_code != 200:
            await update.message.reply_text(f"Ошибка API: {resp.status_code}")
            return

        data = resp.json()
        # API returns TextSearchResponse: {results: [...], translated_query: ...}
        results = data.get("results", data) if isinstance(data, dict) else data

        if not results:
            await update.message.reply_text("Ничего не найдено.")
            return

        media_group = []
        for item in results[:TOP_K]:
            image_id = item["image_id"]
            similarity = item.get("similarity", 0)
            file_name = item.get("file_name", "")

            img_bytes = await fetch_image(client, image_id)
            if img_bytes:
                media_group.append(
                    InputMediaPhoto(
                        media=BytesIO(img_bytes),
                        caption=f"{file_name} [{similarity:.0%}]",
                    )
                )

        if media_group:
            await update.message.reply_media_group(media=media_group)
        else:
            await update.message.reply_text("Не удалось загрузить фото.")


async def search_by_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск по фотографии."""
    photo = update.message.photo[-1]  # наибольшее разрешение
    file = await context.bot.get_file(photo.file_id)

    buf = BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)

    await update.message.reply_text("Ищу похожие фото...")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_URL}/search/image",
            files={"file": ("photo.jpg", buf, "image/jpeg")},
            data={"top_k": str(TOP_K), "similarity_threshold": "0.1"},
            timeout=60,
        )

        if resp.status_code != 200:
            await update.message.reply_text(f"Ошибка API: {resp.status_code}")
            return

        results = resp.json()  # API returns List[SearchResult]

        if not results:
            await update.message.reply_text("Похожих фото не найдено.")
            return

        media_group = []
        for item in results[:TOP_K]:
            image_id = item["image_id"]
            similarity = item.get("similarity", 0)
            file_name = item.get("file_name", "")

            img_bytes = await fetch_image(client, image_id)
            if img_bytes:
                media_group.append(
                    InputMediaPhoto(
                        media=BytesIO(img_bytes),
                        caption=f"{file_name} [{similarity:.0%}]",
                    )
                )

        if media_group:
            media_group[0] = InputMediaPhoto(
                media=media_group[0].media,
                caption=f"Найдено {len(media_group)} похожих фото",
            )
            await update.message.reply_media_group(media=media_group)
        else:
            await update.message.reply_text("Не удалось загрузить фото.")


def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан!")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, search_by_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_by_text))

    logger.info(f"Бот запущен, API: {API_URL}, TOP_K: {TOP_K}")
    app.run_polling()


if __name__ == "__main__":
    main()
