"""Telegram bot для поиска фотографий через Smart Photo Indexing API."""

import os
import logging
from io import BytesIO

import httpx
from telegram import Update, InputMediaPhoto, ReplyKeyboardRemove, BotCommand
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
# URL туннеля cloudflared (устанавливается start_bot.sh)
TUNNEL_URL = os.getenv("TUNNEL_URL", "")
# Whitelist пользователей (user IDs через запятую) — fallback if DB is unavailable
_ENV_ALLOWED_USERS = set()
if os.getenv("TELEGRAM_ALLOWED_USERS"):
    _ENV_ALLOWED_USERS = {int(uid.strip()) for uid in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if uid.strip()}

# Cache: telegram_id → bool (allowed)
_user_access_cache: dict = {}


async def _check_user_access(telegram_id: int) -> bool:
    """Check if telegram user exists in DB (app_user table). Falls back to env whitelist."""
    if telegram_id in _user_access_cache:
        return _user_access_cache[telegram_id]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_URL}/auth/check-telegram/{telegram_id}", timeout=5)
            if resp.status_code == 200:
                allowed = resp.json().get("allowed", False)
                _user_access_cache[telegram_id] = allowed
                return allowed
    except Exception as e:
        logger.warning(f"DB user check failed, falling back to env: {e}")
    # Fallback to env whitelist
    if _ENV_ALLOWED_USERS:
        return telegram_id in _ENV_ALLOWED_USERS
    return False  # No whitelist and no DB → deny




def restricted(func):
    """Декоратор для ограничения доступа к командам бота."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        username = update.effective_user.username or "unknown"

        allowed = await _check_user_access(user_id)
        if not allowed:
            logger.warning(f"Отклонен доступ для пользователя {user_id} (@{username})")
            await update.message.reply_text(
                "⛔️ У вас нет доступа к этому боту.\n"
                f"Ваш ID: {user_id}"
            )
            return

        logger.info(f"Доступ разрешен для пользователя {user_id} (@{username})")
        return await func(update, context)

    return wrapper


async def fetch_image(client: httpx.AsyncClient, image_id: str) -> bytes | None:
    """Получить полноразмерное изображение из API."""
    resp = await client.get(f"{API_URL}/image/{image_id}/full", timeout=60)
    if resp.status_code == 200:
        return resp.content
    return None


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or "unknown"
    
    logger.info(f"📋 User ID: {user_id}, Username: @{username}")
    
    await update.message.reply_text(
        "🔍 Поиск фотографий:\n"
        "- Отправьте текст — AI поиск по описанию\n"
        "- Отправьте фото — поиск похожих\n\n"
        f"📊 Результатов: {TOP_K}\n\n"
        f"👤 Ваш ID: `{user_id}`",
        reply_markup=ReplyKeyboardRemove()
    )


async def _create_session_token(tg_user) -> str | None:
    """Создать сессию через API и вернуть токен."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{API_URL}/auth/session",
                json={"telegram_id": tg_user.id, "display_name": tg_user.full_name or tg_user.username or "User"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("token")
    except Exception as e:
        logger.warning(f"Failed to create session: {e}")
    return None


@restricted
async def show_map(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать ссылку на карту фотографий с токеном сессии."""
    if not TUNNEL_URL:
        await update.message.reply_text(
            "❌ Карта временно недоступна.\n"
            "Туннель не настроен."
        )
        return

    token = await _create_session_token(update.effective_user)
    if not token:
        await update.message.reply_text("❌ Не удалось создать сессию. Попробуйте позже.")
        return

    map_url = f"{TUNNEL_URL}/s/{token}"

    # Получаем статистику карты
    stats_text = ""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_URL}/map/stats", timeout=10)
            if resp.status_code == 200:
                stats = resp.json()
                stats_text = (
                    f"\n\n📊 Статистика:\n"
                    f"• Фото с GPS: {stats.get('with_gps', 0):,}\n"
                    f"• Всего фото: {stats.get('total_photos', 0):,}"
                )
    except Exception as e:
        logger.warning(f"Failed to get map stats: {e}")

    await update.message.reply_text(
        f'<a href="{map_url}">🗺 Открыть карту</a>'
        f"{stats_text}\n\n"
        f"<i>Ссылка действительна 30 мин без активности</i>",
        disable_web_page_preview=True,
        parse_mode="HTML",
    )


@restricted
async def show_books(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать ссылку на библиотеку."""
    if not TUNNEL_URL:
        await update.message.reply_text(
            "❌ Библиотека временно недоступна.\n"
            "Туннель не настроен."
        )
        return

    token = await _create_session_token(update.effective_user)
    if not token:
        await update.message.reply_text("❌ Не удалось создать сессию. Попробуйте позже.")
        return

    books_url = f"{TUNNEL_URL}/sb/{token}"

    # Получаем список книг
    stats_text = ""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_URL}/books/list", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("books", []))
                stats_text = f"\n\n📚 Книг в библиотеке: {count}"
    except Exception as e:
        logger.warning(f"Failed to get books list: {e}")

    await update.message.reply_text(
        f'<a href="{books_url}">📖 Открыть библиотеку</a>'
        f"{stats_text}\n\n"
        f"<i>Ссылка действительна 30 мин без активности</i>",
        disable_web_page_preview=True,
        parse_mode="HTML",
    )


@restricted
async def show_feed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать ссылку на хронологическую ленту фотографий."""
    if not TUNNEL_URL:
        await update.message.reply_text(
            "❌ Лента временно недоступна.\n"
            "Туннель не настроен."
        )
        return

    token = await _create_session_token(update.effective_user)
    if not token:
        await update.message.reply_text("❌ Не удалось создать сессию. Попробуйте позже.")
        return

    feed_url = f"{TUNNEL_URL}/sf/{token}"

    # Получаем общее количество фото
    stats_text = ""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_URL}/timeline/photos?limit=1&offset=0", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("total", 0)
                stats_text = f"\n\n📊 Всего фото: {total:,}"
    except Exception as e:
        logger.warning(f"Failed to get timeline stats: {e}")

    await update.message.reply_text(
        f'<a href="{feed_url}">📷 Открыть ленту</a>'
        f"{stats_text}\n\n"
        f"<i>Ссылка действительна 30 мин без активности</i>",
        disable_web_page_preview=True,
        parse_mode="HTML",
    )


@restricted
async def search_by_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """AI-powered text search: query → Gemini → structured filters → multi-model CLIP search."""
    query = update.message.text.strip()
    if not query:
        return

    await update.message.reply_text(f"🤖 AI ищу: «{query}»")

    async with httpx.AsyncClient() as client:
        # Step 1: Ask AI assistant to interpret the query
        search_request = {
            "query": "",
            "top_k": TOP_K,
            "similarity_threshold": 0.1,
            "formats": BOT_FORMATS,
            "multi_model": True,
        }
        ai_message = ""

        try:
            ai_resp = await client.post(
                f"{API_URL}/ai/search-assistant",
                json={
                    "message": query,
                    "conversation_history": list(context.user_data.get("ai_history", [])),
                    "current_state": {},
                    "telegram_id": update.effective_user.id,
                },
                timeout=30,
            )
            if ai_resp.status_code == 200:
                ai_data = ai_resp.json()
                ai_message = ai_data.get("message", "")
                # Save conversation history for follow-up queries
                context.user_data["ai_history"] = ai_data.get("conversation_history", [])[-10:]

                for action in ai_data.get("actions", []):
                    atype = action.get("type")
                    if atype == "text_search":
                        search_request["query"] = action.get("clip_prompt") or action.get("query", query)
                        if action.get("tag_ids"):
                            search_request["tag_ids"] = action["tag_ids"]
                        if action.get("exclude_tag_ids"):
                            search_request["exclude_tag_ids"] = action["exclude_tag_ids"]
                    elif atype == "set_persons":
                        search_request["person_ids"] = action.get("person_ids", [])
                        search_request["person_mode"] = action.get("mode", "or")
                    elif atype == "set_bounds":
                        for k in ("min_lat", "max_lat", "min_lon", "max_lon"):
                            if k in action:
                                search_request[k] = action[k]
                    elif atype == "set_date_range":
                        if action.get("date_from"):
                            search_request["date_from"] = action["date_from"]
                        if action.get("date_to"):
                            search_request["date_to"] = action["date_to"]
                    elif atype == "set_formats":
                        search_request["formats"] = action.get("formats", BOT_FORMATS)
            else:
                logger.warning(f"AI assistant returned {ai_resp.status_code}, falling back to direct search")
                search_request["query"] = query
        except Exception as e:
            logger.warning(f"AI assistant failed: {e}, falling back to direct search")
            search_request["query"] = query

        # If AI didn't produce a text_search, use original query as fallback
        if not search_request["query"]:
            search_request["query"] = query

        # Step 2: Execute the search
        resp = await client.post(
            f"{API_URL}/search/text",
            json=search_request,
            timeout=60,
        )

        if resp.status_code != 200:
            await update.message.reply_text(f"Ошибка API: {resp.status_code}")
            return

        data = resp.json()
        results = data.get("results", data) if isinstance(data, dict) else data

        if not results:
            no_results_text = "Ничего не найдено."
            if ai_message:
                no_results_text = f"💬 {ai_message}\n\n{no_results_text}"
            await update.message.reply_text(no_results_text)
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
            if ai_message:
                media_group[0] = InputMediaPhoto(
                    media=media_group[0].media,
                    caption=f"💬 {ai_message}",
                )
            await update.message.reply_media_group(media=media_group)
        else:
            await update.message.reply_text("Не удалось загрузить фото.")


@restricted
async def search_by_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск по фотографии."""
    photo = update.message.photo[-1]  # наибольшее разрешение
    file = await context.bot.get_file(photo.file_id)

    buf = BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)

    await update.message.reply_text("🔍 Ищу похожие фото...")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_URL}/search/image",
            files={"file": ("photo.jpg", buf, "image/jpeg")},
            data={
                "top_k": str(TOP_K), 
                "similarity_threshold": "0.1",
            },
            timeout=60,
        )

        if resp.status_code != 200:
            await update.message.reply_text(f"Ошибка API: {resp.status_code}")
            return

        data = resp.json()
        # API returns TextSearchResponse: {results: [...], model: ...}
        results = data.get("results", data) if isinstance(data, dict) else data

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
    """Запуск бота"""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан!")
        return
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Устанавливаем меню команд бота
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("start", "Начать работу с ботом"),
            BotCommand("feed", "Открыть ленту фотографий"),
            BotCommand("map", "Открыть карту фотографий"),
            BotCommand("books", "Открыть библиотеку"),
        ])

    app.post_init = post_init

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("feed", show_feed))
    app.add_handler(CommandHandler("map", show_map))
    app.add_handler(CommandHandler("books", show_books))
    app.add_handler(MessageHandler(filters.PHOTO, search_by_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_by_text))

    logger.info(f"Бот запущен, API: {API_URL}, TOP_K: {TOP_K}")
    if TUNNEL_URL:
        logger.info(f"Tunnel URL: {TUNNEL_URL}")
    else:
        logger.warning("TUNNEL_URL не задан - команда /map будет недоступна")
    app.run_polling()


if __name__ == "__main__":
    main()
