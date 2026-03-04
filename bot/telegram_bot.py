"""Telegram bot для поиска фотографий через Smart Photo Indexing API."""

import os
import logging
from io import BytesIO

import httpx
from telegram import Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://api:8000")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TOP_K = int(os.getenv("TOP_K", "3"))
# Форматы для поиска (только основные фото-форматы)
BOT_FORMATS = os.getenv("BOT_FORMATS", "jpg,jpeg,heic,heif,nef").split(",")
# Модель по умолчанию
DEFAULT_MODEL = "ViT-L/14"
# URL туннеля cloudflared (устанавливается start_bot.sh)
TUNNEL_URL = os.getenv("TUNNEL_URL", "")
# Whitelist пользователей (user IDs через запятую)
ALLOWED_USERS = set()
if os.getenv("TELEGRAM_ALLOWED_USERS"):
    ALLOWED_USERS = {int(uid.strip()) for uid in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if uid.strip()}

# Доступные модели
AVAILABLE_MODELS = {
    "ViT-L/14": {"name": "ViT-L/14", "desc": "Большая (768 dim, лучшее качество)"},
    "SigLIP": {"name": "SigLIP", "desc": "SigLIP so400m (1152 dim, мультиязычная)"},
    "ViT-B/32": {"name": "ViT-B/32", "desc": "Базовая (512 dim, быстрая)"},
    "ViT-B/16": {"name": "ViT-B/16", "desc": "Базовая+ (512 dim, средняя)"},
}


def restricted(func):
    """Декоратор для ограничения доступа к командам бота."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        username = update.effective_user.username or "unknown"
        
        # Если whitelist пуст - разрешаем всем
        if not ALLOWED_USERS:
            logger.warning("TELEGRAM_ALLOWED_USERS не настроен - доступ открыт для всех!")
            return await func(update, context)
        
        # Проверяем наличие пользователя в whitelist
        if user_id not in ALLOWED_USERS:
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
    # Инициализация модели по умолчанию
    if "model" not in context.user_data:
        context.user_data["model"] = DEFAULT_MODEL
    
    current_model = context.user_data.get("model", DEFAULT_MODEL)
    user_id = update.effective_user.id
    username = update.effective_user.username or "unknown"
    
    logger.info(f"📋 User ID: {user_id}, Username: @{username}")
    
    await update.message.reply_text(
        "🔍 Поиск фотографий:\n"
        "- Отправьте текст — поиск по описанию\n"
        "- Отправьте фото — поиск похожих\n"
        "- Используйте Меню → /model для выбора модели\n\n"
        f"⚙️ Модель: {current_model}\n"
        f"📊 Результатов: {TOP_K}\n\n"
        f"👤 Ваш ID: `{user_id}`",
        reply_markup=ReplyKeyboardRemove()  # Удаляем кастомную клавиатуру
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
async def model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать меню выбора модели."""
    current_model = context.user_data.get("model", DEFAULT_MODEL)
    
    keyboard = []
    for model_key, model_info in AVAILABLE_MODELS.items():
        # Добавляем галочку к текущей модели
        prefix = "✅ " if model_key == current_model else "   "
        button_text = f"{prefix}{model_info['name']}"
        keyboard.append([
            InlineKeyboardButton(button_text, callback_data=f"model:{model_key}")
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎯 Выберите модель для поиска:\n\n"
        + "\n".join([f"• {info['name']}: {info['desc']}" for info in AVAILABLE_MODELS.values()]),
        reply_markup=reply_markup
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора модели."""
    query = update.callback_query
    await query.answer()
    
    # Извлекаем выбранную модель из callback_data
    callback_data = query.data
    if not callback_data.startswith("model:"):
        return
    
    selected_model = callback_data.split(":", 1)[1]
    
    if selected_model in AVAILABLE_MODELS:
        context.user_data["model"] = selected_model
        model_info = AVAILABLE_MODELS[selected_model]
        
        # Обновляем сообщение с галочками
        keyboard = []
        for model_key, model_data in AVAILABLE_MODELS.items():
            prefix = "✅ " if model_key == selected_model else "   "
            button_text = f"{prefix}{model_data['name']}"
            keyboard.append([
                InlineKeyboardButton(button_text, callback_data=f"model:{model_key}")
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"✅ Выбрана модель: {model_info['name']}\n"
            f"📝 {model_info['desc']}\n\n"
            "🎯 Выберите модель для поиска:\n\n"
            + "\n".join([f"• {info['name']}: {info['desc']}" for info in AVAILABLE_MODELS.values()]),
            reply_markup=reply_markup
        )
    else:
        await query.edit_message_text("❌ Неизвестная модель")


    
    # Если это команда кнопки - показываем меню модели
@restricted
async def search_by_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск по текстовому запросу."""
    query = update.message.text.strip()
    if not query:
        return

    # Получаем выбранную модель или используем дефолтную
    current_model = context.user_data.get("model", DEFAULT_MODEL)
    
    await update.message.reply_text(f"🔍 Ищу: «{query}»\n⚙️ Модель: {current_model}")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_URL}/search/text",
            json={
                "query": query,
                "top_k": TOP_K,
                "similarity_threshold": 0.1,
                "formats": BOT_FORMATS,
                "model": current_model,  # Передаем выбранную модель
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


@restricted
async def search_by_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск по фотографии."""
    photo = update.message.photo[-1]  # наибольшее разрешение
    file = await context.bot.get_file(photo.file_id)

    buf = BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)

    # Получаем выбранную модель или используем дефолтную
    current_model = context.user_data.get("model", DEFAULT_MODEL)

    await update.message.reply_text(f"🔍 Ищу похожие фото...\n⚙️ Модель: {current_model}")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_URL}/search/image",
            files={"file": ("photo.jpg", buf, "image/jpeg")},
            data={
                "top_k": str(TOP_K), 
                "similarity_threshold": "0.1",
                "model": current_model,  # Передаем выбранную модель
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
    
    # Логирование настроек безопасности
    if ALLOWED_USERS:
        logger.info(f"Whitelist включен: {len(ALLOWED_USERS)} разрешенных пользователей")
    else:
        logger.warning("⚠️  WHITELIST НЕ НАСТРОЕН - доступ открыт для всех!")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Устанавливаем меню команд бота
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("start", "Начать работу с ботом"),
            BotCommand("feed", "Открыть ленту фотографий"),
            BotCommand("map", "Открыть карту фотографий"),
            BotCommand("books", "Открыть библиотеку"),
            BotCommand("model", "Выбрать модель поиска"),
        ])

    app.post_init = post_init

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("feed", show_feed))
    app.add_handler(CommandHandler("map", show_map))
    app.add_handler(CommandHandler("books", show_books))
    app.add_handler(CommandHandler("model", model_menu))
    app.add_handler(CallbackQueryHandler(model_callback, pattern="^model:"))
    app.add_handler(MessageHandler(filters.PHOTO, search_by_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_by_text))

    logger.info(f"Бот запущен, API: {API_URL}, TOP_K: {TOP_K}, DEFAULT_MODEL: {DEFAULT_MODEL}")
    if TUNNEL_URL:
        logger.info(f"Tunnel URL: {TUNNEL_URL}")
    else:
        logger.warning("TUNNEL_URL не задан - команда /map будет недоступна")
    app.run_polling()


if __name__ == "__main__":
    main()
