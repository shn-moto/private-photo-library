#!/bin/sh

# Скрипт запуска бота с поддержкой статического домена Cloudflare Tunnel.
# Если TUNNEL_URL задан в env, используем его сразу. Иначе оставляем fallback
# на quick tunnel для локальной обратной совместимости.

if [ -n "$TUNNEL_URL" ]; then
  echo "Using configured tunnel URL: $TUNNEL_URL"
  exec python -m bot.telegram_bot
fi

echo "Waiting for Cloudflare tunnel to generate URL..."
MAX_RETRIES=30
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
  TUNNEL_URL=$(curl -s http://cloudflared:2000/metrics 2>/dev/null | grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' | head -n 1)

  if [ -n "$TUNNEL_URL" ]; then
    echo "Tunnel is up! URL: $TUNNEL_URL"
    export TUNNEL_URL="$TUNNEL_URL"
    exec python -m bot.telegram_bot
    exit 0
  fi

  echo "Waiting for tunnel URL (attempt $((COUNT+1))/$MAX_RETRIES)..."
  sleep 3
  COUNT=$((COUNT+1))
done

echo "Warning: Failed to get tunnel URL after $MAX_RETRIES attempts"
echo "Starting bot without tunnel URL..."
exec python -m bot.telegram_bot
