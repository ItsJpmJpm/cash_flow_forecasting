"""Cash Flow Forecasting - Telegram Bot for Low Demand Alerts"""

import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, CallbackQueryHandler
)
from typing import Optional, List, Dict
import logging

import sys
sys.path.append(str(__file__).replace("telegram_bot.py", ""))

import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CashFlowAlertBot:
    """Telegram bot for cash flow low demand alerts."""

    def __init__(self, token: str = None):
        """
        Initialize the bot.

        Args:
            token: Telegram bot token (from .env or parameter)
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("Telegram bot token not provided. Set TELEGRAM_BOT_TOKEN in .env")

        self.threshold = float(os.getenv('ALERT_THRESHOLD', config.TELEGRAM_CONFIG['alert_threshold']))
        self.chat_ids: List[int] = []
        self.authorized_users: List[int] = []  # Will be populated on /start

        self.app = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id
        self.authorized_users.append(user_id)
        self.chat_ids.append(update.message.chat_id)

        welcome_message = (
            "💰 *Cash Flow Alert Bot* 💰\n\n"
            "¡Bienvenido al sistema de alertas de flujo de caja!\n\n"
            f"📊 *Umbral actual*: {self.threshold:,.0f} €\n\n"
            "*Comandos disponibles:*\n"
            "/forecast - Ver predicción actual\n"
            "/alert <cantidad> - Cambiar umbral de alerta\n"
            "/status - Ver estado actual\n"
            "/help - Ver ayuda"
        )

        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"User {user_id} started the bot")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = (
            "📖 *Guía de Uso*\n\n"
            "Este bot monitorea el cash flow de TechSolutions Toledo y te avisa cuando "
            "la demanda está baja.\n\n"
            "*触发条件:*\n"
            "El bot verifica cuando el cash flow predicho está por debajo del umbral configurado.\n\n"
            "*Umbral actual:*\n"
            f"{self.threshold:,.0f} €\n\n"
            "*Para cambiar el umbral:*\n"
            "`/alert 60000` - Cambia a 60.000 €\n\n"
            "El bot envía alertas automáticas según el calendario configurado."
        )

        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alert command to set threshold."""
        try:
            if context.args:
                new_threshold = float(context.args[0])
                old_threshold = self.threshold
                self.threshold = new_threshold

                await update.message.reply_text(
                    f"✅ *Umbral actualizado*\n\n"
                    f"Anterior: {old_threshold:,.0f} €\n"
                    f"Nuevo: {new_threshold:,.0f} €",
                    parse_mode='Markdown'
                )
                logger.info(f"Threshold changed from {old_threshold} to {new_threshold}")
            else:
                await update.message.reply_text(
                    f"📊 Umbral actual: {self.threshold:,.0f} €\n\n"
                    "Usa `/alert <cantidad>` para cambiarlo",
                    parse_mode='Markdown'
                )
        except ValueError:
            await update.message.reply_text("❌ Por favor, introduce un número válido.")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        status_text = (
            "📊 *Estado del Sistema*\n\n"
            f"*Empresa:* {config.COMPANY_NAME}\n"
            f"*Ubicación:* {config.COMPANY_LOCATION}\n"
            f"*Umbral de alerta:* {self.threshold:,.0f} €\n"
            f"*Período de datos:* 2021-2023\n"
            f"*Horizonte de predicción:* {config.FORECAST_HORIZON} meses\n\n"
            "ℹ️ Usa /forecast para ver la última predicción"
        )

        await update.message.reply_text(status_text, parse_mode='Markdown')

    async def forecast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /forecast command - placeholder for actual forecast."""
        # This would be connected to the actual model
        forecast_text = (
            "📈 *Predicción de Cash Flow*\n\n"
            "Conectar modelo de predicción...\n\n"
            "Para ver predicciones reales, ejecuta el script de predicción "
            "y actualiza los datos del bot."
        )

        await update.message.reply_text(forecast_text, parse_mode='Markdown')

    async def send_alert(self, chat_id: int,
                         cash_flow: float,
                         percentage_change: float,
                         factors: List[str],
                         prediction_date: str):
        """
        Send low demand alert to user.

        Args:
            chat_id: Telegram chat ID
            cash_flow: Predicted cash flow amount
            percentage_change: % change vs previous month
            factors: List of detected factors
            prediction_date: Date of prediction
        """
        # Emoji based on severity
        if cash_flow < 0:
            emoji = "🔴"
            severity = "CRÍTICO"
        elif cash_flow < self.threshold * 0.5:
            emoji = "🟠"
            severity = "ALTO"
        else:
            emoji = "🟡"
            severity = "MODERADO"

        message = (
            f"{emoji} *ALERTA: Baja demanda detectada*\n\n"
            f"*Severidad:* {severity}\n"
            f"*Fecha:* {prediction_date}\n\n"
            f"💰 *Cash Flow Predicho:* {cash_flow:,.0f} €\n"
            f"📊 *Cambio vs mes anterior:* {percentage_change:+.1f}%\n\n"
            f"*Factores detectados:*\n"
        )

        for factor in factors:
            message += f"   • {factor}\n"

        message += (
            f"\n⚠️ *Recomendación:*\n"
            "Revisar pipeline de proyectos y acelerar cobros pendientes."
        )

        keyboard = [
            [
                InlineKeyboardButton("📊 Ver detalles", callback_data="show_details"),
                InlineKeyboardButton("🔕 Silenciar 24h", callback_data="snooze")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await self.app.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query

        if query.data == "show_details":
            await query.answer("Mostrando detalles...")
            await query.edit_message_text(
                "📊 *Detalles de la alerta*\n\n"
                "Para ver el análisis completo, consulta los notebooks de evaluación."
            )
        elif query.data == "snooze":
            await query.answer("Alerta silenciada por 24 horas")
            await query.edit_message_text(
                "🔕 *Alerta silenciada*\n\n"
                "No recibirás más alertas hasta mañana."
            )

    def build_app(self) -> Application:
        """Build and configure the Telegram application."""
        self.app = Application.builder().token(self.token).build()

        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("alert", self.alert_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("forecast", self.forecast_command))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))

        return self.app

    async def run_prediction_check(self, predictions: List[Dict]):
        """
        Check predictions and send alerts if needed.

        Args:
            predictions: List of dicts with 'date', 'cash_flow', 'previous_cash_flow'
        """
        for pred in predictions:
            cash_flow = pred['cash_flow']

            if cash_flow < self.threshold:
                percentage_change = ((cash_flow - pred['previous_cash_flow']) /
                                     pred['previous_cash_flow'] * 100) if pred['previous_cash_flow'] else 0

                factors = []
                if percentage_change < -10:
                    factors.append("Caída significativa vs mes anterior")
                if cash_flow < 0:
                    factors.append("Cash flow negativo - acción urgente")
                if pred.get('seasonal') and pred['seasonal'] < 0.75:
                    factors.append("Estacionalidad baja (vacaciones/típico bajo)")

                for chat_id in self.chat_ids:
                    await self.send_alert(
                        chat_id=chat_id,
                        cash_flow=cash_flow,
                        percentage_change=percentage_change,
                        factors=factors,
                        prediction_date=pred['date']
                    )

    def run(self):
        """Run the bot."""
        app = self.build_app()
        print(f"Bot starting with token: {self.token[:10]}...")
        print("Press Ctrl+C to stop")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


def create_env_file():
    """Create .env file with template."""
    env_content = (
        "# Telegram Bot Configuration\n"
        "# Get your token from @BotFather on Telegram\n"
        "TELEGRAM_BOT_TOKEN=your_bot_token_here\n\n"
        "# Alert threshold in euros (default: 50000)\n"
        "ALERT_THRESHOLD=50000\n"
    )

    with open('.env', 'w') as f:
        f.write(env_content)

    print("Created .env file. Please fill in your TELEGRAM_BOT_TOKEN")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Cash Flow Telegram Bot')
    parser.add_argument('--create-env', action='store_true',
                        help='Create .env file template')
    parser.add_argument('--token', type=str, help='Telegram bot token')

    args = parser.parse_args()

    if args.create_env:
        create_env_file()
    elif args.token:
        bot = CashFlowAlertBot(token=args.token)
        bot.run()
    else:
        try:
            bot = CashFlowAlertBot()
            bot.run()
        except ValueError as e:
            print(f"Error: {e}")
            print("\nTo set up the bot:")
            print("1. Create a bot via @BotFather on Telegram")
            print("2. Get your token")
            print("3. Create .env file or pass --token argument")
            print("\nUsage: python telegram_bot.py --token YOUR_TOKEN")