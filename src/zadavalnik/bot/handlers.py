import logging
import json # Для json.dumps в tool message - БОЛЬШЕ НЕ НУЖЕН ДЛЯ ЭТОЙ ЦЕЛИ
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from zadavalnik.database.db import (
    get_db_session, 
    get_or_create_telegram_user_in_db,
    log_test_attempt_start,
    update_test_attempt_status,
    count_user_daily_tests,
    log_rate_limit_attempt
)
from zadavalnik.database.models import TestStatus
from zadavalnik.ai.openai_client import OpenAIClient
from zadavalnik.bot.states import UserState
from zadavalnik.config.settings import settings

logger = logging.getLogger(__name__)

def setup_handlers(app: Application):
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("newtest", new_test_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

def _clear_user_test_state(context: ContextTypes.DEFAULT_TYPE):
    keys_to_clear = [
        'current_state', 'current_topic', 'gpt_chat_history', 
        'current_question_num', 'total_questions', 'active_test_attempt_id'
    ]
    for key in keys_to_clear:
        if key in context.user_data:
            del context.user_data[key]

async def _initialize_new_test_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_tg = update.effective_user
    user_id = user_tg.id
    _clear_user_test_state(context)

    if settings.TEST_USER_TGID != int(user_id):
        async for db in get_db_session():
            await get_or_create_telegram_user_in_db(db, user_tg)
            tests_today = await count_user_daily_tests(db, user_id)
            logger.info(f"User {user_id} has {tests_today} tests today. Limit: {settings.MAX_TESTS_PER_DAY}")
            if tests_today >= settings.MAX_TESTS_PER_DAY:
                await log_rate_limit_attempt(db, user_id)
                await update.message.reply_text(
                    f"Вы уже прошли максимальное количество тестов на сегодня ({settings.MAX_TESTS_PER_DAY}). "
                    "Пожалуйста, возвращайтесь завтра!"
                )
                return False

    context.user_data['current_state'] = UserState.AWAITING_TOPIC
    await update.message.reply_text(
        "Добро пожаловать в Задавальник!\n"
        "Введите тему, по которой вы хотите пройти тест."
    )
    return True

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} used /start")
    await _initialize_new_test_session(update, context)

async def new_test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} used /newtest")
    await _initialize_new_test_session(update, context)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text_received = update.message.text
    current_state = context.user_data.get('current_state')

    logger.info(f"User {user_id} (state: {current_state}) sent text: '{text_received}'")

    openai_client: OpenAIClient = context.application.bot_data.get('openai_client')
    if not openai_client:
        logger.error("OpenAI client not found in bot_data.")
        await update.message.reply_text("Ошибка конфигурации бота. Обратитесь к администратору.")
        return

    if current_state == UserState.AWAITING_TOPIC:
        if len(text_received) < 3:
            await update.message.reply_text("Тема не задана. Попробуйте снова.")
            return

        await update.message.reply_text(f"Подготавливаю вопросы по теме: \"{text_received}\".")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        # Получаем структурированные данные и обновленную историю
        gpt_response_data, gpt_history = await openai_client.start_test_session(topic=text_received)
        
        # gpt_response_data - это уже распарсенный JSON, если модель его вернула корректно
        if gpt_response_data:
            # Логика добавления tool_message больше не нужна, gpt_history уже содержит ответ ассистента.
            
            async for db in get_db_session():
                attempt = await log_test_attempt_start(db, user_id, text_received)
                context.user_data['active_test_attempt_id'] = attempt.id
            
            context.user_data.update({
                'current_topic': text_received,
                'current_state': UserState.IN_TEST,
                'gpt_chat_history': gpt_history, # Сохраняем историю, включающую ответ ассистента с JSON
                'current_question_num': gpt_response_data.get("current_question_number"),
                'total_questions': gpt_response_data.get("total_questions_in_test")
            })
            await update.message.reply_text(gpt_response_data["message_to_user"])
            
            if gpt_response_data.get("is_final_summary"):
                async for db in get_db_session():
                    await update_test_attempt_status(db, context.user_data['active_test_attempt_id'], TestStatus.COMPLETED, end_time=True)
                context.user_data['current_state'] = UserState.TEST_COMPLETED
                await update.message.reply_text("Тест завершен! Для нового теста используйте /newtest.")
        else:
            logger.warning(f"Failed to start AI test session for user {user_id}, topic: {text_received}. Raw AI response might be in logs if parsing failed. Response data: {gpt_response_data}")
            await update.message.reply_text("Не удалось начать тест. Попробуйте другую тему или повторите позже. Возможно, ИИ вернул некорректный формат данных.")
            # Не меняем состояние, пользователь может попробовать ввести другую тему
    
    elif current_state == UserState.IN_TEST:
        active_test_id = context.user_data.get('active_test_attempt_id')
        if not active_test_id:
            logger.error(f"User {user_id} in IN_TEST state but no active_test_attempt_id found.")
            await update.message.reply_text("Произошла ошибка сессии. Пожалуйста, начните новый тест: /newtest")
            _clear_user_test_state(context)
            context.user_data['current_state'] = UserState.AWAITING_TOPIC # или START
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        current_gpt_history = context.user_data.get('gpt_chat_history', [])
        gpt_response_data, gpt_history = await openai_client.continue_test_session(
            history=current_gpt_history,
            user_message_text=text_received
        )

        if gpt_response_data:
            # Логика добавления tool_message больше не нужна
            context.user_data.update({
                'gpt_chat_history': gpt_history,
                'current_question_num': gpt_response_data.get("current_question_number"),
                # total_questions не должен меняться
            })

            await update.message.reply_text(gpt_response_data["message_to_user"])

            if gpt_response_data.get("is_final_summary"):
                async for db in get_db_session():
                    await update_test_attempt_status(db, active_test_id, TestStatus.COMPLETED, end_time=True)
                context.user_data['current_state'] = UserState.TEST_COMPLETED
                logger.info(f"Test {active_test_id} completed for user {user_id}")
                await update.message.reply_text("Тест завершен! Чтобы начать новый, используйте команду /newtest.")
        else:
            logger.warning(f"Failed to continue AI test session for user {user_id}, test_id {active_test_id}. Raw AI response might be in logs. Response data: {gpt_response_data}")
            await update.message.reply_text("Произошла ошибка при общении с ИИ. Попробуйте ответить еще раз. Если ошибка повторится, начните новый тест: /newtest. Возможно, ИИ вернул некорректный формат данных.")

    elif current_state == UserState.TEST_COMPLETED:
        await update.message.reply_text("Тест уже завершен. Чтобы начать новый, используйте команду /newtest.")
    
    elif current_state == UserState.START or not current_state:
         await update.message.reply_text("Пожалуйста, используйте команду /start или /newtest, чтобы начать.")
         _clear_user_test_state(context)

    else:
        logger.error(f"User {user_id} is in an unknown state: {current_state}")
        await update.message.reply_text("Произошла внутренняя ошибка состояния. Пожалуйста, перезапустите бота командой /start.")
        _clear_user_test_state(context)
        context.user_data['current_state'] = UserState.START