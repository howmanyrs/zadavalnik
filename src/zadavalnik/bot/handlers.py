import logging
import json # Для json.dumps в tool message - БОЛЬШЕ НЕ НУЖЕН ДЛЯ ЭТОЙ ЦЕЛИ
import base64
import io
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
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

def _clear_user_test_state(context: ContextTypes.DEFAULT_TYPE):
    keys_to_clear = [
        'current_state', 'current_topic', 'gpt_chat_history', 
        'current_question_num', 'total_questions', 'active_test_attempt_id',
        'test_from_image', 'test_from_document'
    ]
    for key in keys_to_clear:
        if key in context.user_data:
            del context.user_data[key]

async def _process_image_to_base64(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает изображение: скачивает и конвертирует в base64"""
    user_id = update.effective_user.id
    
    # Получаем самое большое изображение из отправленного
    photo = update.message.photo[-1]  # Берем самое большое разрешение
    
    # Скачиваем файл
    file = await context.bot.get_file(photo.file_id)
    
    # Скачиваем изображение в память
    image_bytes = io.BytesIO()
    await file.download_to_memory(image_bytes)
    image_bytes.seek(0)
    
    # Конвертируем в base64
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    
    logger.info(f"Successfully converted image to base64 for user {user_id}. Size: {len(image_base64)} chars")
    
    # Определяем формат изображения
    image_format = "jpeg"  # По умолчанию JPEG, можно улучшить определение формата
    
    return image_base64, image_format

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


async def _process_test_start_from_response(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                      gpt_response_data, gpt_history, topic, is_image_test=False):
    """Общая функция для обработки начала теста после получения ответа от OpenAI"""
    user_id = update.effective_user.id
    
    if not gpt_response_data:
        return False
        
    # Очищаем предыдущее состояние и логируем новую попытку в БД
    _clear_user_test_state(context)
    
    async for db in get_db_session():
        attempt = await log_test_attempt_start(db, user_id, topic)
        context.user_data['active_test_attempt_id'] = attempt.id
    
    # Обновляем состояние пользователя
    context.user_data.update({
        'current_topic': topic,
        'current_state': UserState.IN_TEST,
        'gpt_chat_history': gpt_history,
        'current_question_num': gpt_response_data.get("current_question_number"),
        'total_questions': gpt_response_data.get("total_questions_in_test"),
        'test_from_image': is_image_test  # Флаг, что тест создан из изображения
    })
    
    await update.message.reply_text(gpt_response_data["message_to_user"])
    
    # Проверяем, не завершился ли тест сразу
    if gpt_response_data.get("is_final_summary"):
        async for db in get_db_session():
            await update_test_attempt_status(db, context.user_data['active_test_attempt_id'], TestStatus.COMPLETED, end_time=True)
        context.user_data['current_state'] = UserState.TEST_COMPLETED
        await update.message.reply_text("Тест завершен! Для нового теста используйте /newtest.")
    
    return True

async def _get_openai_client(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получение клиента OpenAI из контекста бота"""
    openai_client: OpenAIClient = context.application.bot_data.get('openai_client')
    if not openai_client:
        logger.error("OpenAI client not found in bot_data.")
        await update.message.reply_text("Ошибка конфигурации бота. Обратитесь к администратору.")
        return None
    return openai_client

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик фотографий - анализ изображения и создание теста"""
    user_id = update.effective_user.id
    current_state = context.user_data.get('current_state')
    
    logger.info(f"User {user_id} (state: {current_state}) sent a photo")
    
    openai_client = await _get_openai_client(update, context)
    if not openai_client:
        return
    
    try:
        if current_state == UserState.AWAITING_TOPIC:
            # Уведомляем пользователя о начале обработки
            await update.message.reply_text("Анализирую изображение и создаю тест...")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # Обрабатываем изображение
            image_base64, image_format = await _process_image_to_base64(update, context)
            
            # Анализируем изображение через OpenAI и создаем тест
            gpt_response_data, gpt_history = await openai_client.analyze_image_and_start_test(
                image_base64=image_base64,
                image_format=image_format
            )
            
            # Получаем определенную тему из ответа ИИ и начинаем тест
            detected_topic = gpt_response_data.get("detected_topic", "Тест по изображению") if gpt_response_data else None
            
            success = await _process_test_start_from_response(
                update, context, gpt_response_data, gpt_history, 
                detected_topic, is_image_test=True
            )
            
            if not success:
                logger.warning(f"Failed to analyze image and start test for user {user_id}. Response data: {gpt_response_data}")
                await update.message.reply_text(
                    "Не удалось проанализировать изображение или создать тест. "
                    "Попробуйте другое изображение или начните обычный тест командой /newtest."
                )
        
        elif current_state == UserState.IN_TEST:
            # Обработка фото во время теста
            await update.message.reply_text("Обрабатываю ваше изображение в рамках текущего теста...")
            
            # Обрабатываем изображение
            image_base64, image_format = await _process_image_to_base64(update, context)
            
            # Тут должна быть логика обработки фото в контексте текущего теста
            # Это может потребовать создания отдельного метода в OpenAIClient
            # Пока используем текстовый вариант продолжения теста с уведомлением
            await update.message.reply_text(
                "Я вижу, что вы отправили изображение. Пожалуйста, опишите ваш ответ словами."
            )
            
        elif current_state == UserState.TEST_COMPLETED:
            await update.message.reply_text(
                "Тест уже завершен. Если хотите создать новый тест из изображения, используйте сначала команду /newtest."
            )
        
        elif current_state == UserState.START or not current_state:
            # Помогаем пользователю начать
            await update.message.reply_text(
                "Для начала работы, пожалуйста, используйте команду /start или /newtest, затем отправьте изображение."
            )
            _clear_user_test_state(context)
        
        else:
            logger.error(f"User {user_id} is in an unknown state: {current_state}")
            await update.message.reply_text("Произошла внутренняя ошибка состояния. Пожалуйста, перезапустите бота командой /start.")
            _clear_user_test_state(context)
            context.user_data['current_state'] = UserState.START
        
    except Exception as e:
        logger.error(f"Error processing photo for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "Произошла ошибка при обработке изображения. Попробуйте еще раз."
        )


async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик документов - анализ текстового файла и создание теста"""
    user_id = update.effective_user.id
    current_state = context.user_data.get('current_state')
    
    logger.info(f"User {user_id} (state: {current_state}) sent a document")
    
    openai_client = await _get_openai_client(update, context)
    if not openai_client:
        return
    
    try:
        if current_state == UserState.AWAITING_TOPIC:
            # Проверяем тип файла
            document = update.message.document
            
            # Проверяем, что это текстовый файл
            if not document.mime_type or not document.mime_type.startswith('text/'):
                if not document.file_name or not document.file_name.lower().endswith('.txt'):
                    await update.message.reply_text(
                        "Пожалуйста, отправьте текстовый файл в формате .txt"
                    )
                    return
            
            # Проверяем размер файла (примерно 50,000 слов = ~300KB для среднего текста)
            max_file_size = 500 * 1024  # 500KB для безопасности
            if document.file_size > max_file_size:
                await update.message.reply_text(
                    f"Файл слишком большой. Максимальный размер: {max_file_size // 1024}KB"
                )
                return
            
            await update.message.reply_text("Загружаю и анализирую документ...")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # Скачиваем файл
            file = await context.bot.get_file(document.file_id)
            
            # Скачиваем документ в память
            document_bytes = io.BytesIO()
            await file.download_to_memory(document_bytes)
            document_bytes.seek(0)
            
            # Читаем текст
            try:
                text_content = document_bytes.read().decode('utf-8')
            except UnicodeDecodeError:
                await update.message.reply_text(
                    "Не удалось прочитать файл. Убедитесь, что это текстовый файл в кодировке UTF-8."
                )
                return
            
            # Проверяем количество слов
            word_count = len(text_content.split())
            if word_count > 50000:
                await update.message.reply_text(
                    f"Документ содержит {word_count} слов, что превышает лимит в 50,000 слов. "
                    "Пожалуйста, отправьте более короткий документ."
                )
                return
            
            logger.info(f"Document processed for user {user_id}. Word count: {word_count}")
            
            # Получаем структурированные данные и обновленную историю от OpenAI
            gpt_response_data, gpt_history = await openai_client.analyze_text_and_start_test(text_content)
            
            if gpt_response_data:
                # Определяем тему на основе документа (первые 100 символов как краткое описание)
                topic = f"Документ: {document.file_name or 'text_document.txt'}"
                
                # Используем общую функцию для обработки начала теста
                success = await _process_test_start_from_response(
                    update, context, gpt_response_data, gpt_history, topic, is_image_test=False
                )
                
                # Помечаем, что тест создан из документа
                if success:
                    context.user_data['test_from_document'] = True
                    await update.message.reply_text(f"✅ Документ обработан ({word_count} слов)")
                
            else:
                logger.warning(f"Failed to start AI test session from document for user {user_id}")
                await update.message.reply_text(
                    "Не удалось создать тест на основе документа. Попробуйте другой файл или повторите позже."
                )
        
        elif current_state == UserState.IN_TEST:
            await update.message.reply_text(
                "Вы находитесь в процессе прохождения теста. Пожалуйста, ответьте на текущий вопрос текстом."
            )
            
        elif current_state == UserState.TEST_COMPLETED:
            await update.message.reply_text(
                "Тест уже завершен. Если хотите создать новый тест из документа, используйте команду /newtest."
            )
        
        elif current_state == UserState.START or not current_state:
            await update.message.reply_text(
                "Для начала работы используйте команду /start или /newtest, затем отправьте текстовый документ."
            )
            _clear_user_test_state(context)
        
        else:
            logger.error(f"User {user_id} is in an unknown state: {current_state}")
            await update.message.reply_text("Произошла внутренняя ошибка состояния. Пожалуйста, перезапустите бота командой /start.")
            _clear_user_test_state(context)
            context.user_data['current_state'] = UserState.START
        
    except Exception as e:
        logger.error(f"Error processing document for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "Произошла ошибка при обработке документа. Попробуйте еще раз."
        )


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
        
        # Проверяем, был ли тест создан из изображения или документа
        test_from_image = context.user_data.get('test_from_image', False)
        test_from_document = context.user_data.get('test_from_document', False)
        
        if test_from_image:
            # Используем метод для продолжения теста из изображения
            gpt_response_data, gpt_history = await openai_client.continue_image_test_session(
                history=current_gpt_history,
                user_message_text=text_received
            )
        elif test_from_document:
            # Используем метод для продолжения теста из документа
            gpt_response_data, gpt_history = await openai_client.continue_text_test_session(
                history=current_gpt_history,
                user_message_text=text_received
            )
        else:
            # Используем обычный метод для продолжения теста
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