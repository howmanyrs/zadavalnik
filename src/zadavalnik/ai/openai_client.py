import json
import logging
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI

from zadavalnik.config.settings import settings # Убедитесь, что импорт settings корректен

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str, model_name: str = settings.OPENAI_MODEL):
        self.client = AsyncOpenAI(api_key=api_key, base_url=settings.OPENAI_API_URL)
        self.model = model_name

    def _get_system_prompt_for_test(self, topic: str, history: Optional[List[Dict]] = None) -> str:
        return f"""
        Ты — бот Zadavalnik помощник для повторения материала.
        Проводящишь интерактивное тестирование по теме: "{topic}".

        Твоя задача — задавать вопросы пользователю один за другим.
        Тест должен состоять из нескольких вопросов (например, 3-5, определи это сам в первом сообщении).

        В начале первого вопроса сообщи "Сейчас мы проведем интерактивный тест по [тема теста]" и количество вопросов.

        Когда ты получил ответ пользователя на вопрос, ты должен:
        ## 1
        Если ответ пользователя хоть как-то связан с вопросом: 
            Дать на него краткий комментарий: (правильно/неправильно, дай короткое пояснение при необходимости).
        Если же пользователь отвечает не по теме вопроса или говорит, что не знает, забыл, простит сказать ответ.
            Просто дать ответ ответ на вопрос без комментариев.
        ## 2
        В этом же сообщении (через строку) пиши текст СЛЕДУЮЩЕГО вопроса.
        
        ВАЖНО: Ты ДОЛЖЕН форматировать КАЖДОЕ свое сообщение пользователю как JSON объект.
        Этот JSON объект должен содержать следующие поля:
        - "message_to_user": (string) Текст сообщения для пользователя. Это то, что увидит пользователь.
        - "current_question_number": (integer) Текущий порядковый номер ЗАДАВАЕМОГО вопроса. Начинается с 1 для первого вопроса, 2 для второго и т.д. Если ты комментируешь ответ на вопрос N и затем задаешь вопрос N+1, current_question_number должен быть N+1.
        - "total_questions_in_test": (integer) Общее количество вопросов, которое ты планируешь задать в этом тесте. Должно быть установлено в первом вызове и не меняться.
        - "is_final_summary": (integer) Установи в 1, если это финальное сообщение с подведением итогов теста. В остальных случаях 0.

        Пример твоего ответа:
        {{
            "message_to_user": "Верно! Молодец.\\n\\nСледующий вопрос: Как называется столица Франции?",
            "current_question_number": 2,
            "total_questions_in_test": 3,
            "is_final_summary": 0
        }}

        Еще пример (финальное резюме):
        {{
            "message_to_user": "Тест завершен. Вы ответили правильно на 2 из 3 вопросов. Стоит повторить тему X.",
            "current_question_number": 3, 
            "total_questions_in_test": 3,
            "is_final_summary": 1
        }}
                
        - Поле 'total_questions_in_test' должно быть заполнено с первого же сообщения и оставаться консистентным.
        - Поле 'current_question_number' должно корректно инкрементироваться для каждого НОВОГО вопроса. 
        - Поле 'is_final_summary' должно быть 1 ТОЛЬКО для самого последнего сообщения, завершающего тест.
        
        Если это был последний вопрос: 
        - предоставь краткое резюме в 'message_to_user': На какие вопросы пользователь ответил верно, а какие темы стоит повторить.
        - Если были неточности в формулировках ответов пользователя, укажи на них.

        Веди диалог последовательно. Твой ответ должен быть ТОЛЬКО JSON объектом, без какого-либо другого текста до или после него.
        """


    async def _make_openai_call(self, current_messages_for_api: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
        logger.debug(f"OpenAIClient: Sending messages to API: {json.dumps(current_messages_for_api, indent=2, ensure_ascii=False)}")
        
        final_history_after_call = list(current_messages_for_api)
        parsed_data: Optional[Dict] = None

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=current_messages_for_api,
                response_format={"type": "json_object"}, 
                max_tokens=3000,
            )
            
            response_message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            assistant_response_content = response_message.content
            
            # Проверяем, был ли ответ обрезан
            if finish_reason == "length":
                logger.warning(f"OpenAI response was truncated due to token limit. Consider increasing max_tokens or reducing context.")
            
            assistant_message_dict_for_history = {"role": "assistant", "content": assistant_response_content}
            final_history_after_call.append(assistant_message_dict_for_history)

            if assistant_response_content:
                logger.debug(f"OpenAIClient: Raw assistant_response_content before parsing (len={len(assistant_response_content)}): >>>{assistant_response_content}<<<")
                
                content_to_parse = assistant_response_content.strip()

                try:
                    parsed_data = json.loads(content_to_parse)
                    logger.debug(f"OpenAIClient: Successfully parsed directly (after strip): {parsed_data}")
                except json.JSONDecodeError as e_direct:
                    logger.warning(
                        f"OpenAIClient: Direct JSON parsing failed for content (len={len(content_to_parse)}): "
                        f">>>{content_to_parse}<<< Error: {e_direct}. Will attempt to extract JSON block."
                    )
                    
                    json_start = content_to_parse.find('{')
                    json_end = content_to_parse.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str_extracted = content_to_parse[json_start:json_end]
                        logger.debug(
                            f"OpenAIClient: Extracted JSON string for parsing (len={len(json_str_extracted)}): "
                            f">>>{json_str_extracted}<<<"
                        )
                        try:
                            parsed_data = json.loads(json_str_extracted)
                            logger.debug(f"OpenAIClient: Successfully parsed after extraction: {parsed_data}")
                        except json.JSONDecodeError as e_extracted:
                            logger.error(
                                f"OpenAIClient: Failed to parse extracted JSON string: >>>{json_str_extracted}<<<", 
                                exc_info=True
                            )
                            logger.error(f"JSONDecodeError details: msg='{e_extracted.msg}', doc_len={len(e_extracted.doc)}, pos={e_extracted.pos}")
                            
                            context_window = 20
                            start_idx = max(0, e_extracted.pos - context_window)
                            end_idx = min(len(e_extracted.doc), e_extracted.pos + context_window + 5)
                            
                            error_context_raw = e_extracted.doc[start_idx:end_idx]
                            error_context_repr = repr(error_context_raw)
                            error_context_bytes = error_context_raw.encode('utf-8', errors='surrogatepass')

                            logger.error(f"Context around error position ({e_extracted.pos}) (raw): >>>{error_context_raw}<<<")
                            logger.error(f"Context around error position ({e_extracted.pos}) (repr): >>>{error_context_repr}<<<")
                            logger.error(f"Context around error position ({e_extracted.pos}) (bytes): >>>{error_context_bytes}<<<")

                            if e_extracted.pos < len(e_extracted.doc):
                                char_at_error = e_extracted.doc[e_extracted.pos]
                                byte_at_error_char = char_at_error.encode('utf-8', errors='surrogatepass')
                                logger.error(f"Character at error position ({e_extracted.pos}): '{char_at_error}' (ASCII/Unicode: {ord(char_at_error)}), Bytes: {byte_at_error_char}")
                            else:
                                logger.error(f"Error position ({e_extracted.pos}) is at or beyond the end of the document (len: {len(e_extracted.doc)}).")
                    else:
                        logger.error(f"OpenAIClient: Could not find JSON block ({{...}}) in content: >>>{content_to_parse}<<<")
            else:
                logger.warning(f"OpenAIClient: AI response had no content. Finish reason: {finish_reason}")
                # Если ответ был обрезан, возвращаем специальное сообщение
                if finish_reason == "length":
                    parsed_data = {
                        "message_to_user": "Извините, произошла ошибка при генерации ответа. Попробуйте переформулировать ваш вопрос более кратко.",
                        "current_question_number": 1,
                        "total_questions_in_test": 1,
                        "is_final_summary": 1
                    }
            
            return parsed_data, final_history_after_call

        except Exception as e:
            logger.error(f"Exception in _make_openai_call during API request or initial processing", exc_info=True)
            return None, current_messages_for_api


    async def start_test_session(self, topic: str) -> Tuple[Optional[Dict], List[Dict]]:
        system_message_content = self._get_system_prompt_for_test(topic)
        messages_for_api_call = [{"role": "system", "content": system_message_content}]
        
        parsed_data, updated_history = await self._make_openai_call(messages_for_api_call)
        return parsed_data, updated_history


    async def continue_test_session(self, history: List[Dict], user_message_text: str) -> Tuple[Optional[Dict], List[Dict]]:
        messages_for_api_call = list(history)
        messages_for_api_call.append({"role": "user", "content": user_message_text})
        
        parsed_data, updated_history = await self._make_openai_call(messages_for_api_call)
        return parsed_data, updated_history

    def _get_system_prompt_for_image_analysis(self) -> str:
        return """
        Ты — бот Zadavalnik помощник для повторения материала.
        Проводящишь интерактивное тестирование по теме: "{Тема, связанная с тем, что ты увидел на изображении}".

        Твоя задача — задавать вопросы пользователю один за другим.
        Тест должен состоять из нескольких вопросов (например, 3-5, определи это сам в первом сообщении).

        В начале первого вопроса сообщи "Сейчас мы проведем интерактивный тест по [тема теста]" и количество вопросов.

        Когда ты получил ответ пользователя на вопрос, ты должен:
        ## 1
        Если ответ пользователя хоть как-то связан с вопросом: 
            Дать на него краткий комментарий: (правильно/неправильно, дай короткое пояснение при необходимости).
        Если же пользователь отвечает не по теме вопроса или говорит, что не знает, забыл, простит сказать ответ.
            Просто дать ответ ответ на вопрос без комментариев.
        ## 2
        В этом же сообщении (через строку) пиши текст СЛЕДУЮЩЕГО вопроса.
        
        ВАЖНО: Ты ДОЛЖЕН форматировать КАЖДОЕ свое сообщение пользователю как JSON объект.
        Этот JSON объект должен содержать следующие поля:
        - "message_to_user": (string) Текст сообщения для пользователя. Это то, что увидит пользователь.
        - "current_question_number": (integer) Текущий порядковый номер ЗАДАВАЕМОГО вопроса. Начинается с 1 для первого вопроса, 2 для второго и т.д. Если ты комментируешь ответ на вопрос N и затем задаешь вопрос N+1, current_question_number должен быть N+1.
        - "total_questions_in_test": (integer) Общее количество вопросов, которое ты планируешь задать в этом тесте. Должно быть установлено в первом вызове и не меняться.
        - "is_final_summary": (integer) Установи в 1, если это финальное сообщение с подведением итогов теста. В остальных случаях 0.

        Пример твоего ответа:
        {{
            "message_to_user": "Верно! Молодец.\\n\\nСледующий вопрос: Как называется столица Франции?",
            "current_question_number": 2,
            "total_questions_in_test": 3,
            "is_final_summary": 0
        }}

        Еще пример (финальное резюме):
        {{
            "message_to_user": "Тест завершен. Вы ответили правильно на 2 из 3 вопросов. Стоит повторить тему X.",
            "current_question_number": 3, 
            "total_questions_in_test": 3,
            "is_final_summary": 1
        }}
                
        - Поле 'total_questions_in_test' должно быть заполнено с первого же сообщения и оставаться консистентным.
        - Поле 'current_question_number' должно корректно инкрементироваться для каждого НОВОГО вопроса. 
        - Поле 'is_final_summary' должно быть 1 ТОЛЬКО для самого последнего сообщения, завершающего тест.
        
        Если это был последний вопрос: 
        - предоставь краткое резюме в 'message_to_user': На какие вопросы пользователь ответил верно, а какие темы стоит повторить.
        - Если были неточности в формулировках ответов пользователя, укажи на них.

        Веди диалог последовательно. Твой ответ должен быть ТОЛЬКО JSON объектом, без какого-либо другого текста до или после него.
        """

    async def analyze_image_and_start_test(self, image_base64: str, image_format: str = "jpeg") -> Tuple[Optional[Dict], List[Dict]]:
        """Анализ изображения и создание теста на основе его содержимого"""
        system_message_content = self._get_system_prompt_for_image_analysis()
        
        messages_for_api_call = [
            {"role": "system", "content": system_message_content},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Проанализируй это изображение и создай тест на основе его содержимого."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        parsed_data, updated_history = await self._make_openai_call(messages_for_api_call)
        return parsed_data, updated_history

    async def continue_image_test_session(self, history: List[Dict], user_message_text: str) -> Tuple[Optional[Dict], List[Dict]]:
        """Продолжение теста, начатого с изображения"""
        messages_for_api_call = list(history)
        messages_for_api_call.append({"role": "user", "content": user_message_text})
        
        parsed_data, updated_history = await self._make_openai_call(messages_for_api_call)
        return parsed_data, updated_history

    def _get_system_prompt_for_text_analysis(self) -> str:
        return """
        Ты — бот Zadavalnik помощник для повторения материала.
        Проводящишь интерактивное тестирование по теме: "{Тема, связанная с содержанием текстового документа}".

        Твоя задача — задавать вопросы пользователю один за другим.
        Тест должен состоять из нескольких вопросов (например, 3-5, определи это сам в первом сообщении).

        В начале первого вопроса сообщи "Сейчас мы проведем интерактивный тест по [тема теста]" и количество вопросов.

        Когда ты получил ответ пользователя на вопрос, ты должен:
        ## 1
        Если ответ пользователя хоть как-то связан с вопросом: 
            Дать на него краткий комментарий: (правильно/неправильно, дай короткое пояснение при необходимости).
        Если же пользователь отвечает не по теме вопроса или говорит, что не знает, забыл, простит сказать ответ.
            Просто дать ответ ответ на вопрос без комментариев.
        ## 2
        В этом же сообщении (через строку) пиши текст СЛЕДУЮЩЕГО вопроса.
        
        ВАЖНО: Ты ДОЛЖЕН форматировать КАЖДОЕ свое сообщение пользователю как JSON объект.
        Этот JSON объект должен содержать следующие поля:
        - "message_to_user": (string) Текст сообщения для пользователя. Это то, что увидит пользователь.
        - "current_question_number": (integer) Текущий порядковый номер ЗАДАВАЕМОГО вопроса. Начинается с 1 для первого вопроса, 2 для второго и т.д. Если ты комментируешь ответ на вопрос N и затем задаешь вопрос N+1, current_question_number должен быть N+1.
        - "total_questions_in_test": (integer) Общее количество вопросов, которое ты планируешь задать в этом тесте. Должно быть установлено в первом вызове и не меняться.
        - "is_final_summary": (integer) Установи в 1, если это финальное сообщение с подведением итогов теста. В остальных случаях 0.

        Пример твоего ответа:
        {{
            "message_to_user": "Верно! Молодец.\\n\\nСледующий вопрос: Как называется столица Франции?",
            "current_question_number": 2,
            "total_questions_in_test": 3,
            "is_final_summary": 0
        }}

        Еще пример (финальное резюме):
        {{
            "message_to_user": "Тест завершен. Вы ответили правильно на 2 из 3 вопросов. Стоит повторить тему X.",
            "current_question_number": 3, 
            "total_questions_in_test": 3,
            "is_final_summary": 1
        }}
                
        - Поле 'total_questions_in_test' должно быть заполнено с первого же сообщения и оставаться консистентным.
        - Поле 'current_question_number' должно корректно инкрементироваться для каждого НОВОГО вопроса. 
        - Поле 'is_final_summary' должно быть 1 ТОЛЬКО для самого последнего сообщения, завершающего тест.
        
        Если это был последний вопрос: 
        - предоставь краткое резюме в 'message_to_user': На какие вопросы пользователь ответил верно, а какие темы стоит повторить.
        - Если были неточности в формулировках ответов пользователя, укажи на них.

        Веди диалог последовательно. Твой ответ должен быть ТОЛЬКО JSON объектом, без какого-либо другого текста до или после него.
        """

    async def analyze_text_and_start_test(self, text_content: str) -> Tuple[Optional[Dict], List[Dict]]:
        """Анализ текстового документа и создание теста на основе его содержимого"""
        system_message_content = self._get_system_prompt_for_text_analysis()
        
        messages_for_api_call = [
            {"role": "system", "content": system_message_content},
            {
                "role": "user", 
                "content": f"Проанализируй этот текст и создай тест на основе его содержимого:\n\n{text_content}"
            }
        ]
        
        parsed_data, updated_history = await self._make_openai_call(messages_for_api_call)
        return parsed_data, updated_history

    async def continue_text_test_session(self, history: List[Dict], user_message_text: str) -> Tuple[Optional[Dict], List[Dict]]:
        """Продолжение теста, начатого с текстового документа"""
        messages_for_api_call = list(history)
        messages_for_api_call.append({"role": "user", "content": user_message_text})
        
        parsed_data, updated_history = await self._make_openai_call(messages_for_api_call)
        return parsed_data, updated_history