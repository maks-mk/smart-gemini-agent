import asyncio
import logging
import os
import time
import re
from functools import wraps
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

# Rich imports
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.columns import Columns
from rich.tree import Tree
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.status import Status
from rich import box
from rich.rule import Rule

#print("DEBUG cwd перед созданием конфига:", os.getcwd())

# ===== ЛОКАЛЬНЫЕ ИНСТРУМЕНТЫ =====
class DeleteFileInput(BaseModel):
    """Входные параметры для инструмента удаления файлов"""
    file_path: str = Field(description="Относительный путь к файлу для удаления")

class SafeDeleteFileTool(BaseTool):
    """Безопасный инструмент для удаления файлов только внутри рабочей директории"""
    name: str = "safe_delete_file"
    description: str = "Безопасно удаляет файл только внутри рабочей директории. Принимает относительный путь к файлу."
    args_schema: type[BaseModel] = DeleteFileInput
    working_directory: Path = None
    
    def __init__(self, working_directory: str, **kwargs):
        super().__init__(**kwargs)
        self.working_directory = Path(working_directory).resolve()
    
    def _run(self, file_path: str) -> str:
        """Выполняет удаление файла"""
        try:
            # Нормализуем путь и проверяем безопасность
            target_path = Path(self.working_directory) / file_path
            target_path = target_path.resolve()
            
            # Проверяем, что файл находится внутри рабочей директории
            if not str(target_path).startswith(str(self.working_directory)):
                return f"❌ ОШИБКА: Попытка удаления файла вне рабочей директории. Файл: {target_path}"
            
            # Проверяем существование файла
            if not target_path.exists():
                return f"❌ ОШИБКА: Файл не найден: {file_path}"
            
            # Проверяем, что это файл, а не директория
            if target_path.is_dir():
                return f"❌ ОШИБКА: {file_path} является директорией. Используйте инструмент для удаления директорий."
            
            # Удаляем файл
            target_path.unlink()
            
            return f"✅ УСПЕХ: Файл {file_path} успешно удален"
            
        except PermissionError:
            return f"❌ ОШИБКА: Нет прав для удаления файла {file_path}"
        except Exception as e:
            return f"❌ ОШИБКА: Не удалось удалить файл {file_path}. Причина: {str(e)}"
    
    async def _arun(self, file_path: str) -> str:
        """Асинхронная версия удаления файла"""
        return self._run(file_path)

class DeleteDirectoryInput(BaseModel):
    """Входные параметры для инструмента удаления директорий"""
    dir_path: str = Field(description="Относительный путь к директории для удаления")
    recursive: bool = Field(default=False, description="Удалить директорию рекурсивно (со всем содержимым)")

class SafeDeleteDirectoryTool(BaseTool):
    """Безопасный инструмент для удаления директорий только внутри рабочей директории"""
    name: str = "safe_delete_directory"
    description: str = "Безопасно удаляет директорию только внутри рабочей директории. Может удалять рекурсивно."
    args_schema: type[BaseModel] = DeleteDirectoryInput
    working_directory: Path = None
    
    def __init__(self, working_directory: str, **kwargs):
        super().__init__(**kwargs)
        self.working_directory = Path(working_directory).resolve()
    
    def _run(self, dir_path: str, recursive: bool = False) -> str:
        """Выполняет удаление директории"""
        try:
            # Нормализуем путь и проверяем безопасность
            target_path = Path(self.working_directory) / dir_path
            target_path = target_path.resolve()
            
            # Проверяем, что директория находится внутри рабочей директории
            if not str(target_path).startswith(str(self.working_directory)):
                return f"❌ ОШИБКА: Попытка удаления директории вне рабочей директории. Директория: {target_path}"
            
            # Проверяем существование директории
            if not target_path.exists():
                return f"❌ ОШИБКА: Директория не найдена: {dir_path}"
            
            # Проверяем, что это директория, а не файл
            if not target_path.is_dir():
                return f"❌ ОШИБКА: {dir_path} является файлом. Используйте инструмент для удаления файлов."
            
            # Удаляем директорию
            if recursive:
                import shutil
                shutil.rmtree(target_path)
                return f"✅ УСПЕХ: Директория {dir_path} и все её содержимое успешно удалены"
            else:
                target_path.rmdir()  # Удаляет только пустую директорию
                return f"✅ УСПЕХ: Пустая директория {dir_path} успешно удалена"
            
        except OSError as e:
            if "Directory not empty" in str(e):
                return f"❌ ОШИБКА: Директория {dir_path} не пуста. Используйте recursive=true для рекурсивного удаления."
            return f"❌ ОШИБКА: Не удалось удалить директорию {dir_path}. Причина: {str(e)}"
        except Exception as e:
            return f"❌ ОШИБКА: Не удалось удалить директорию {dir_path}. Причина: {str(e)}"
    
    async def _arun(self, dir_path: str, recursive: bool = False) -> str:
        """Асинхронная версия удаления директории"""
        return self._run(dir_path, recursive)

# ===== НАСТРОЙКА ЛОГИРОВАНИЯ С ПОДАВЛЕНИЕМ ПРЕДУПРЕЖДЕНИЙ =====
class IgnoreSchemaWarnings(logging.Filter):
    def filter(self, record):
        ignore_messages = [
            "Key 'additionalProperties' is not supported in schema, ignoring",
            "Key '$schema' is not supported in schema, ignoring"
        ]
        return not any(msg in record.getMessage() for msg in ignore_messages)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_agent.log', encoding='utf-8')
    ]
)

# Применить фильтр ко всем обработчикам
for handler in logging.root.handlers:
    handler.addFilter(IgnoreSchemaWarnings())

# Дополнительно подавить логгеры MCP компонентов
mcp_loggers = [
    'langchain_mcp_adapters',
    'mcp',
    'jsonschema'
]
for logger_name in mcp_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# ===== ДЕКОРАТОРЫ =====
def retry_on_failure(max_retries: int = 2, delay: float = 1.0):
    """Декоратор для повторения операций при неудаче"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Попытка {attempt + 1} неудачна, повтор через {delay}с")
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


# ===== УПРОЩЕННАЯ КОНФИГУРАЦИЯ =====
@dataclass
class AgentConfig:
    """Упрощенная конфигурация AI-агента для Gemini"""
    filesystem_path: str = None  # По умолчанию None
    use_memory: bool = True
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    
    def __post_init__(self):
        """Автоматическая установка рабочей директории при инициализации"""
        if self.filesystem_path is None:
            self.filesystem_path = os.getcwd()
            logger.info(f"Рабочая директория не указана, используется текущая: {self.filesystem_path}")
        
        # Нормализация пути (добавление завершающего слеша)
        if not self.filesystem_path.endswith(os.sep):
            self.filesystem_path += os.sep
    
    def validate(self) -> None:
        """Простая валидация"""
        if not os.path.exists(self.filesystem_path):
            raise ValueError(f"Путь не существует: {self.filesystem_path}")
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("Отсутствует переменная окружения: GOOGLE_API_KEY")
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Конфигурация MCP сервера"""
        return {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", self.filesystem_path],
                "transport": "stdio"
            },
            "duckduckgo": {
                "command": "uvx",
                "args": ["duckduckgo-mcp-server"],
                "transport": "stdio"
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "transport": "stdio",
            }
        }


# ===== ОСНОВНОЙ КЛАСС АГЕНТА =====
class FileSystemAgent:
    """
    Умный AI-агент для работы с файловой системой (только Gemini)
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = None
        self.checkpointer = None
        self.mcp_client = None
        self.tools = []
        self.tools_map = {}  # Карта инструментов по функциональности
        self.context_memory = {}  # Контекстная память о файловой системе
        self._initialized = False
        
        logger.info("Создан умный агент с Gemini")
        logger.info(f"Рабочая директория: {config.filesystem_path}")
    
    @property
    def is_ready(self) -> bool:
        """Проверяет готовность агента"""
        return self._initialized and self.agent is not None
    
    async def initialize(self) -> bool:
        """Инициализация агента"""
        if self._initialized:
            logger.warning("Агент уже инициализирован")
            return True
        
        logger.info("Инициализация агента...")
        
        try:
            self.config.validate()
            await self._init_mcp_client()
            
            # Создание Gemini модели
            api_key = os.getenv("GOOGLE_API_KEY")
            model = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                google_api_key=api_key,
                temperature=self.config.temperature
            )
            
            if self.config.use_memory:
                self.checkpointer = InMemorySaver()
                logger.info("Память агента включена")
            
            self.agent = create_react_agent(
                model=model,
                tools=self.tools,
                checkpointer=self.checkpointer,
                prompt=self._get_system_prompt()
            )
            
            self._initialized = True
            logger.info("✅ Агент успешно инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    @retry_on_failure()
    async def _init_mcp_client(self):
        """Инициализация MCP клиента"""
        logger.info("Инициализация MCP клиента...")
        
        # Временно подавить предупреждения во время инициализации
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            self.mcp_client = MultiServerMCPClient(self.config.get_mcp_config())
            self.tools = await self.mcp_client.get_tools()
        finally:
            # Восстановить уровень логирования
            logging.getLogger().setLevel(old_level)
        
        if not self.tools:
            raise Exception("Нет доступных MCP инструментов")
        
        # Добавляем локальные инструменты для удаления
        self._add_local_tools()
        
        # Анализируем и категоризируем инструменты
        self._analyze_tools()
        
        logger.info(f"Загружено {len(self.tools)} инструментов")
        for tool in self.tools:
            logger.info(f"  • {tool.name}")
    
    def _add_local_tools(self):
        """Добавление локальных инструментов"""
        # Создаем локальные инструменты для удаления
        delete_file_tool = SafeDeleteFileTool(self.config.filesystem_path)
        delete_dir_tool = SafeDeleteDirectoryTool(self.config.filesystem_path)
        
        # Добавляем к списку инструментов
        self.tools.extend([delete_file_tool, delete_dir_tool])
        
        logger.info("Добавлены локальные инструменты:")
        logger.info(f"  • {delete_file_tool.name}: {delete_file_tool.description}")
        logger.info(f"  • {delete_dir_tool.name}: {delete_dir_tool.description}")
    
    def _analyze_tools(self):
        """Анализ и категоризация доступных инструментов"""
        self.tools_map = {
            'read_file': [],
            'write_file': [],
            'list_directory': [],
            'create_directory': [],
            'delete_file': [],
            'move_file': [],
            'search': [],
            'web_search': [],
            'fetch_url': [],
            'other': []
        }
        
        for tool in self.tools:
            name = tool.name.lower()
            
            # Анализируем по названию и описанию
            if any(keyword in name for keyword in ['read', 'get', 'cat', 'show', 'view']):
                if 'directory' in name or 'dir' in name or 'list' in name:
                    self.tools_map['list_directory'].append(tool)
                else:
                    self.tools_map['read_file'].append(tool)
            elif any(keyword in name for keyword in ['write', 'create', 'save', 'put']):
                if 'directory' in name or 'dir' in name or 'folder' in name:
                    self.tools_map['create_directory'].append(tool)
                else:
                    self.tools_map['write_file'].append(tool)
            elif any(keyword in name for keyword in ['delete', 'remove', 'rm', 'unlink']):
                self.tools_map['delete_file'].append(tool)
            elif any(keyword in name for keyword in ['shell', 'exec', 'run', 'command']):
                # Shell команды могут использоваться для удаления
                self.tools_map['delete_file'].append(tool)
            elif name in ['safe_delete_file', 'safe_delete_directory']:
                # Наши локальные инструменты удаления
                self.tools_map['delete_file'].append(tool)
            elif any(keyword in name for keyword in ['move', 'rename', 'mv']):
                self.tools_map['move_file'].append(tool)
            elif any(keyword in name for keyword in ['search', 'find', 'grep']):
                self.tools_map['search'].append(tool)
            elif any(keyword in name for keyword in ['web', 'duckduckgo', 'google']):
                self.tools_map['web_search'].append(tool)
            elif any(keyword in name for keyword in ['fetch', 'download', 'get_url', 'http']):
                self.tools_map['fetch_url'].append(tool)
            else:
                self.tools_map['other'].append(tool)
        
        logger.info("Карта инструментов создана:")
        for category, tools in self.tools_map.items():
            if tools:
                logger.info(f"  {category}: {[t.name for t in tools]}")
    
    def _analyze_user_intent(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Анализ намерений пользователя и извлечение параметров"""
        user_input_lower = user_input.lower().strip()
        
        # Проверяем контекстные ссылки на предыдущие варианты
        if self._is_context_reference(user_input_lower):
            return self._handle_context_reference(user_input_lower)
        
        # Паттерны для различных операций
        patterns = {
            'create_file': [
                r'создай файл\s+([^\s]+)',
                r'создать файл\s+([^\s]+)',
                r'сделай файл\s+([^\s]+)',
                r'новый файл\s+([^\s]+)',
                r'create file\s+([^\s]+)',
                r'make file\s+([^\s]+)'
            ],
            'create_directory': [
                r'создай папку\s+([^\s]+)',
                r'создать папку\s+([^\s]+)',
                r'создай директорию\s+([^\s]+)',
                r'новая папка\s+([^\s]+)',
                r'create folder\s+([^\s]+)',
                r'make directory\s+([^\s]+)',
                r'mkdir\s+([^\s]+)'
            ],
            'read_file': [
                r'читай\s+([^\s]+)',
                r'прочитай\s+([^\s]+)',
                r'покажи содержимое\s+([^\s]+)',
                r'открой\s+([^\s]+)',
                r'read\s+([^\s]+)',
                r'show\s+([^\s]+)',
                r'cat\s+([^\s]+)'
            ],
            'list_directory': [
                r'покажи файлы',
                r'список файлов',
                r'что в папке',
                r'содержимое папки\s*([^\s]*)',
                r'ls\s*([^\s]*)',
                r'dir\s*([^\s]*)',
                r'list files'
            ],
            'delete_file': [
                r'удали\s+файл\s+(.+)',
                r'удали\s+(.+)',
                r'удалить\s+файл\s+(.+)',
                r'удалить\s+(.+)',
                r'убери\s+файл\s+(.+)',
                r'убери\s+(.+)',
                r'delete\s+file\s+(.+)',
                r'delete\s+(.+)',
                r'remove\s+file\s+(.+)',
                r'remove\s+(.+)',
                r'rm\s+(.+)'
            ],
            'search': [
                r'найди\s+(.+)',
                r'поиск\s+(.+)',
                r'ищи\s+(.+)',
                r'search\s+(.+)',
                r'find\s+(.+)'
            ],
            'web_search': [
                r'найди в интернете\s+(.+)',
                r'поиск в сети\s+(.+)',
                r'гугли\s+(.+)',
                r'web search\s+(.+)',
                r'google\s+(.+)'
            ]
        }
        
        # Поиск совпадений
        for intent, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, user_input_lower)
                if match:
                    params = {'target': match.group(1) if match.groups() else None}
                    
                    # Дополнительный анализ для извлечения контента
                    if intent == 'create_file':
                        content_match = re.search(r'с содержимым\s+(.+)|с текстом\s+(.+)|with content\s+(.+)', user_input_lower)
                        if content_match:
                            params['content'] = content_match.group(1) or content_match.group(2) or content_match.group(3)
                    
                    return intent, params
        
        # Если точное намерение не найдено, попробуем определить по ключевым словам
        if any(word in user_input_lower for word in ['файл', 'file']) and any(word in user_input_lower for word in ['создай', 'create', 'новый']):
            return 'create_file', {'target': None}
        elif any(word in user_input_lower for word in ['папка', 'folder', 'директория', 'directory']) and any(word in user_input_lower for word in ['создай', 'create', 'новая']):
            return 'create_directory', {'target': None}
        elif any(word in user_input_lower for word in ['читай', 'read', 'покажи', 'show', 'открой']):
            return 'read_file', {'target': None}
        elif any(word in user_input_lower for word in ['удали', 'delete', 'убери', 'remove']):
            return 'delete_file', {'target': None}
        elif any(word in user_input_lower for word in ['список', 'файлы', 'содержимое', 'ls', 'dir']):
            return 'list_directory', {'target': None}
        
        return 'general', {}
    
    def _is_context_reference(self, user_input: str) -> bool:
        """Проверяет, является ли ввод ссылкой на предыдущий контекст"""
        # Числовые ссылки на варианты (1, 2, 3, 4)
        if user_input in ['1', '2', '3', '4', '5']:
            return True
        
        # Ключевые слова для ссылок на контекст (только короткие фразы)
        context_keywords = [
            'первый', 'второй', 'третий', 'четвертый', 'пятый',
            'первый вариант', 'второй вариант', 'третий вариант',
            'да', 'давай', 'сделай это', 'выполни'
        ]
        
        # Проверяем только если это короткая фраза (не более 2 слов) и не содержит имена файлов
        if len(user_input.split()) <= 2 and not any(ext in user_input for ext in ['.', 'файл']):
            return any(keyword in user_input for keyword in context_keywords)
        
        # Специальные случаи для переименования только если есть контекст удаления
        if len(user_input.split()) <= 2 and any(word in user_input for word in ['переименуй']):
            return self.context_memory.get('last_intent') == 'delete_file'
        
        return False
    
    def _handle_context_reference(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Обрабатывает ссылки на предыдущий контекст"""
        last_intent = self.context_memory.get('last_intent')
        last_params = self.context_memory.get('last_params', {})
        last_suggestions = self.context_memory.get('last_suggestions', [])
        
        # Если это числовая ссылка на вариант
        if user_input in ['1', '2', '3', '4', '5']:
            option_num = int(user_input) - 1
            
            if last_intent == 'delete_file' and last_suggestions:
                target_file = last_params.get('target')
                
                if option_num == 0:  # Переименовать файл
                    return 'move_file', {
                        'target': target_file,
                        'action': 'rename_to_backup',
                        'context_action': 'rename_for_deletion'
                    }
                elif option_num == 1:  # Переместить в папку для удаления
                    return 'move_file', {
                        'target': target_file,
                        'action': 'move_to_delete_folder',
                        'context_action': 'move_for_deletion'
                    }
                elif option_num == 2:  # Очистить содержимое
                    return 'write_file', {
                        'target': target_file,
                        'content': '',
                        'context_action': 'clear_content'
                    }
        
        # Если это текстовая ссылка на переименование
        if any(word in user_input for word in ['переименуй', 'rename']):
            if last_intent == 'delete_file':
                return 'move_file', {
                    'target': last_params.get('target'),
                    'action': 'rename_to_backup',
                    'context_action': 'rename_for_deletion'
                }
        
        # Если это общая ссылка на выполнение действия
        if any(word in user_input for word in ['да', 'давай', 'сделай', 'выполни']):
            if last_intent and last_params:
                return last_intent, last_params
        
        return 'general', {'context_reference': True, 'original_input': user_input}
    
    def _get_smart_system_prompt(self) -> str:
        """Умный системный промпт с детальным описанием инструментов"""
        tools_description = self._generate_tools_description()
        
        return f"""Ты умный AI-ассистент для работы с файловой системой и веб-поиском.

РАБОЧАЯ ДИРЕКТОРИЯ: {self.config.filesystem_path}
Все файловые операции выполняются относительно этой директории.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_description}

ПРАВИЛА РАБОТЫ:
1. Анализируй запрос пользователя и определяй нужную операцию
2. Используй ТОЛЬКО предоставленные инструменты
3. Всегда указывай полные пути для файлов и папок
4. При создании файлов с содержимым используй соответствующие параметры
5. Если путь не указан явно, работай в текущей рабочей директории
6. Для поиска в интернете используй web-инструменты
7. Для поиска файлов используй файловые инструменты поиска
8. УДАЛЕНИЕ ФАЙЛОВ: Используй safe_delete_file для файлов и safe_delete_directory для папок
9. Эти инструменты безопасно удаляют только внутри рабочей директории
10. НЕ ВЫДУМЫВАЙ несуществующие инструменты или команды

ПРИМЕРЫ ОБРАБОТКИ ЗАПРОСОВ:
- "создай файл test.txt с текстом Hello" → используй инструмент записи файла
- "покажи содержимое config.py" → используй инструмент чтения файла  
- "удали старый файл backup.txt" → используй safe_delete_file с параметром file_path="backup.txt"
- "удали папку temp" → используй safe_delete_directory с параметрами dir_path="temp", recursive=true
- "найди файлы с расширением .py" → используй инструмент поиска файлов
- "найди в интернете информацию о Python" → используй веб-поиск

ФОРМАТ ОТВЕТА:
- Кратко подтверди выполненное действие
- При ошибках объясни причину и предложи решение
- Для сложных операций опиши что делаешь пошагово"""
    
    def _generate_tools_description(self) -> str:
        """Генерация детального описания инструментов"""
        descriptions = []
        
        for category, tools in self.tools_map.items():
            category_desc = {
                'read_file': 'ЧТЕНИЕ ФАЙЛОВ',
                'write_file': 'СОЗДАНИЕ/ЗАПИСЬ ФАЙЛОВ', 
                'list_directory': 'ПРОСМОТР ДИРЕКТОРИЙ',
                'create_directory': 'СОЗДАНИЕ ПАПОК',
                'delete_file': 'УДАЛЕНИЕ ФАЙЛОВ/ПАПОК',
                'move_file': 'ПЕРЕМЕЩЕНИЕ/ПЕРЕИМЕНОВАНИЕ',
                'search': 'ПОИСК ФАЙЛОВ',
                'web_search': 'ВЕБ-ПОИСК',
                'fetch_url': 'ЗАГРУЗКА ИЗ ИНТЕРНЕТА',
                'other': 'ДРУГИЕ ИНСТРУМЕНТЫ'
            }.get(category, category.upper())
            
            descriptions.append(f"\n{category_desc}:")
            
            if not tools:
                descriptions.append("  • Нет доступных инструментов в этой категории")
            else:
                for tool in tools:
                    tool_desc = self._get_tool_description(tool)
                    descriptions.append(f"  • {tool.name}: {tool_desc}")
        
        return '\n'.join(descriptions)
    
    def _get_tool_description(self, tool) -> str:
        """Получение описания инструмента"""
        if hasattr(tool, 'description') and tool.description:
            return tool.description[:100] + ('...' if len(tool.description) > 100 else '')
        
        # Генерируем описание на основе названия
        name = tool.name.lower()
        if 'read' in name:
            return "Читает содержимое файла"
        elif 'write' in name:
            return "Создает или записывает файл"
        elif 'list' in name:
            return "Показывает содержимое директории"
        elif 'create' in name and 'dir' in name:
            return "Создает новую папку"
        elif 'delete' in name or 'remove' in name:
            return "Удаляет файл или папку"
        elif 'shell' in name or 'exec' in name or 'run' in name or 'command' in name:
            return "Выполняет shell команды (можно использовать для удаления файлов)"
        elif name == 'safe_delete_file':
            return "Безопасно удаляет файлы только внутри рабочей директории"
        elif name == 'safe_delete_directory':
            return "Безопасно удаляет директории только внутри рабочей директории"
        elif 'move' in name:
            return "Перемещает или переименовывает файл"
        elif 'search' in name:
            return "Ищет файлы по критериям"
        elif 'web' in name or 'duckduckgo' in name:
            return "Поиск информации в интернете"
        elif 'fetch' in name:
            return "Загружает данные по URL"
        else:
            return "Специальный инструмент"
    
    def _get_system_prompt(self) -> str:
        """Системный промпт"""
        return self._get_smart_system_prompt()

    
    @retry_on_failure()
    async def process_message(self, user_input: str, thread_id: str = "default") -> str:
        """Умная обработка сообщения пользователя с анализом намерений"""
        if not self.is_ready:
            return "❌ Агент не готов. Попробуйте переинициализировать."
        
        try:
            # Анализируем намерения пользователя
            intent, params = self._analyze_user_intent(user_input)
            logger.info(f"Определено намерение: {intent}, параметры: {params}")
            
            # Создаем улучшенный контекст на основе анализа
            enhanced_input = self._create_enhanced_context(user_input, intent, params)
            
            config = {"configurable": {"thread_id": thread_id}}
            message_input = {"messages": [HumanMessage(content=enhanced_input)]}
            
            response = await self.agent.ainvoke(message_input, config)
            
            # Обновляем контекстную память
            self._update_context_memory(intent, params, response)
            
            # ИСПРАВЛЕНИЕ: правильно извлекаем последнее сообщение
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                if messages:
                    last_message = messages[-1]
                    # Извлекаем содержимое в зависимости от типа сообщения
                    if hasattr(last_message, 'content'):
                        return str(last_message.content)
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        return str(last_message['content'])
                    else:
                        return str(last_message)
                else:
                    return "❌ Получен пустой ответ от агента"
            else:
                # Если response не в ожидаемом формате
                return str(response)
            
        except Exception as e:
            error_msg = f"❌ Ошибка обработки: {e}"
            logger.error(error_msg)
            logger.error(f"Тип ответа: {type(response)}")
            if 'response' in locals():
                logger.error(f"Содержимое ответа: {response}")
            return error_msg
    
    def _create_enhanced_context(self, user_input: str, intent: str, params: Dict[str, Any]) -> str:
        """Создание улучшенного контекста на основе анализа намерений"""
        base_context = f"Рабочая директория: '{self.config.filesystem_path}'"
        
        # Добавляем специфичные инструкции на основе намерения
        intent_instructions = {
            'create_file': f"ЗАДАЧА: Создать файл. Рекомендуемые инструменты: {[t.name for t in self.tools_map.get('write_file', [])]}",
            'create_directory': f"ЗАДАЧА: Создать папку. Рекомендуемые инструменты: {[t.name for t in self.tools_map.get('create_directory', [])]}",
            'read_file': f"ЗАДАЧА: Прочитать файл. Рекомендуемые инструменты: {[t.name for t in self.tools_map.get('read_file', [])]}",
            'list_directory': f"ЗАДАЧА: Показать содержимое папки. Рекомендуемые инструменты: {[t.name for t in self.tools_map.get('list_directory', [])]}",
            'delete_file': self._get_delete_instruction(),
            'move_file': self._get_move_instruction(params),
            'search': f"ЗАДАЧА: Поиск файлов. Рекомендуемые инструменты: {[t.name for t in self.tools_map.get('search', [])]}",
            'web_search': f"ЗАДАЧА: Поиск в интернете. Рекомендуемые инструменты: {[t.name for t in self.tools_map.get('web_search', [])]}"
        }
        
        instruction = intent_instructions.get(intent, "ЗАДАЧА: Общий запрос")
        
        # Добавляем контекстную информацию из памяти
        context_info = ""
        if self.context_memory:
            recent_files = self.context_memory.get('recent_files', [])
            if recent_files:
                context_info = f"\nНедавно работали с файлами: {recent_files[-5:]}"
        
        # Формируем итоговый контекст
        enhanced_context = f"""{base_context}
{instruction}

ПАРАМЕТРЫ ЗАПРОСА:
- Цель: {params.get('target', 'не указана')}
- Содержимое: {params.get('content', 'не указано')}
{context_info}

ОРИГИНАЛЬНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_input}

ИНСТРУКЦИЯ: Выполни запрос максимально точно, используя подходящие инструменты."""
        
        return enhanced_context
    
    def _get_delete_instruction(self) -> str:
        """Получение инструкций для удаления файлов"""
        delete_tools = self.tools_map.get('delete_file', [])
        
        if delete_tools:
            tool_names = [t.name for t in delete_tools]
            if 'safe_delete_file' in tool_names or 'safe_delete_directory' in tool_names:
                return f"""ЗАДАЧА: Удалить файл/папку. 
ДОСТУПНЫЕ ИНСТРУМЕНТЫ: {tool_names}
ИНСТРУКЦИЯ: Используй safe_delete_file для файлов и safe_delete_directory для папок.
Эти инструменты безопасно удаляют только внутри рабочей директории."""
            else:
                return f"ЗАДАЧА: Удалить файл/папку. Рекомендуемые инструменты: {tool_names}"
        else:
            return """ЗАДАЧА: Удалить файл/папку.
ВАЖНО: У меня НЕТ инструментов для удаления файлов!
ИНСТРУКЦИЯ: Честно сообщи пользователю, что удаление файлов недоступно.
Предложи альтернативы:
1. Переименовать файл в .backup
2. Переместить в папку для удаления
3. Очистить содержимое файла (записать пустую строку)
4. Использовать внешние инструменты операционной системы"""
    
    def _get_move_instruction(self, params: Dict[str, Any]) -> str:
        """Получение инструкций для перемещения/переименования файлов"""
        move_tools = self.tools_map.get('move_file', [])
        context_action = params.get('context_action', '')
        
        if context_action == 'rename_for_deletion':
            return f"""ЗАДАЧА: Переименовать файл для пометки на удаление.
КОНТЕКСТ: Пользователь выбрал вариант 1 из предложенных альтернатив удаления.
ИНСТРУКЦИЯ: Переименуй файл {params.get('target', '')} добавив расширение .backup
Рекомендуемые инструменты: {[t.name for t in move_tools]}"""
        
        elif context_action == 'move_for_deletion':
            return f"""ЗАДАЧА: Переместить файл в папку для удаления.
КОНТЕКСТ: Пользователь выбрал вариант 2 из предложенных альтернатив удаления.
ИНСТРУКЦИЯ: 
1. Создай папку 'to_delete' если её нет
2. Перемести файл {params.get('target', '')} в эту папку
Рекомендуемые инструменты: {[t.name for t in move_tools]}"""
        
        else:
            return f"ЗАДАЧА: Переместить/переименовать файл. Рекомендуемые инструменты: {[t.name for t in move_tools]}"
    
    def _update_context_memory(self, intent: str, params: Dict[str, Any], response: Any):
        """Обновление контекстной памяти агента"""
        if 'recent_files' not in self.context_memory:
            self.context_memory['recent_files'] = []
        
        # Запоминаем файлы, с которыми работали
        target = params.get('target')
        if target and intent in ['create_file', 'read_file', 'delete_file', 'move_file']:
            if target not in self.context_memory['recent_files']:
                self.context_memory['recent_files'].append(target)
                # Ограничиваем размер памяти
                if len(self.context_memory['recent_files']) > 20:
                    self.context_memory['recent_files'] = self.context_memory['recent_files'][-20:]
        
        # Запоминаем последнее намерение и параметры
        self.context_memory['last_intent'] = intent
        self.context_memory['last_params'] = params
        
        # Запоминаем предложения для запросов на удаление
        if intent == 'delete_file':
            self.context_memory['last_suggestions'] = [
                'rename_to_backup',
                'move_to_delete_folder', 
                'clear_content',
                'use_external_tools'
            ]
        
        # Очищаем предложения для других типов запросов
        elif intent != 'general':
            self.context_memory.pop('last_suggestions', None)
    
    def get_status(self) -> Dict[str, Any]:
        """Информация о состоянии умного агента"""
        tools_by_category = {k: len(v) for k, v in self.tools_map.items() if v}
        
        return {
            "initialized": self._initialized,
            "model": "Smart Gemini Agent",
            "model_name": self.config.model_name,
            "filesystem_path": self.config.filesystem_path,
            "memory_enabled": self.config.use_memory,
            "tools_count": len(self.tools),
            "tools_by_category": tools_by_category,
            "context_memory_items": len(self.context_memory),
            "recent_files_count": len(self.context_memory.get('recent_files', [])),
            "intelligence_features": [
                "Intent Analysis",
                "Smart Tool Selection", 
                "Context Memory",
                "Enhanced Prompting"
            ]
        }


# ===== БОГАТЫЙ ТЕРМИНАЛЬНЫЙ ИНТЕРФЕЙС =====
class RichInteractiveChat:
    """Богатый терминальный интерфейс для AI-агента"""
    
    def __init__(self, agent):
        self.console = Console()
        self.agent = agent
        self.history = []
        self.current_thread = "main"
        self.show_timestamps = True
        self.theme = "dark"
        
        # Стили
        self.styles = {
            "user": "bold blue",
            "agent": "green",
            "system": "yellow",
            "error": "bold red",
            "success": "bold green",
            "info": "cyan",
            "warning": "orange3",
            "path": "bold magenta",
            "command": "bold white on blue"
        }
    
    def clear_screen(self):
        """Очистка экрана"""
        self.console.clear()
    
    def print_header(self):
        """Отображение заголовка приложения"""
        header_text = Text("� GSmart Gemini FileSystem Agent", style="bold white")
        subtitle = Text("Intelligent file operations with intent analysis", style="dim italic")
        
        header_content = Align.center(
            Text.assemble(header_text, "\n", subtitle)
        )
        
        header_panel = Panel(
            header_content,
            box=box.DOUBLE,
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(header_panel)
        self.console.print()
    
    def print_status_bar(self):
        """Статус-бар с информацией о системе"""
        if not self.agent:
            return
            
        status = self.agent.get_status()
        
        status_items = [
            f"🔧 [bold]{status.get('model', 'Unknown')}[/bold]",
            f"📁 [bold magenta]{os.path.basename(status.get('filesystem_path', ''))}/[/bold magenta]",
            f"🧠 [{'green' if status.get('memory_enabled') else 'red'}]Memory[/]",
            f"🔧 {status.get('tools_count', 0)} tools",
            f"💬 Thread: [bold]{self.current_thread}[/bold]"
        ]
        
        status_table = Table.grid(padding=1)
        for item in status_items:
            status_table.add_column()
        
        status_table.add_row(*status_items)
        
        status_panel = Panel(
            status_table,
            title="[bold]System Status[/bold]",
            border_style="dim",
            height=3
        )
        
        self.console.print(status_panel)
    
    def display_file_tree(self, path: str, max_depth: int = 3, show_hidden: bool = False):
        """Красивое отображение файловой структуры"""
        def add_tree_items(tree_node, current_path, current_depth: int = 0):
            if current_depth >= max_depth:
                return
            
            try:
                current_path = Path(current_path)
                items = sorted(current_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                
                for item in items:
                    if not show_hidden and item.name.startswith('.'):
                        continue
                    
                    if item.is_dir():
                        # Директория
                        emoji = "📁"
                        try:
                            if any(item.iterdir()):
                                emoji = "📁"
                            else:
                                emoji = "📂"
                        except:
                            emoji = "📁"
                        
                        dir_node = tree_node.add(f"{emoji} [bold blue]{item.name}/[/bold blue]")
                        if current_depth < max_depth - 1:
                            add_tree_items(dir_node, item, current_depth + 1)
                    else:
                        # Файл
                        try:
                            size = item.stat().st_size
                            size_str = self._format_file_size(size)
                            
                            # Эмодзи и цвет по расширению
                            emoji = self._get_file_emoji(item.suffix.lower())
                            color = self._get_file_color(item.suffix.lower())
                            
                            tree_node.add(f"{emoji} [{color}]{item.name}[/{color}] [dim]({size_str})[/dim]")
                        except:
                            tree_node.add(f"📄 {item.name}")
                        
            except PermissionError:
                tree_node.add("❌ [red]Permission Denied[/red]")
            except Exception as e:
                tree_node.add(f"❌ [red]Error: {str(e)}[/red]")
        
        tree = Tree(f"📁 [bold blue]{Path(path).name}/[/bold blue]")
        add_tree_items(tree, path)
        
        panel = Panel(
            tree,
            title=f"[bold]File Tree: {path}[/bold]",
            border_style="blue",
            expand=False
        )
        
        self.console.print(panel)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Форматирование размера файла"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    def _get_file_emoji(self, extension: str) -> str:
        """Получение эмодзи для файла по расширению"""
        emoji_map = {
            '.py': '🐍', '.js': '🟨', '.html': '🌐', '.css': '🎨',
            '.json': '📋', '.xml': '📄', '.yml': '⚙️', '.yaml': '⚙️',
            '.md': '📝', '.txt': '📄', '.pdf': '📕', '.doc': '📘',
            '.xlsx': '📊', '.csv': '📈', '.zip': '🗜️', '.tar': '📦',
            '.jpg': '🖼️', '.png': '🖼️', '.gif': '🎞️', '.mp4': '🎬',
            '.mp3': '🎵', '.wav': '🎵', '.exe': '⚙️', '.dll': '🔧',
            '.log': '📋', '.sql': '🗃️', '.db': '🗃️', '.sqlite': '🗃️'
        }
        return emoji_map.get(extension, '📄')
    
    def _get_file_color(self, extension: str) -> str:
        """Получение цвета для файла по расширению"""
        color_map = {
            '.py': 'yellow', '.js': 'bright_yellow', '.html': 'bright_blue',
            '.css': 'magenta', '.json': 'cyan', '.xml': 'green',
            '.md': 'bright_white', '.txt': 'white', '.log': 'dim white',
            '.jpg': 'bright_magenta', '.png': 'bright_magenta',
            '.zip': 'red', '.exe': 'bright_red'
        }
        return color_map.get(extension, 'white')
    
    def display_help(self):
        """Отображение справки по командам"""
        commands = {
            "Файловые операции": [
                ("создай файл <имя>", "Создать новый файл"),
                ("создай папку <имя>", "Создать новую папку"),
                ("покажи файлы", "Показать структуру файлов"),
                ("читай <файл>", "Прочитать содержимое файла"),
                ("удали <файл>", "✅ Удалить файл (безопасно)"),
                ("удали папку <имя>", "✅ Удалить папку (безопасно)"),
                ("переименуй <старое> <новое>", "Переименовать файл"),
                ("найди <паттерн>", "Найти файлы по паттерну"),
            ],
            "Веб-операции": [
                ("найди в интернете <запрос>", "Поиск информации в интернете"),
                ("скачай <URL>", "Загрузить контент по URL"),
                ("поиск <запрос>", "Веб-поиск через DuckDuckGo"),
            ],
            "Системные команды": [
                ("/help", "Показать эту справку"),
                ("/clear", "Очистить экран"),
                ("/status", "Показать статус системы"),
                ("/tools", "Показать доступные инструменты MCP"),
                ("/tree [путь]", "Показать дерево файлов"),
                ("/history", "Показать историю команд"),
                ("/thread <имя>", "Переключить контекст"),
                ("/quit", "Выйти из программы"),
            ],
            "Настройки": [
                ("/theme <light|dark>", "Сменить тему"),
                ("/timestamps <on|off>", "Включить/выключить временные метки"),
                ("/export", "Экспорт истории диалога"),
            ]
        }
        
        help_panels = []
        for category, cmd_list in commands.items():
            table = Table(show_header=False, box=None, pad_edge=False)
            table.add_column("Command", style="bold cyan", no_wrap=True)
            table.add_column("Description", style="dim")
            
            for cmd, desc in cmd_list:
                table.add_row(cmd, desc)
            
            panel = Panel(
                table,
                title=f"[bold]{category}[/bold]",
                border_style="blue"
            )
            help_panels.append(panel)
        
        self.console.print(Columns(help_panels, equal=True, expand=True))
    
    def display_history(self, limit: int = 10):
        """Отображение истории команд"""
        if not self.history:
            self.console.print("[dim]История пуста[/dim]")
            return
        
        table = Table(
            "Time", "Type", "Message",
            title="[bold]Command History[/bold]",
            show_lines=True
        )
        
        recent_history = self.history[-limit:] if len(self.history) > limit else self.history
        
        for entry in recent_history:
            timestamp = entry.get('timestamp', '')
            msg_type = entry.get('type', 'unknown')
            message = entry.get('message', '')[:100] + ('...' if len(entry.get('message', '')) > 100 else '')
            
            style = self.styles.get(msg_type, "white")
            table.add_row(
                timestamp,
                f"[{style}]{msg_type.upper()}[/{style}]",
                message
            )
        
        self.console.print(table)
    
    def display_tools_info(self):
        """Отображение доступных MCP инструментов с умной категоризацией"""
        if not self.agent or not self.agent.tools:
            self.console.print("[red]❌ Нет доступных инструментов[/red]")
            return
        
        # Показываем инструменты по категориям
        if hasattr(self.agent, 'tools_map') and self.agent.tools_map:
            for category, tools in self.agent.tools_map.items():
                if not tools:
                    continue
                
                category_names = {
                    'read_file': '📖 Чтение файлов',
                    'write_file': '✏️ Создание/запись файлов',
                    'list_directory': '📁 Просмотр директорий',
                    'create_directory': '📂 Создание папок',
                    'delete_file': '🗑️ Удаление файлов',
                    'move_file': '📦 Перемещение файлов',
                    'search': '🔍 Поиск файлов',
                    'web_search': '🌐 Веб-поиск',
                    'fetch_url': '⬇️ Загрузка из интернета',
                    'other': '🔧 Другие инструменты'
                }
                
                category_title = category_names.get(category, category.title())
                
                table = Table(
                    title=f"[bold]{category_title}[/bold]",
                    box=box.ROUNDED,
                    show_header=True
                )
                table.add_column("Tool Name", style="cyan", no_wrap=True)
                table.add_column("Description", style="white")
                
                for tool in tools:
                    description = self.agent._get_tool_description(tool)
                    table.add_row(tool.name, description)
                
                self.console.print(table)
                self.console.print()
        else:
            # Fallback к старому отображению
            table = Table(
                title="[bold]Available MCP Tools[/bold]", 
                box=box.ROUNDED,
                show_header=True
            )
            table.add_column("Tool Name", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")
            
            for tool in self.agent.tools:
                description = ""
                if hasattr(tool, 'description') and tool.description:
                    description = tool.description[:80] + ('...' if len(tool.description) > 80 else '')
                
                if not description:
                    description = "Нет описания"
                
                table.add_row(tool.name, description)
            
            self.console.print(table)
        
        # Дополнительная информация
        self.console.print(f"[dim]Всего инструментов: {len(self.agent.tools)}[/dim]")
        self.console.print("[dim]🧠 Агент автоматически выбирает подходящие инструменты на основе анализа ваших запросов[/dim]")
        
        # Показываем примеры умных запросов
        examples_table = Table(
            title="[bold]🎯 Примеры умных запросов[/bold]",
            box=box.SIMPLE,
            show_header=False
        )
        examples_table.add_column("Request", style="green")
        examples_table.add_column("→", style="dim", width=3)
        examples_table.add_column("Action", style="yellow")
        
        examples = [
            ("создай файл readme.md с описанием проекта", "→", "Создаст файл с содержимым"),
            ("покажи все Python файлы", "→", "Найдет и покажет .py файлы"),
            ("удали все временные файлы", "→", "Найдет и удалит temp файлы"),
            ("найди в интернете последние новости Python", "→", "Выполнит веб-поиск"),
            ("скачай документацию с example.com", "→", "Загрузит контент по URL")
        ]
        
        for req, arrow, action in examples:
            examples_table.add_row(req, arrow, action)
        
        self.console.print(examples_table)
    
    def display_agent_response(self, response: str, response_time: float = None):
        """Красивое отображение ответа агента"""
        # Определяем тип контента
        if response.startswith('```') and response.endswith('```'):
            # Код
            lines = response.strip('`').split('\n')
            language = lines[0] if lines[0] else 'text'
            code = '\n'.join(lines[1:])
            
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            panel = Panel(
                syntax,
                title="[bold green]🤖 Gemini Response (Code)[/bold green]",
                border_style="green"
            )
        else:
            # Обычный текст или markdown
            try:
                # Попробуем интерпретировать как markdown
                content = Markdown(response)
            except:
                content = Text(response)
            
            panel = Panel(
                content,
                title="[bold green]🤖 Gemini Response[/bold green]",
                border_style="green"
            )
        
        self.console.print(panel)
        
        # Показать время ответа, если доступно
        if response_time:
            self.console.print(f"[dim]⏱️ Response time: {response_time:.2f}s[/dim]")
    
    def display_error(self, error_message: str):
        """Отображение ошибки"""
        panel = Panel(
            f"❌ {error_message}",
            title="[bold red]Error[/bold red]",
            border_style="red"
        )
        self.console.print(panel)
    
    def display_success(self, message: str):
        """Отображение успешного выполнения"""
        panel = Panel(
            f"✅ {message}",
            title="[bold green]Success[/bold green]",
            border_style="green"
        )
        self.console.print(panel)
    
    def get_user_input(self) -> Optional[str]:
        """Получение ввода пользователя с красивым промптом"""
        try:
            prompt_text = Text()
            prompt_text.append("💬 You", style="bold blue")
            prompt_text.append(" › ", style="dim")
            
            user_input = Prompt.ask(prompt_text, console=self.console).strip()
            
            if user_input.lower() in ['quit', 'exit', '/quit']:
                return None
            
            return user_input
            
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Goodbye! 👋[/dim]")
            return None
    
    def process_system_command(self, command: str) -> bool:
        """Обработка системных команд"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            self.display_help()
            return True
        
        elif cmd == '/clear':
            self.clear_screen()
            self.print_header()
            self.print_status_bar()
            return True
        
        elif cmd == '/status' and self.agent:
            status = self.agent.get_status()
            self.display_status_info(status)
            return True
        
        elif cmd == '/tools' and self.agent:
            self.display_tools_info()
            return True
        
        elif cmd == '/tree':
            path = parts[1] if len(parts) > 1 else (
                self.agent.config.filesystem_path if self.agent else os.getcwd()
            )
            self.display_file_tree(path)
            return True
        
        elif cmd == '/history':
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            self.display_history(limit)
            return True
        
        elif cmd == '/thread':
            if len(parts) > 1:
                self.current_thread = parts[1]
                self.console.print(f"[green]✅ Switched to thread: {self.current_thread}[/green]")
            else:
                self.console.print(f"[blue]Current thread: {self.current_thread}[/blue]")
            return True
        
        elif cmd == '/theme':
            if len(parts) > 1 and parts[1] in ['light', 'dark']:
                self.theme = parts[1]
                self.console.print(f"[green]✅ Theme changed to: {self.theme}[/green]")
            else:
                self.console.print(f"[blue]Current theme: {self.theme}[/blue]")
            return True
        
        elif cmd == '/timestamps':
            if len(parts) > 1 and parts[1] in ['on', 'off']:
                self.show_timestamps = parts[1] == 'on'
                status = "enabled" if self.show_timestamps else "disabled" 
                self.console.print(f"[green]✅ Timestamps {status}[/green]")
            else:
                status = "enabled" if self.show_timestamps else "disabled"
                self.console.print(f"[blue]Timestamps: {status}[/blue]")
            return True
        
        elif cmd == '/export':
            self.export_history()
            return True
        
        return False
    
    def display_status_info(self, status: Dict[str, Any]):
        """Подробное отображение статуса умной системы"""
        # Основная информация
        main_table = Table(title="[bold]🤖 Smart Agent Status[/bold]", box=box.ROUNDED)
        main_table.add_column("Property", style="cyan")
        main_table.add_column("Value", style="green")
        
        # Исключаем сложные объекты из основной таблицы
        simple_status = {k: v for k, v in status.items() 
                        if not isinstance(v, (dict, list)) or k == 'intelligence_features'}
        
        for key, value in simple_status.items():
            if key == 'intelligence_features':
                value = ', '.join(value)
            table_key = key.replace('_', ' ').title()
            main_table.add_row(table_key, str(value))
        
        self.console.print(main_table)
        
        # Детали по категориям инструментов
        if 'tools_by_category' in status and status['tools_by_category']:
            tools_table = Table(title="[bold]🔧 Tools by Category[/bold]", box=box.SIMPLE)
            tools_table.add_column("Category", style="magenta")
            tools_table.add_column("Count", style="yellow", justify="right")
            
            category_icons = {
                'read_file': '📖',
                'write_file': '✏️',
                'list_directory': '📁',
                'create_directory': '📂',
                'delete_file': '🗑️',
                'move_file': '📦',
                'search': '🔍',
                'web_search': '🌐',
                'fetch_url': '⬇️',
                'other': '🔧'
            }
            
            for category, count in status['tools_by_category'].items():
                icon = category_icons.get(category, '•')
                category_name = category.replace('_', ' ').title()
                tools_table.add_row(f"{icon} {category_name}", str(count))
            
            self.console.print(tools_table)
        
        # Информация о памяти контекста
        if status.get('recent_files_count', 0) > 0:
            memory_panel = Panel(
                f"🧠 Context Memory: {status['context_memory_items']} items\n"
                f"📝 Recent Files: {status['recent_files_count']} files tracked",
                title="[bold]Memory Status[/bold]",
                border_style="blue"
            )
            self.console.print(memory_panel)
    
    def export_history(self):
        """Экспорт истории в файл"""
        if not self.history:
            self.console.print("[yellow]⚠️ История пуста[/yellow]")
            return
        
        filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Gemini AI Agent Chat History\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for entry in self.history:
                    f.write(f"## {entry['type'].upper()} - {entry['timestamp']}\n\n")
                    f.write(f"{entry['message']}\n\n---\n\n")
            
            self.console.print(f"[green]✅ История экспортирована в: {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка экспорта: {e}[/red]")
    
    def add_to_history(self, message: str, msg_type: str = "user"):
        """Добавление сообщения в историю"""
        timestamp = datetime.now().strftime('%H:%M:%S') if self.show_timestamps else ''
        
        self.history.append({
            'timestamp': timestamp,
            'type': msg_type,
            'message': message,
            'thread': self.current_thread
        })
    
    async def run(self):
        """Основной цикл чата"""
        self.clear_screen()
        self.print_header()
        self.print_status_bar()
        
        self.console.print("[dim]Type /help for available commands, /quit to exit[/dim]")
        self.console.print(Rule(style="dim"))
        
        while True:
            user_input = self.get_user_input()
            
            if user_input is None:
                break
            
            if not user_input:
                continue
            
            # Добавить в историю
            self.add_to_history(user_input, "user")
            
            # Обработка системных команд
            if user_input.startswith('/'):
                if self.process_system_command(user_input):
                    continue
            
            # Отправка агенту
            if self.agent:
                try:
                    with Status("[dim]🤔 Gemini is thinking...[/dim]", console=self.console):
                        start_time = time.time()
                        response = await self.agent.process_message(user_input, self.current_thread)
                        response_time = time.time() - start_time
                    
                    self.add_to_history(response, "agent")
                    self.display_agent_response(response, response_time)
                    
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    self.add_to_history(error_msg, "error")
                    self.display_error(error_msg)
            else:
                self.display_error("Agent not initialized")
            
            self.console.print()  # Добавить пустую строку для разделения
        
        self.console.print("[dim]Goodbye! 👋[/dim]")


# ===== ГЛАВНАЯ ФУНКЦИЯ =====
async def main():
    """Главная функция с Rich интерфейсом"""
    load_dotenv()
    
    try:
        # Создание конфигурации (только для Gemini)
        config = AgentConfig(
            filesystem_path=os.getenv("FILESYSTEM_PATH"),  # Может быть None
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("TEMPERATURE", "0.0"))
        )
        
        # Создание и инициализация агента
        agent = FileSystemAgent(config)
        
        # Создаем богатый интерфейс для показа прогресса инициализации
        console = Console()
        
        with console.status("[bold green]Initializing Gemini Agent...", spinner="dots"):
            if not await agent.initialize():
                console.print("❌ [bold red]Не удалось инициализировать агента[/bold red]")
                return
        
        console.print("✅ [bold green]Gemini Agent successfully initialized![/bold green]")
        
        # Запуск богатого чата
        chat = RichInteractiveChat(agent)
        await chat.run()
        
    except Exception as e:
        Console().print(f"❌ [bold red]Критическая ошибка: {e}[/bold red]")
    
    logger.info("🏁 Завершение работы")


if __name__ == "__main__":
    asyncio.run(main())