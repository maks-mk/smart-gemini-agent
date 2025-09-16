import asyncio
import logging
import os
import time
import re
import json
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
    level=logging.INFO,  # Возвращаем обычный уровень логирования
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
    """Конфигурация AI-агента для Gemini с загрузкой из файла"""
    filesystem_path: str = None  # По умолчанию None
    use_memory: bool = True
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    debug_intent_analysis: bool = False
    prompt_file: str = "prompt.md"
    mcp_config_file: str = "mcp.json"
    max_context_files: int = 20
    
    @classmethod
    def from_file(cls, config_file: str = "config.json") -> 'AgentConfig':
        """Создание конфигурации из файла"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Извлекаем настройки агента
                agent_config = config_data.get('agent', {})
                files_config = config_data.get('files', {})
                logging_config = config_data.get('logging', {})
                
                return cls(
                    model_name=agent_config.get('model_name', 'gemini-2.5-flash'),
                    temperature=agent_config.get('temperature', 0.0),
                    use_memory=agent_config.get('use_memory', True),
                    max_context_files=agent_config.get('max_context_files', 20),
                    debug_intent_analysis=logging_config.get('debug_intent_analysis', False),
                    prompt_file=files_config.get('prompt_file', 'prompt.md'),
                    mcp_config_file=files_config.get('mcp_config_file', 'mcp.json')
                )
            else:
                logger.info(f"Файл конфигурации {config_file} не найден, используются настройки по умолчанию")
                return cls()
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации из {config_file}: {e}")
            return cls()
   
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

    # Старый метод закомментирован (удалите, если не нужен)
    # def get_mcp_config(self) -> Dict[str, Any]:
    #     """Конфигурация MCP сервера"""
    #     return {
    #         "filesystem": {
    #             "command": "npx",
    #             "args": ["-y", "@modelcontextprotocol/server-filesystem", self.filesystem_path],
    #             "transport": "stdio"
    #         },
    #         "duckduckgo": {
    #             "command": "uvx",
    #             "args": ["duckduckgo-mcp-server"],
    #             "transport": "stdio"
    #         },
    #         "fetch": {
    #             "command": "uvx",
    #             "args": ["mcp-server-fetch"],
    #             "transport": "stdio",
    #         }
    #     }

    def get_mcp_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации MCP серверов из файла"""
        mcp_config_path = self.mcp_config_file
        
        try:
            if not os.path.exists(mcp_config_path):
                logger.warning(f"Файл {mcp_config_path} не найден, используется конфигурация по умолчанию")
                return self._get_default_mcp_config()
            
            with open(mcp_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Заменяем плейсхолдеры во всей конфигурации
            # Нормализуем путь для JSON (используем прямые слеши)
            normalized_path = self.filesystem_path.replace('\\', '/')
            config_str = json.dumps(config)
            config_str = config_str.replace('{filesystem_path}', normalized_path)
            config = json.loads(config_str)
            
            logger.info(f"✅ Загружена конфигурация MCP из {mcp_config_path}: {list(config.keys())}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга {mcp_config_path}: {e}")
            logger.error(f"Проблемный путь: {self.filesystem_path}")
            logger.info("Используется конфигурация по умолчанию")
            return self._get_default_mcp_config()
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {mcp_config_path}: {e}")
            return self._get_default_mcp_config()
    
    def _get_default_mcp_config(self) -> Dict[str, Any]:
        """Конфигурация MCP серверов по умолчанию"""
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
                "transport": "stdio"
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
        """Универсальный анализ и категоризация доступных инструментов"""
        # Инициализируем базовые категории
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
        
        # Паттерны для автоматической категоризации
        categorization_patterns = {
            'read_file': [
                r'(read|get|cat|show|view).*file',
                r'(read|get)_.*',
                r'.*_read.*'
            ],
            'write_file': [
                r'(write|create|save|put).*file',
                r'(write|create)_.*',
                r'.*_write.*',
                r'.*_create.*'
            ],
            'list_directory': [
                r'(list|ls|dir).*',
                r'.*(directory|dir|folder).*list',
                r'.*list.*(directory|dir|folder)',
                r'get.*directory'
            ],
            'create_directory': [
                r'create.*(directory|dir|folder)',
                r'mkdir',
                r'make.*dir'
            ],
            'delete_file': [
                r'(delete|remove|rm|unlink).*',
                r'safe_delete.*',
                r'.*(shell|exec|run|command).*'  # Shell команды могут удалять
            ],
            'move_file': [
                r'(move|mv|rename).*',
                r'.*_move.*',
                r'.*_rename.*'
            ],
            'search': [
                r'(search|find|grep).*file',
                r'.*search.*',
                r'.*find.*',
                r'.*grep.*'
            ],
            'web_search': [
                r'.*(web|internet|duckduckgo|google).*search',
                r'.*web.*',
                r'.*duckduckgo.*'
            ],
            'fetch_url': [
                r'(fetch|download|get).*url',
                r'(http|https|url).*',
                r'.*fetch.*',
                r'.*download.*'
            ]
        }
        
        # Категоризируем каждый инструмент
        for tool in self.tools:
            name = tool.name.lower()
            description = getattr(tool, 'description', '').lower() if hasattr(tool, 'description') else ''
            
            categorized = False
            
            # Проверяем каждую категорию
            for category, patterns in categorization_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, name) or (description and re.search(pattern, description)):
                        self.tools_map[category].append(tool)
                        categorized = True
                        break
                if categorized:
                    break
            
            # Если инструмент не попал ни в одну категорию
            if not categorized:
                self.tools_map['other'].append(tool)
        
        # Логируем результаты категоризации
        logger.info("Автоматическая категоризация инструментов:")
        for category, tools in self.tools_map.items():
            if tools:
                logger.info(f"  {category}: {[t.name for t in tools]}")
        
        # Предупреждение если есть много неопознанных инструментов
        if len(self.tools_map['other']) > len(self.tools) * 0.3:
            logger.warning(f"Много инструментов в категории 'other' ({len(self.tools_map['other'])}). Возможно, нужно обновить паттерны категоризации.")
    
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
                r'читай\s+файл\s+([^\s]+)',
                r'читай\s+([^\s]+)',
                r'прочитай\s+файл\s+([^\s]+)',
                r'прочитай\s+([^\s]+)',
                r'покажи\s+файл\s+([^\s]+)',
                r'покажи\s+содержимое\s+файла\s+([^\s]+)',
                r'покажи\s+содержимое\s+([^\s]+)',
                r'открой\s+файл\s+([^\s]+)',
                r'открой\s+([^\s]+)',
                r'read\s+file\s+([^\s]+)',
                r'read\s+([^\s]+)',
                r'show\s+file\s+([^\s]+)',
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
                    target = match.group(1) if match.groups() else None
                    # Очищаем target от лишних слов
                    if target:
                        target = target.strip()
                        # Убираем слова-паразиты
                        target = re.sub(r'^(файл|файла|file)\s+', '', target)
                        target = re.sub(r'^(папку|папка|folder|directory)\s+', '', target)
                    
                    params = {'target': target}
                    
                    # Дополнительный анализ для извлечения контента
                    if intent == 'create_file':
                        content_match = re.search(r'с содержимым\s+(.+)|с текстом\s+(.+)|with content\s+(.+)', user_input_lower)
                        if content_match:
                            params['content'] = content_match.group(1) or content_match.group(2) or content_match.group(3)
                    
                    if self.config.debug_intent_analysis:
                        logger.info(f"🎯 Найдено совпадение: паттерн='{pattern}', намерение='{intent}', параметры={params}")
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
    

    
    def _load_prompt_from_file(self) -> str:
        """Загрузка промпта из файла с подстановкой переменных"""
        prompt_file = self.config.prompt_file
        
        try:
            if not os.path.exists(prompt_file):
                logger.warning(f"Файл {prompt_file} не найден, используется промпт по умолчанию")
                return self._get_default_prompt()
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            
            # Удаляем заголовок markdown если есть
            if prompt_template.startswith('# '):
                lines = prompt_template.split('\n')
                # Находим первую пустую строку после заголовка
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() == '' and i > 0:
                        start_idx = i + 1
                        break
                prompt_template = '\n'.join(lines[start_idx:])
            
            # Подставляем переменные безопасно
            tools_description = self._generate_tools_description()
            prompt = prompt_template.replace('{filesystem_path}', self.config.filesystem_path)
            prompt = prompt.replace('{tools_description}', tools_description)
            
            logger.info(f"✅ Загружен промпт из {prompt_file}")
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки промпта из {prompt_file}: {e}")
            logger.info("Используется промпт по умолчанию")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Универсальный промпт по умолчанию на случай ошибки загрузки"""
        tools_description = self._generate_tools_description()
        
        return f"""Ты умный AI-ассистент с доступом к различным инструментам.

РАБОЧАЯ ДИРЕКТОРИЯ: {self.config.filesystem_path}
Все файловые операции выполняются относительно этой директории.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_description}

ПРАВИЛА РАБОТЫ:
1. Анализируй запрос пользователя и определяй нужную операцию
2. Используй ТОЛЬКО предоставленные инструменты из списка выше
3. Автоматически выбирай подходящий инструмент на основе его названия и описания
4. Всегда указывай полные пути для файлов и папок
5. При создании файлов с содержимым используй соответствующие параметры
6. Если путь не указан явно, работай в текущей рабочей директории
7. Адаптируйся к доступным инструментам - если нет конкретного инструмента, найди альтернативу
8. НЕ ВЫДУМЫВАЙ несуществующие инструменты или команды

ФОРМАТ ОТВЕТА:
- Кратко подтверди выполненное действие
- При ошибках объясни причину и предложи решение
- Для сложных операций опиши что делаешь пошагово"""
    
    def _generate_tools_description(self) -> str:
        """Генерация детального описания инструментов"""
        descriptions = []
        
        # Автоматическое определение категорий на основе доступных инструментов
        category_names = {
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
        }
        
        # Показываем только категории с доступными инструментами
        for category, tools in self.tools_map.items():
            if tools:  # Показываем только непустые категории
                category_desc = category_names.get(category, category.replace('_', ' ').upper())
                descriptions.append(f"\n{category_desc}:")
                
                for tool in tools:
                    tool_desc = self._get_tool_description(tool)
                    descriptions.append(f"  • {tool.name}: {tool_desc}")
        
        # Если нет инструментов вообще
        if not descriptions:
            descriptions.append("\nИнструменты не загружены или недоступны.")
        
        return '\n'.join(descriptions)
    
    def _get_tool_description(self, tool) -> str:
        """Получение описания инструмента с автоматическим анализом"""
        # Сначала пробуем использовать оригинальное описание
        if hasattr(tool, 'description') and tool.description:
            desc = tool.description.strip()
            if desc:
                return desc[:150] + ('...' if len(desc) > 150 else '')
        
        # Автоматическое определение функции на основе названия
        name = tool.name.lower()
        
        # Паттерны для определения функций инструментов
        patterns = {
            # Чтение файлов
            r'(read|get|cat|show|view).*file': "Читает содержимое файла",
            r'(read|get)_.*': "Читает данные",
            
            # Запись файлов
            r'(write|create|save|put).*file': "Создает или записывает файл",
            r'(write|create)_.*': "Создает или записывает данные",
            
            # Работа с директориями
            r'(list|ls|dir).*': "Показывает содержимое директории",
            r'.*(directory|dir|folder).*list': "Показывает содержимое директории",
            r'create.*(directory|dir|folder)': "Создает новую папку",
            r'mkdir': "Создает новую папку",
            
            # Удаление
            r'(delete|remove|rm|unlink).*': "Удаляет файл или папку",
            r'safe_delete_file': "Безопасно удаляет файлы только внутри рабочей директории",
            r'safe_delete_directory': "Безопасно удаляет директории только внутри рабочей директории",
            
            # Перемещение
            r'(move|mv|rename).*': "Перемещает или переименовывает файл",
            
            # Поиск
            r'(search|find|grep).*file': "Ищет файлы по критериям",
            r'(search|find).*': "Выполняет поиск",
            
            # Веб-функции
            r'.*(web|internet|duckduckgo|google).*search': "Поиск информации в интернете",
            r'(fetch|download|get).*url': "Загружает данные по URL",
            r'(http|https|url).*': "Работает с веб-ресурсами",
            
            # Системные команды
            r'(shell|exec|run|command).*': "Выполняет системные команды",
            
            # Специальные функции
            r'.*server.*': "MCP сервер инструмент",
            r'.*mcp.*': "Model Context Protocol инструмент"
        }
        
        # Проверяем паттерны
        for pattern, description in patterns.items():
            if re.search(pattern, name):
                return description
        
        # Если ничего не подошло, возвращаем общее описание
        return f"Инструмент: {tool.name}"
    
    def _get_system_prompt(self) -> str:
        """Системный промпт"""
        return self._load_prompt_from_file()

    
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
            
            # УЛУЧШЕННОЕ извлечение содержимого ответа
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                if messages:
                    last_message = messages[-1]
                    # Извлекаем содержимое в зависимости от типа сообщения
                    if hasattr(last_message, 'content'):
                        content = str(last_message.content)
                    elif isinstance(last_message, dict) and 'content' in last_message:
                        content = str(last_message['content'])
                    else:
                        content = str(last_message)
                    
                    # Дополнительная проверка на пустое содержимое
                    if not content or content.strip() == "":
                        logger.warning("Получено пустое содержимое от агента")
                        return "❌ Агент вернул пустой ответ"
                    
                    logger.debug(f"Извлечено содержимое: {content[:100]}...")
                    return content
                else:
                    logger.warning("Получен ответ без сообщений")
                    return "❌ Получен пустой ответ от агента"
            else:
                # Если response не в ожидаемом формате
                content = str(response)
                logger.debug(f"Ответ не в ожидаемом формате, преобразуем в строку: {type(response)}")
                return content
            
        except Exception as e:
            error_msg = f"❌ Ошибка обработки: {e}"
            logger.error(error_msg)
            logger.error(f"Тип ответа: {type(response) if 'response' in locals() else 'не определен'}")
            if 'response' in locals():
                logger.error(f"Содержимое ответа: {str(response)[:500]}...")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
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
        """Красивое отображение ответа агента с улучшенным форматированием"""
        # Проверяем, содержит ли ответ плохо отформатированное содержимое файла
        response = self._improve_file_content_formatting(response)
        
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
    
    def _improve_file_content_formatting(self, response: str) -> str:
        """Улучшает форматирование содержимого файлов в ответе"""
        import re
        
        # Логируем исходный ответ для отладки
        logger.debug(f"Исходный ответ для форматирования: {response[:200]}...")
        
        # Специальная обработка для ответов-массивов
        if response.startswith('[') and response.endswith(']'):
            logger.debug("Обнаружен ответ в формате массива")
            try:
                import ast
                # Пробуем безопасно распарсить массив
                parsed = ast.literal_eval(response)
                if isinstance(parsed, list) and len(parsed) >= 2:
                    description = str(parsed[0])
                    content = str(parsed[1])
                    
                    # Извлекаем имя файла из описания
                    filename_match = re.search(r'([^\s]+\.[a-zA-Z0-9]+)', description)
                    if filename_match:
                        filename = filename_match.group(1)
                        
                        # Убираем лишние символы из содержимого
                        if content.startswith('```') and content.endswith('```'):
                            # Содержимое уже в markdown формате
                            lines = content.strip('`').split('\n')
                            if lines and lines[0]:  # Первая строка - язык
                                language = lines[0]
                                actual_content = '\n'.join(lines[1:])
                            else:
                                language = self._get_language_by_filename(filename)
                                actual_content = '\n'.join(lines[1:])
                        else:
                            language = self._get_language_by_filename(filename)
                            actual_content = content
                        
                        # Автоматическое форматирование содержимого
                        actual_content = self._format_file_content(actual_content, language, filename)
                        
                        formatted_response = f"Содержимое файла `{filename}`:\n\n```{language}\n{actual_content}\n```"
                        logger.debug(f"Успешно отформатирован ответ-массив для файла {filename}")
                        return formatted_response
            except (ValueError, SyntaxError) as e:
                logger.debug(f"Не удалось распарсить массив: {e}")
        
        # Проверяем, не является ли ответ уже правильно отформатированным
        if "```" in response and ("Содержимое файла" in response or "содержимое файла" in response):
            logger.debug("Ответ уже правильно отформатирован")
            return response
        
        # Паттерны для поиска плохо отформатированного содержимого файлов
        patterns = [
            # Массив с описанием и содержимым
            r"\['Содержимое файла ([^']+):', '([^']+)'\]",
            r"\['Файл ([^']+) содержит[^']*:', '([^']+)'\]",
            r"\['([^']*файл[^']*)', '([^']+)'\]",
            # Простой массив с двумя элементами
            r"\[([^,]+), '([^']+)'\]",
            # Массив с кавычками
            r'\["([^"]*файл[^"]*)", "([^"]+)"\]',
            # Строка с содержимым файла без массива
            r'Содержимое файла ([^:]+):\s*(.+?)(?=\n\n|\Z)',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                logger.debug(f"Найдено совпадение с паттерном {i}: {pattern}")
                
                first_part = match.group(1).strip()
                content = match.group(2).strip()
                
                # Извлекаем имя файла из первой части
                filename_match = re.search(r'([^\s]+\.[a-zA-Z0-9]+)', first_part)
                if filename_match:
                    filename = filename_match.group(1)
                else:
                    # Пробуем найти имя файла в самом ответе
                    filename_in_response = re.search(r'([^\s]+\.[a-zA-Z0-9]+)', response)
                    filename = filename_in_response.group(1) if filename_in_response else "file.txt"
                
                # Убираем escape-символы и лишние кавычки
                content = content.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                content = content.strip('"\'')
                
                # Если содержимое пустое или слишком короткое, возвращаем исходный ответ
                if len(content.strip()) < 3:
                    logger.debug("Содержимое слишком короткое, возвращаем исходный ответ")
                    return response
                
                # Определяем язык по расширению файла
                language = self._get_language_by_filename(filename)
                
                # Автоматическое форматирование содержимого
                content = self._format_file_content(content, language, filename)
                
                # Создаем правильно отформатированный ответ
                formatted_response = f"Содержимое файла `{filename}`:\n\n```{language}\n{content}\n```"
                logger.debug(f"Создан отформатированный ответ для файла {filename}")
                return formatted_response
        
        # Проверяем, есть ли в ответе упоминание файла без правильного форматирования
        if re.search(r'(файл|file).*\.(txt|py|js|json|md|yml|xml|csv)', response, re.IGNORECASE):
            logger.debug("Найдено упоминание файла, но не удалось извлечь содержимое")
        
        logger.debug("Паттерны не найдены, возвращаем исходный ответ")
        return response
    
    def _get_language_by_filename(self, filename: str) -> str:
        """Определяет язык для подсветки синтаксиса по имени файла"""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        language_map = {
            'json': 'json',
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'md': 'markdown',
            'yml': 'yaml',
            'yaml': 'yaml',
            'xml': 'xml',
            'html': 'html',
            'css': 'css',
            'sql': 'sql',
            'sh': 'bash',
            'ps1': 'powershell',
            'csv': 'csv',
            'txt': 'text',
            'log': 'text',
            'ini': 'ini',
            'cfg': 'ini',
            'conf': 'text',
            'env': 'bash'
        }
        
        return language_map.get(extension, 'text')
    
    def _format_file_content(self, content: str, language: str, filename: str) -> str:
        """Автоматическое форматирование содержимого файлов"""
        if not content or not content.strip():
            return content
        
        try:
            if language == 'json':
                # Форматирование JSON
                import json
                parsed_json = json.loads(content)
                formatted_content = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                logger.debug(f"JSON файл {filename} автоматически отформатирован")
                return formatted_content
                
            elif language == 'xml':
                # Форматирование XML
                try:
                    import xml.dom.minidom
                    dom = xml.dom.minidom.parseString(content)
                    formatted_content = dom.toprettyxml(indent="  ")
                    # Убираем лишние пустые строки
                    lines = [line for line in formatted_content.split('\n') if line.strip()]
                    formatted_content = '\n'.join(lines)
                    logger.debug(f"XML файл {filename} автоматически отформатирован")
                    return formatted_content
                except:
                    pass  # Если не удалось, оставляем как есть
                    
            elif language in ['yaml', 'yml']:
                # Для YAML пробуем базовое форматирование отступов
                lines = content.split('\n')
                formatted_lines = []
                indent_level = 0
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        formatted_lines.append('')
                        continue
                        
                    # Простое форматирование YAML (базовое)
                    if ':' in stripped and not stripped.startswith('-'):
                        formatted_lines.append('  ' * indent_level + stripped)
                        if not stripped.endswith(':'):
                            indent_level = max(0, indent_level)
                    else:
                        formatted_lines.append('  ' * indent_level + stripped)
                
                if len(formatted_lines) != len(lines):  # Если что-то изменилось
                    logger.debug(f"YAML файл {filename} автоматически отформатирован")
                    return '\n'.join(formatted_lines)
                    
        except Exception as e:
            logger.debug(f"Не удалось отформатировать {language} файл {filename}: {e}")
        
        # Если форматирование не удалось или не нужно, возвращаем как есть
        return content
    
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
                    
                    # Дополнительное логирование для отладки
                    logger.debug(f"Тип ответа: {type(response)}")
                    logger.debug(f"Длина ответа: {len(str(response))}")
                    logger.debug(f"Первые 200 символов ответа: {str(response)[:200]}")
                    
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
        # Создание конфигурации из файла или переменных окружения
        config = AgentConfig.from_file("config.json")
        
        # Переопределяем из переменных окружения если они заданы
        if os.getenv("FILESYSTEM_PATH"):
            config.filesystem_path = os.getenv("FILESYSTEM_PATH")
        if os.getenv("GEMINI_MODEL"):
            config.model_name = os.getenv("GEMINI_MODEL")
        if os.getenv("TEMPERATURE"):
            config.temperature = float(os.getenv("TEMPERATURE"))
        
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