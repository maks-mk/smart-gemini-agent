# 📚 API Documentation

## Основные классы

### FileSystemAgent

Главный класс агента, который управляет всеми операциями.

```python
class FileSystemAgent:
    def __init__(self, config: AgentConfig)
    async def initialize(self) -> bool
    async def process_message(self, user_input: str, thread_id: str = "default") -> str
    def get_status(self) -> Dict[str, Any]
```

#### Методы

##### `__init__(config: AgentConfig)`
Инициализирует агент с заданной конфигурацией.

**Параметры:**
- `config` (AgentConfig): Конфигурация агента

##### `async initialize() -> bool`
Асинхронно инициализирует агент, подключает MCP серверы и создает инструменты.

**Возвращает:**
- `bool`: True если инициализация успешна, False в противном случае

##### `async process_message(user_input: str, thread_id: str = "default") -> str`
Обрабатывает сообщение пользователя и возвращает ответ.

**Параметры:**
- `user_input` (str): Сообщение пользователя
- `thread_id` (str): Идентификатор потока для контекста

**Возвращает:**
- `str`: Ответ агента

##### `get_status() -> Dict[str, Any]`
Возвращает текущий статус агента.

**Возвращает:**
- `Dict[str, Any]`: Словарь с информацией о статусе

### AgentConfig

Класс конфигурации для агента.

```python
@dataclass
class AgentConfig:
    filesystem_path: str = None
    use_memory: bool = True
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.0
```

#### Поля

- `filesystem_path` (str): Путь к рабочей директории
- `use_memory` (bool): Включить ли память агента
- `model_name` (str): Название модели Gemini
- `temperature` (float): Температура для генерации ответов

### SafeDeleteFileTool

Локальный инструмент для безопасного удаления файлов.

```python
class SafeDeleteFileTool(BaseTool):
    name: str = "safe_delete_file"
    description: str = "Безопасно удаляет файл только внутри рабочей директории"
    
    def _run(self, file_path: str) -> str
    async def _arun(self, file_path: str) -> str
```

#### Методы

##### `_run(file_path: str) -> str`
Синхронно удаляет файл.

**Параметры:**
- `file_path` (str): Относительный путь к файлу

**Возвращает:**
- `str`: Сообщение о результате операции

### SafeDeleteDirectoryTool

Локальный инструмент для безопасного удаления директорий.

```python
class SafeDeleteDirectoryTool(BaseTool):
    name: str = "safe_delete_directory"
    description: str = "Безопасно удаляет директорию только внутри рабочей директории"
    
    def _run(self, dir_path: str, recursive: bool = False) -> str
    async def _arun(self, dir_path: str, recursive: bool = False) -> str
```

#### Методы

##### `_run(dir_path: str, recursive: bool = False) -> str`
Синхронно удаляет директорию.

**Параметры:**
- `dir_path` (str): Относительный путь к директории
- `recursive` (bool): Удалить рекурсивно со всем содержимым

**Возвращает:**
- `str`: Сообщение о результате операции

## Внутренние методы

### Анализ намерений

#### `_analyze_user_intent(user_input: str) -> Tuple[str, Dict[str, Any]]`
Анализирует намерения пользователя из текста.

**Параметры:**
- `user_input` (str): Ввод пользователя

**Возвращает:**
- `Tuple[str, Dict[str, Any]]`: Намерение и извлеченные параметры

**Поддерживаемые намерения:**
- `create_file`: Создание файла
- `create_directory`: Создание директории
- `read_file`: Чтение файла
- `list_directory`: Просмотр директории
- `delete_file`: Удаление файла
- `move_file`: Перемещение файла
- `search`: Поиск файлов
- `web_search`: Веб-поиск
- `general`: Общий запрос

### Контекстная память

#### `_update_context_memory(intent: str, params: Dict[str, Any], response: Any)`
Обновляет контекстную память агента.

**Параметры:**
- `intent` (str): Намерение пользователя
- `params` (Dict[str, Any]): Параметры запроса
- `response` (Any): Ответ агента

### Категоризация инструментов

#### `_analyze_tools()`
Анализирует и категоризирует доступные инструменты.

**Категории:**
- `read_file`: Чтение файлов
- `write_file`: Создание/запись файлов
- `list_directory`: Просмотр директорий
- `create_directory`: Создание папок
- `delete_file`: Удаление файлов/папок
- `move_file`: Перемещение/переименование
- `search`: Поиск файлов
- `web_search`: Веб-поиск
- `fetch_url`: Загрузка из интернета
- `other`: Другие инструменты

## Использование API

### Базовое использование

```python
import asyncio
from gemini_agent import FileSystemAgent, AgentConfig

async def main():
    # Создание конфигурации
    config = AgentConfig(
        filesystem_path="/path/to/workspace",
        model_name="gemini-2.0-flash",
        temperature=0.0
    )
    
    # Создание и инициализация агента
    agent = FileSystemAgent(config)
    if not await agent.initialize():
        print("Ошибка инициализации")
        return
    
    # Обработка сообщения
    response = await agent.process_message("создай файл test.txt")
    print(response)
    
    # Получение статуса
    status = agent.get_status()
    print(f"Инструментов: {status['tools_count']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Работа с контекстом

```python
async def context_example():
    agent = FileSystemAgent(AgentConfig())
    await agent.initialize()
    
    # Первый запрос
    response1 = await agent.process_message(
        "удали файл old.txt", 
        thread_id="session1"
    )
    print(response1)
    
    # Контекстный запрос (ссылка на предыдущий)
    response2 = await agent.process_message(
        "1",  # Выбор первого варианта
        thread_id="session1"
    )
    print(response2)
```

### Создание собственных инструментов

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CustomToolInput(BaseModel):
    param: str = Field(description="Описание параметра")

class CustomTool(BaseTool):
    name: str = "custom_tool"
    description: str = "Описание инструмента"
    args_schema: type[BaseModel] = CustomToolInput
    
    def _run(self, param: str) -> str:
        # Логика инструмента
        return f"Результат: {param}"
    
    async def _arun(self, param: str) -> str:
        return self._run(param)

# Добавление к агенту
def add_custom_tool(agent: FileSystemAgent):
    custom_tool = CustomTool()
    agent.tools.append(custom_tool)
    agent._analyze_tools()  # Пересчет категорий
```

## Обработка ошибок

### Типы ошибок

```python
# Ошибка инициализации
if not await agent.initialize():
    print("Агент не инициализирован")

# Ошибка обработки сообщения
try:
    response = await agent.process_message("invalid command")
except Exception as e:
    print(f"Ошибка: {e}")

# Проверка готовности агента
if not agent.is_ready:
    print("Агент не готов к работе")
```

### Логирование

```python
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Агент автоматически логирует свои действия
# Логи сохраняются в файл ai_agent.log
```

## Конфигурация MCP серверов

### Стандартная конфигурация

```python
def get_mcp_config(self) -> Dict[str, Any]:
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
```

### Добавление собственного MCP сервера

```python
class CustomAgentConfig(AgentConfig):
    def get_mcp_config(self) -> Dict[str, Any]:
        config = super().get_mcp_config()
        config["custom_server"] = {
            "command": "python",
            "args": ["-m", "my_custom_mcp_server"],
            "transport": "stdio"
        }
        return config
```

## События и хуки

### Перехват обработки сообщений

```python
class CustomAgent(FileSystemAgent):
    async def process_message(self, user_input: str, thread_id: str = "default") -> str:
        # Предобработка
        print(f"Обрабатываем: {user_input}")
        
        # Вызов родительского метода
        response = await super().process_message(user_input, thread_id)
        
        # Постобработка
        print(f"Ответ: {response}")
        
        return response
```

### Кастомизация анализа намерений

```python
class SmartAgent(FileSystemAgent):
    def _analyze_user_intent(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        # Собственная логика анализа
        if "мой файл" in user_input.lower():
            return "read_file", {"target": "my_file.txt"}
        
        # Fallback к стандартной логике
        return super()._analyze_user_intent(user_input)
```

## Тестирование

### Unit тесты

```python
import pytest
from gemini_agent import FileSystemAgent, AgentConfig

@pytest.mark.asyncio
async def test_agent_initialization():
    config = AgentConfig(filesystem_path="/tmp")
    agent = FileSystemAgent(config)
    
    # Мокаем API ключ
    import os
    os.environ["GOOGLE_API_KEY"] = "test_key"
    
    result = await agent.initialize()
    assert result is True
    assert agent.is_ready is True

@pytest.mark.asyncio
async def test_message_processing():
    agent = FileSystemAgent(AgentConfig())
    await agent.initialize()
    
    response = await agent.process_message("покажи файлы")
    assert isinstance(response, str)
    assert len(response) > 0
```

### Интеграционные тесты

```python
@pytest.mark.asyncio
async def test_file_operations():
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = AgentConfig(filesystem_path=temp_dir)
        agent = FileSystemAgent(config)
        await agent.initialize()
        
        # Создание файла
        response = await agent.process_message("создай файл test.txt")
        assert "успешно" in response.lower()
        
        # Проверка существования
        assert os.path.exists(os.path.join(temp_dir, "test.txt"))
        
        # Удаление файла
        response = await agent.process_message("удали test.txt")
        assert "успешно" in response.lower()
        
        # Проверка удаления
        assert not os.path.exists(os.path.join(temp_dir, "test.txt"))
```

---

**Документация обновляется с каждым релизом. Актуальную версию смотрите в [GitHub](https://github.com/maks-mk/smart-gemini-agent).**