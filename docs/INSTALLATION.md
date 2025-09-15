# 🚀 Подробная инструкция по установке

## Системные требования

### Минимальные требования
- **Python**: 3.8 или выше
- **RAM**: 2 GB
- **Диск**: 500 MB свободного места
- **Интернет**: для загрузки зависимостей и работы с Gemini API

### Рекомендуемые требования
- **Python**: 3.10 или выше
- **RAM**: 4 GB или больше
- **Диск**: 1 GB свободного места
- **Терминал**: с поддержкой цветов и Unicode

## Пошаговая установка

### Шаг 1: Установка Python

#### Windows
1. Скачайте Python с [python.org](https://python.org)
2. Запустите установщик
3. ✅ Обязательно отметьте "Add Python to PATH"
4. Выберите "Install Now"

#### macOS
```bash
# Используя Homebrew (рекомендуется)
brew install python

# Или скачайте с python.org
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Шаг 2: Установка Node.js (для MCP серверов)

#### Windows/macOS
Скачайте с [nodejs.org](https://nodejs.org) и установите LTS версию.

#### Linux
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Или используйте snap
sudo snap install node --classic
```

### Шаг 3: Установка uv/uvx (для Python MCP серверов)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Шаг 4: Получение Google API Key

1. Перейдите на [Google AI Studio](https://aistudio.google.com/)
2. Войдите в аккаунт Google
3. Нажмите "Get API Key"
4. Создайте новый проект или выберите существующий
5. Скопируйте API ключ

### Шаг 5: Клонирование репозитория

```bash
git clone https://github.com/maks-mk/smart-gemini-agent.git
cd smart-gemini-agent
```

### Шаг 6: Создание виртуального окружения

```bash
# Создание виртуального окружения
python -m venv venv

# Активация
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Шаг 7: Установка зависимостей

```bash
# Обновление pip
pip install --upgrade pip

# Установка основных зависимостей
pip install -r requirements.txt

# Установка MCP серверов
pip install duckduckgo-mcp-server
pip install mcp-server-fetch
```

### Шаг 8: Настройка переменных окружения

```bash
# Копирование примера конфигурации
cp .env.example .env

# Редактирование .env файла
# Windows
notepad .env

# macOS
open -e .env

# Linux
nano .env
```

Заполните `.env` файл:
```env
GOOGLE_API_KEY=your_actual_api_key_here
FILESYSTEM_PATH=/path/to/your/workspace
GEMINI_MODEL=gemini-2.0-flash
TEMPERATURE=0.0
```

### Шаг 9: Проверка установки

```bash
# Запуск агента
python gemini_agent.py

# Если все работает, вы увидите:
# ╔═══════════════════════════════════════╗
# ║    🧠 Smart Gemini FileSystem Agent   ║
# ╚═══════════════════════════════════════╝
```

## Устранение проблем

### Проблема: "GOOGLE_API_KEY not found"
**Решение:**
1. Убедитесь, что файл `.env` создан
2. Проверьте, что API ключ правильно указан
3. Перезапустите терминал

### Проблема: "ModuleNotFoundError"
**Решение:**
```bash
# Убедитесь, что виртуальное окружение активировано
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Переустановите зависимости
pip install -r requirements.txt
```

### Проблема: "MCP server not found"
**Решение:**
```bash
# Проверьте установку Node.js
node --version

# Проверьте установку uv
uv --version

# Переустановите MCP серверы
pip install --upgrade duckduckgo-mcp-server mcp-server-fetch
```

### Проблема: "Permission denied"
**Решение:**
```bash
# Linux/Mac - добавьте права на выполнение
chmod +x gemini_agent.py

# Windows - запустите как администратор
```

### Проблема: Медленная работа
**Возможные причины:**
1. Медленное интернет-соединение
2. Недостаточно RAM
3. Старая версия Python

**Решения:**
1. Проверьте скорость интернета
2. Закройте другие приложения
3. Обновите Python до версии 3.10+

## Дополнительные настройки

### Настройка рабочей директории
```bash
# В .env файле укажите полный путь
FILESYSTEM_PATH=/home/user/my-workspace

# Или используйте относительный путь
FILESYSTEM_PATH=./workspace
```

### Настройка модели Gemini
```bash
# Доступные модели:
GEMINI_MODEL=gemini-2.0-flash      # Быстрая (рекомендуется)
GEMINI_MODEL=gemini-1.5-pro       # Более мощная
GEMINI_MODEL=gemini-1.5-flash     # Баланс скорости и качества
```

### Настройка температуры
```bash
TEMPERATURE=0.0    # Детерминированные ответы
TEMPERATURE=0.3    # Слегка креативные ответы
TEMPERATURE=0.7    # Креативные ответы
TEMPERATURE=1.0    # Очень креативные ответы
```

## Обновление

### Обновление до новой версии
```bash
# Получение последних изменений
git pull origin main

# Обновление зависимостей
pip install -r requirements.txt --upgrade

# Проверка изменений в .env.example
diff .env .env.example
```

### Откат к предыдущей версии
```bash
# Просмотр доступных версий
git tag

# Переключение на конкретную версию
git checkout v1.0.0

# Возврат к последней версии
git checkout main
```

## Автоматическая установка

### Скрипт для Linux/macOS
```bash
#!/bin/bash
# install.sh

echo "🚀 Установка Smart Gemini Agent..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python не найден. Установите Python 3.8+"
    exit 1
fi

# Клонирование репозитория
git clone https://github.com/maks-mk/smart-gemini-agent.git
cd smart-gemini-agent

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
pip install duckduckgo-mcp-server mcp-server-fetch

# Создание конфигурации
cp .env.example .env

echo "✅ Установка завершена!"
echo "📝 Отредактируйте файл .env и добавьте ваш GOOGLE_API_KEY"
echo "🚀 Запустите: python gemini_agent.py"
```

### Скрипт для Windows (PowerShell)
```powershell
# install.ps1

Write-Host "🚀 Установка Smart Gemini Agent..." -ForegroundColor Green

# Проверка Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python не найден. Установите Python 3.8+" -ForegroundColor Red
    exit 1
}

# Клонирование репозитория
git clone https://github.com/maks-mk/smart-gemini-agent.git
Set-Location smart-gemini-agent

# Создание виртуального окружения
python -m venv venv
.\venv\Scripts\Activate.ps1

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
pip install duckduckgo-mcp-server mcp-server-fetch

# Создание конфигурации
Copy-Item .env.example .env

Write-Host "✅ Установка завершена!" -ForegroundColor Green
Write-Host "📝 Отредактируйте файл .env и добавьте ваш GOOGLE_API_KEY" -ForegroundColor Yellow
Write-Host "🚀 Запустите: python gemini_agent.py" -ForegroundColor Cyan
```

---

**Нужна помощь?** Создайте [Issue](https://github.com/maks-mk/smart-gemini-agent/issues) или обратитесь в [Discussions](https://github.com/maks-mk/smart-gemini-agent/discussions).