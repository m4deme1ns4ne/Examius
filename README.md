# Examius

Examius - это API-сервис, который предоставляет интеллектуальные ответы на вопросы с использованием LLM (Large Language Model) и RAG (Retrieval-Augmented Generation) технологии.

## Особенности

- Использование OpenAI для генерации ответов
- RAG система для улучшения качества ответов
- Сохранение истории диалога
- REST API интерфейс
- Docker поддержка

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/yourusername/examius.git
cd examius
mkdir data_for_rag
```

2. Установите зависимости с помощью Poetry:

```bash
poetry install
```

3. Создайте файл `.env` в корневой директории проекта и добавьте необходимые переменные окружения:

```env
OPENAI_API_KEY=your_api_key_here
PROXY=your_proxy_if_needed
```

## Запуск

### Локальный запуск

```bash
poetry run python main.py
```

### Запуск через Docker

```bash
docker build -t examius .
docker run -p 8000:8000 examius
```

## API Endpoints

### GET /health

Проверка состояния сервиса.

### POST /ask

Получение ответа на вопрос.

Пример запроса:

```json
{
  "question": "Ваш вопрос здесь"
}
```

Пример ответа:

```json
{
  "answer": "Ответ на вопрос",
  "history": ["История диалога"]
}
```

## Структура проекта

```
examius/
├── app/
│   ├── core/
│   ├── models/
│   └── utils/
├── data_for_rag/
├── main.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Разработка

Для форматирования кода используется Black:

```bash
poetry run black .
```

## Документация API

После запуска API, полная документация доступна по следующим URL:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Лицензия

Этот проект лицензирован под лицензией MIT. Подробности можно найти в файле [LICENSE](LICENSE).

## Контакты

- **Имя**: Александр Волжанин
- **Email**: alexandervolzhanin2004@gmail.com
- **GitHub**: [m4deme1ns4ne](https://github.com/m4deme1ns4ne)
