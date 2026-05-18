# 🏢 АХО Бот — Корпоративный AI-ассистент

AI-бот для автоматизации заявок в административно-хозяйственный отдел (АХО). Поддерживает 6 сценариев обслуживания через естественный диалог с сотрудниками.

---

## Архитектура

### Общая схема

```
Пользователь (Web UI)
        │
        ▼
   FastAPI REST API
        │
        ▼
┌─────────────────────────────────────────┐
│         LangGraph Orchestrator          │
│                                         │
│   ┌────────────┐                        │
│   │   Router    │──── классификация ──►  │
│   │   Agent     │      intent           │
│   └─────┬──────┘                        │
│         │ делегирование                 │
│         ▼                               │
│   ┌────────────────┐   ┌────────────┐   │
│   │  Specialist    │◄──│    RAG     │   │
│   │  Agent (1/6)   │   │ (ChromaDB) │   │
│   └───────┬────────┘   └────────────┘   │
│           │ submit                      │
│           ▼                             │
│   ┌────────────────┐                    │
│   │  MCP Tool      │                    │
│   │  Execution     │                    │
│   └────────────────┘                    │
└─────────────────────────────────────────┘
        │
        ▼
   SQLite (sessions, history, requests)
```

### Multi-Agent система

Архитектура построена на разделении ответственности между агентами:

**Router Agent** принимает сообщение пользователя, классифицирует intent (один из 6 сценариев или unknown) и делегирует запрос специализированному агенту. Также обрабатывает мета-команды: подтверждение, отмену и запрос на редактирование заявки.

**6 Specialist Agents** — каждый отвечает за свой домен, имеет собственный системный промпт, набор обязательных полей и MCP-инструменты:

| Агент | Домен | MCP Tools | Поля |
|-------|-------|-----------|------|
| office_supplies | Канцтовары и продукты | `place_order_komus`, `place_order_vkusvill` | items, delivery_office |
| sim_card | Корпоративные SIM-карты | `request_sim_activation` (МТС) | employee_name, department, manager, sim_type |
| business_trip | Командировки | `book_hotel` (Ostrovok.ru), `book_transfer` | city, check_in, check_out |
| parking_pass | Парковочные пропуска | `issue_parking_pass` | car_plate, car_brand, dates |
| taxi | Корпоративное такси | `order_taxi` (Яндекс Go) | pickup, destination, time |
| incident | Инциденты «Непорядок!» | `create_incident_ticket` | category, description, location |

### Жизненный цикл заявки

```
greeting → classify (Router) → collect (Specialist) ⟷ edit → confirm → submit (Tool) → done
                                                                          ↓
                                                                      cancel → done
```

1. Router Agent классифицирует intent и извлекает начальные сущности
2. Specialist Agent собирает недостающие поля через диалог
3. При наличии всех полей — генерирует сводку для подтверждения
4. Пользователь может подтвердить, отредактировать или отменить
5. При подтверждении — вызывается MCP-инструмент, заявка сохраняется

---

## Стек

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| **LLM** | Google Gemini 2.5 Flash (через OpenRouter) | Классификация, извлечение полей, генерация ответов |
| **Оркестрация** | LangGraph | Граф состояний: Router → Specialist → Submit |
| **Backend** | FastAPI + Uvicorn | REST API, WebSocket |
| **RAG** | ChromaDB (all-MiniLM-L6-v2) | Поиск по корпоративным регламентам и FAQ |
| **Guardrails** | Pydantic v2 | Валидация LLM output и финальных заявок |
| **Persistence** | SQLite | Сессии, история чата, заявки |
| **Frontend** | Vanilla HTML/CSS/JS | Чат-интерфейс + debug-панель |
| **Контейнеризация** | Docker | Единый контейнер |
| **CI/CD** | GitHub Actions | Lint, тесты, Docker build |
| **Тестирование** | pytest | Unit-тесты + eval-сценарии |

---

## Как запускать

### Docker (рекомендуется)

```bash
# 1. Клонировать
git clone <repo> && cd aho-bot

# 2. Создать .env
cp .env.example .env
# Вписать API ключ OpenRouter

# 3. Запустить
docker-compose up --build

# 4. Открыть
# http://localhost:8000
```

### Без Docker

```bash
# 1. Установить зависимости (Python 3.11+)
pip install -r requirements.txt

# 2. Создать .env
cp .env.example .env
# Вписать API ключ OpenRouter

# 3. Запустить
python -m uvicorn app.main:app --reload --port 8000
```

### Переменные окружения

| Переменная | Значение | Описание |
|-----------|----------|----------|
| `OPENAI_API_KEY` | `sk-or-v1-...` | API ключ OpenRouter |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | Base URL провайдера |
| `MODEL_NAME` | `google/gemini-2.5-flash` | Модель LLM |

---

## Как тестировать

### Unit-тесты (без LLM, быстрые)

```bash
pytest tests/test_core.py -v
```

17 тестов: Pydantic-схемы, SQLite storage (save/load/delete сессий, история чата), все 8 MCP-инструментов, guardrails (валидация router output, collect output, JSON parsing с markdown-обёрткой).

### Eval-тесты (с LLM, нужен API ключ)

```bash
pytest tests/test_evals.py -v
```

15 сценариев:
- **Intent classification** — 13 тестов: по 2 формулировки на каждый из 6 сценариев + 1 edge case
- **Full dialog flows** — полный цикл от сообщения до submit
- **RAG verification** — проверка что для командировок подтягивается контекст из базы знаний
- **Guardrails** — out-of-scope запросы и пустые сообщения не ломают систему
- **Tool execution** — прямой вызов mock-инструментов

### Только tool-тесты (без LLM)

```bash
pytest tests/test_evals.py -v -k "TestToolExecution"
```

---

## Принятые решения

### Почему OpenRouter, а не прямой OpenAI API

OpenRouter даёт доступ к десяткам моделей через единый API. Gemini 2.5 Flash выбран за хорошее соотношение качество/скорость/цена для задач классификации и извлечения сущностей на русском языке. При необходимости модель меняется одной переменной окружения.

### Почему LangGraph, а не чистый код

LangGraph даёт декларативный граф состояний с условными переходами. Это упрощает отладку (видно какой узел вызвался и почему), делает архитектуру расширяемой (добавить нового агента = добавить узел) и соответствует промышленным практикам оркестрации LLM-агентов.

### Почему SQLite, а не Redis/PostgreSQL

Для прототипа с одним контейнером SQLite — оптимальный выбор: zero-config, файловая БД, хватает для demo-нагрузки. В продакшне замена на PostgreSQL потребует минимальных изменений в storage.py.

### Почему mock MCP tools, а не реальные API

Реальных API Комуса, МТС, Ostrovok в открытом доступе нет. Mock-tools архитектурно идентичны реальным: принимают структурированные аргументы, возвращают результат с ID. Замена на реальные MCP-серверы — замена одной функции в `mcp_tools.py`.

### Почему ChromaDB in-memory, а не persistent

Для demo-сценария база знаний небольшая (2 markdown-файла, ~120 строк). In-memory ChromaDB загружается за секунду и не требует отдельного процесса. При масштабировании — переход на persistent mode или внешний vector store.

### Почему RAG-запрос по intent+keywords, а не по сырому сообщению

Пользователь пишет «Еду в Питер» — в этой фразе нет слов «лимит», «суточные», «проживание». Если искать по сырому сообщению, релевантные регламенты не находятся. Вместо этого запрос обогащается доменными ключевыми словами на основе определённого intent, что существенно повышает recall.

### Почему Pydantic guardrails на трёх уровнях

1. **Router output** — валидация что agent ∈ допустимых, confidence ∈ [0,1]
2. **Collect output** — валидация структуры extracted_fields
3. **Final request** — валидация полной заявки через Pydantic-модель (типы полей, обязательные значения)

Это защита от галлюцинаций LLM: если модель вернёт невалидный intent или сломанный JSON, система корректно обработает ошибку вместо crash.

### Почему debug-панель, а не Langfuse/LangSmith

Для demo-видео встроенная trace-панель нагляднее: все шаги видны в реальном времени рядом с чатом. Подключение Langfuse — вопрос одного middleware, но для тестового задания не даёт дополнительной ценности.

---

## Ограничения системы

### Не реализовано

| Фича | Причина | Сложность добавления |
|-------|---------|---------------------|
| **Streaming responses** | Требует переработки pipeline: SSE от LLM → LangGraph → WebSocket → UI | Средняя (3-4 часа) |
| **Structured outputs** | LLM возвращает JSON через промпт-инструкцию, а не через native `response_format` / tool schema | Низкая (1 час) |
| **Hallucination mitigation** | Нет отдельного механизма (fact-checking, grounding score). RAG + guardrails частично покрывают | Средняя |
| **Prompt versioning** | Есть `PROMPT_VERSION` в коде, но нет системы хранения/A/B/rollback | Низкая |
| **Retry/fallback** | Нет retry при ошибках LLM, нет fallback-модели | Низкая (30 мин) |
| **Rate limiting** | Нет ограничения частоты запросов | Низкая |
| **Async LLM calls** | FastAPI async, но LLM вызовы синхронные (`invoke` вместо `ainvoke`) | Низкая (1 час) |
| **Observability** | Debug trace panel вместо Langfuse/LangSmith/OpenTelemetry | Средняя |
| **UX: статус заявки** | Бэкенд готов (`/api/requests`), нет UI. Чистый CRUD-фронтенд | Низкая |
| **UX: редактирование** | Редактирование через диалог работает. Нет формы для правки полей | Низкая |

### Технические ограничения

- **Однопоточность** — SQLite не поддерживает конкурентную запись. При >10 одновременных пользователей нужен переход на PostgreSQL.
- **Контекст LLM** — история диалога передаётся целиком. При длинных разговорах (>50 сообщений) может превысить контекстное окно модели.
- **RAG-база** — 2 документа, ~30 чанков. Не масштабируется на сотни регламентов без оптимизации (фильтрация по metadata, re-ranking).
- **MCP tools — заглушки** — возвращают mock-данные. Архитектура готова к замене на реальные MCP-серверы.
- **Один LLM-провайдер** — нет fallback на другую модель при недоступности OpenRouter.

---

## Структура проекта

```
aho-bot/
├── app/
│   ├── agents/
│   │   ├── graph.py            # LangGraph: Router → Specialists → Tools
│   │   └── prompts.py          # Промпты v2.0 (router, agents, collect, confirm)
│   ├── models/
│   │   └── schemas.py          # Pydantic-модели заявок + guardrails
│   ├── services/
│   │   ├── storage.py          # SQLite (sessions, chat history, requests)
│   │   ├── rag.py              # ChromaDB knowledge base
│   │   └── employees.py        # Справочник сотрудников
│   ├── tools/
│   │   └── mcp_tools.py        # 8 mock MCP-инструментов
│   ├── config.py               # Settings (env vars)
│   └── main.py                 # FastAPI endpoints
├── data/
│   ├── knowledge/
│   │   ├── regulations.md      # Корпоративные регламенты АХО
│   │   └── faq.md              # FAQ
│   └── employees.json          # 15 тестовых сотрудников
├── tests/
│   ├── test_core.py            # 17 unit-тестов
│   └── test_evals.py           # 15 eval-сценариев
├── web/
│   └── index.html              # SPA: чат + debug-панель
├── .github/workflows/ci.yml    # CI/CD pipeline
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
