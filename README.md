# Chatbot Service - UCB Commerce

Microservicio de asistencia inteligente basado en **RAG (Retrieval-Augmented Generation)** para la plataforma UCB Commerce. Este servicio actúa como el núcleo cognitivo de la atención al cliente, proporcionando respuestas contextualizadas en tiempo real sobre productos, inventario y procesos institucionales.

## 🚀 Características Principales

- **Arquitectura RAG Avanzada**: Combina la potencia de modelos LLM (Llama 3.1 vía Groq) con una base de conocimiento vectorial dinámica.
- **Sincronización en Tiempo Real**: Mantiene una consistencia estricta con el catálogo de productos. Cualquier cambio en precios, stock o disponibilidad en el `products-service` se refleja instantáneamente en los embeddings del chatbot, garantizando que el asistente nunca ofrezca información obsoleta.
- **Diseño Modular**: Estructurado siguiendo los principios de *Clean Architecture* para facilitar la escalabilidad y el mantenimiento.
- **Base de Conocimiento Híbrida**: Integra información institucional estática (políticas, horarios) con datos transaccionales dinámicos (productos).

## 🛠 Stack Tecnológico

- **Runtime**: Python 3.10+
- **Framework Web**: FastAPI (High performance async framework)
- **Vector Database**: Supabase (pgvector)
- **LLM Inference**: Groq (Llama 3.1-8b-instant)
- **Embeddings**: OpenAI (text-embedding-3-small)

## 📂 Arquitectura del Proyecto

El proyecto ha sido refactorizado para seguir una estructura modular:

```
app/
├── core/
│   └── config.py          # Gestión de configuración y clientes (Singleton pattern)
├── routers/
│   └── chat.py            # Definición de endpoints API (Interface Layer)
├── services/
│   └── rag_service.py     # Lógica de negocio RAG (Chunking, Embedding, Retrieval)
└── main.py                # Punto de entrada y composición de la aplicación
```

## 🔄 Flujo de Datos y Consistencia

1.  **Ingesta de Datos**:
    - Documentos institucionales se cargan vía endpoint `/upload`.
    - **Productos**: Se sincronizan automáticamente desde el `products-service` mediante hooks de eventos.
2.  **Generación de Embeddings**: Se utiliza `text-embedding-3-small` para vectorizar la información.
3.  **Almacenamiento**: Los vectores se persisten en la tabla `rag_ucbcommerce_chunks` de Supabase.
4.  **Recuperación (Retrieval)**:
    - Ante una consulta del usuario, se genera el embedding de la pregunta.
    - Se realiza una búsqueda de similitud semántica (cosine similarity) en Supabase.
5.  **Generación (Generation)**:
    - El contexto recuperado se inyecta en el prompt del sistema.
    - El LLM genera una respuesta precisa basada estrictamente en la evidencia proporcionada.

## 🚀 Instalación y Despliegue

### Prerrequisitos

- Python 3.10+
- Acceso a Supabase (con extensión `vector` habilitada)
- API Keys de OpenAI y Groq

### Pasos

1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuración de Entorno (`.env`):**
    ```env
    OPENAI_API_KEY=sk-...
    GROQ_API_KEY=gsk_...
    SUPABASE_URL=https://...
    SUPABASE_SERVICE_ROLE_KEY=...
    ```

3.  **Ejecutar el servicio:**
    ```bash
    uvicorn app.main:app --reload --port 8004
    ```

## 📡 Endpoints Principales

- `POST /chat`: Endpoint principal de interacción. Acepta `question` y devuelve `answer` + `chunks_used`.
- `POST /upload`: Carga de documentos de texto plano para conocimiento institucional estático.