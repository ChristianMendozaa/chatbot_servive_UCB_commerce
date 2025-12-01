# Chatbot Service - UCB Commerce

Microservicio de asistencia inteligente (Chatbot) para la plataforma UCB Commerce.

## Descripción

Este servicio implementa un sistema RAG (Retrieval-Augmented Generation) para responder preguntas de los usuarios sobre la plataforma, productos o procesos de compra. Utiliza una base de conocimiento local para proporcionar respuestas contextualizadas y precisas.

## Tecnologías

- **Lenguaje:** Python 3.10+
- **Framework:** FastAPI (asumido por estructura)
- **IA/ML:** Integración con LLMs (posiblemente OpenAI o Gemini, según configuración)
- **Base de Conocimiento:** Archivo de texto plano (`UCB Commerce RAG.txt`) para contexto.

## Funcionalidades

- **Endpoint de Chat:** Recibe preguntas de los usuarios y devuelve respuestas generadas por IA.
- **Contextualización:** Utiliza información específica de UCB Commerce para responder dudas sobre horarios, ubicación, métodos de pago, etc.

## Estructura del Proyecto

```
app/
└── main.py     # Punto de entrada y lógica del servicio
UCB Commerce RAG.txt # Base de conocimiento
```

## Instalación y Ejecución

1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configurar variables de entorno:**
    Configurar `.env` con las API Keys necesarias (ej. OpenAI API Key).

3.  **Ejecutar el servidor:**
    ```bash
    uvicorn app.main:app --reload --port 8004
    ```