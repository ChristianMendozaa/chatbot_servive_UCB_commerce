import json
import re
from typing import Dict, Any, List
from app.core.config import groq_client
from app.core.tools import (
    rag_search_tool,
    add_to_cart_tool,
    remove_from_cart_tool,
    clear_cart_tool,
    create_order_tool,
    navigate_tool
)

# System prompt for the ReAct agent
SYSTEM_PROMPT = """
Eres un asistente inteligente para el e-commerce de la Universidad Católica Boliviana (UCB).
Tu objetivo es ayudar a los usuarios a encontrar productos, gestionar su carrito y realizar pedidos.

Tienes acceso a las siguientes herramientas:

1. rag_search_tool(query: str): Busca información sobre productos, precios, stock o la universidad. Úsala para responder preguntas generales.
2. add_to_cart_tool(product_id: str, quantity: int): Agrega un producto al carrito. Requiere el ID del producto.
3. remove_from_cart_tool(product_id: str): Elimina un producto del carrito. Requiere el ID del producto.
4. clear_cart_tool(): Vacía todo el carrito.
5. create_order_tool(): Crea un pedido con los items actuales del carrito.
6. navigate_tool(product_id: str): Genera un comando para llevar al usuario a la página de detalle de un producto. Úsala cuando el usuario quiera "ver" o "ir a" un producto específico.

FORMATO DE RESPUESTA (ReAct):
Debes razonar paso a paso antes de actuar. Usa el siguiente formato estrictamente:

Thought: Tu razonamiento sobre qué hacer.
Action: El nombre de la herramienta a usar (una de las lista de arriba).
Action Input: Los argumentos para la herramienta en formato JSON.
Observation: El resultado de la herramienta (esto lo generaré yo).
... (puedes repetir Thought/Action/Observation N veces)
Thought: Ya tengo suficiente información.
Final Answer: Tu respuesta final al usuario.

Si la respuesta final incluye un comando de navegación, asegúrate de mencionarlo.
Si no necesitas usar herramientas, puedes ir directo a Final Answer.

Ejemplo:
User: Quiero comprar 2 poleras UCB (ID: 123)
Thought: El usuario quiere agregar un producto al carrito.
Action: add_to_cart_tool
Action Input: {"product_id": "123", "quantity": 2}
Observation: Producto agregado al carrito exitosamente.
Thought: Ya agregué el producto.
Final Answer: He agregado 2 poleras UCB a tu carrito.

IMPORTANTE:
- Los IDs de productos son cadenas alfanuméricas largas (ej: "ne8jwGSSjCqzPXRLzq8r").
- NUNCA inventes un ID. NUNCA uses el nombre del producto como ID (ej: "ID de la mochila" es INCORRECTO).
- Si el usuario te pide una acción sobre un producto (agregar, navegar, etc.) y NO tienes el ID exacto en tu contexto actual, DEBES usar `rag_search_tool` primero para buscar el producto y obtener su ID.
- Solo cuando tengas el ID real (ej: "ne8jwGSSjCqzPXRLzq8r"), ejecuta la herramienta correspondiente.
- Siempre responde en español y sé amable.
- Si una herramienta devuelve un error (ej: "Error creando el pedido"), NO intentes ejecutar la misma acción inmediatamente de nuevo. En su lugar, informa al usuario sobre el error y sugiere una solución o pide más detalles.
- Si el usuario dice "quiero comprar X" o "hazme un pedido de X", PRIMERO debes usar `add_to_cart_tool` para agregar X al carrito, y LUEGO usar `create_order_tool`. No puedes crear un pedido si el carrito está vacío o si el producto no ha sido agregado.

ESTILO DE RESPUESTA:
- Cuando uses `navigate_tool`, NO digas "haz clic en el enlace". Di algo como "Te he redirigido a la página del producto..." o "Ya estás en la página del producto...". El sistema lo hace automáticamente.
- Sé PROACTIVO. No digas "puedes agregarlo al carrito". OFRECE hacerlo tú. Di: "¿Quieres que lo agregue a tu carrito por ti?" o "¿Te gustaría que proceda con el pedido?".
- Recuerda que tienes superpoderes: puedes agregar/quitar items, vaciar el carrito y crear pedidos. ¡Úsalos!
"""

async def run_agent(question: str, cookies: Dict[str, str] = None, history: List[Dict[str, str]] = []) -> Dict[str, Any]:
    """
    Ejecuta el bucle ReAct del agente.
    """
    # Construir historial previo
    formatted_history = []
    for msg in history:
        role = "user" if msg.get("sender") == "user" else "assistant"
        content = msg.get("text", "")
        formatted_history.append({"role": role, "content": content})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *formatted_history,
        {"role": "user", "content": question}
    ]

    max_steps = 8
    current_step = 0

    navigation_command = None

    while current_step < max_steps:
        # 1. Llamar al LLM
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.1,
            stop=["Observation:"] # Detenerse antes de alucinar la observación
        )
        
        response_text = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": response_text})
        
        # 2. Parsear la respuesta
        # Buscamos "Action:" y "Action Input:"
        action_match = re.search(r"Action:\s*(\w+)", response_text)
        
        if not action_match:
            # Si no hay acción, asumimos que es la respuesta final o que el modelo terminó
            # Buscamos "Final Answer:"
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
            answer = ""
            if final_answer_match:
                answer = final_answer_match.group(1).strip()
            else:
                answer = response_text.strip()
            
            # Si hubo navegación, la adjuntamos para que el frontend la procese
            if navigation_command and navigation_command not in answer:
                answer += f"\n{navigation_command}"
            
            return {"answer": answer}

        action_name = action_match.group(1)
        
        # Parsear input
        input_match = re.search(r"Action Input:\s*(\{.*\})", response_text, re.DOTALL)
        if not input_match:
             # Si falla el parseo, intentamos recuperar o devolvemos error al modelo
             observation = "Error: No encontré Action Input válido en JSON."
        else:
            try:
                action_input = json.loads(input_match.group(1))
                
                # 3. Ejecutar herramienta
                observation = await execute_tool(action_name, action_input, cookies)
                
                # Capturar comando de navegación
                if action_name == "navigate_tool":
                    navigation_command = observation
                    
            except json.JSONDecodeError:
                observation = "Error: Action Input no es un JSON válido."
            except Exception as e:
                observation = f"Error ejecutando herramienta: {str(e)}"

        # 4. Agregar observación al historial
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        current_step += 1

    return {"answer": "Lo siento, alcancé el límite de pasos y no pude completar tu solicitud."}

async def execute_tool(name: str, args: Dict[str, Any], cookies: Dict[str, str]) -> str:
    if name == "rag_search_tool":
        return await rag_search_tool(args.get("query"))
    elif name == "add_to_cart_tool":
        return await add_to_cart_tool(args.get("product_id"), args.get("quantity", 1), cookies)
    elif name == "remove_from_cart_tool":
        return await remove_from_cart_tool(args.get("product_id"), cookies)
    elif name == "clear_cart_tool":
        return await clear_cart_tool(cookies)
    elif name == "create_order_tool":
        return await create_order_tool(cookies)
    elif name == "navigate_tool":
        return navigate_tool(args.get("product_id"))
    else:
        return f"Error: Herramienta '{name}' no encontrada."
