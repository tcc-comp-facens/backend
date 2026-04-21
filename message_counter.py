"""
Contador atômico de mensagens entre agentes.

Contabiliza chamadas de método entre agentes em cada topologia para
comparação quantitativa do overhead de comunicação entre Estrela e
Hierárquica. Cada chamada de método conta como 2 mensagens (ida + volta).

O MessageCounter é instanciado por análise e passado ao orquestrador/coordenador.
O total é persistido no Neo4j e enviado via WebSocket como evento `metric`.

Requisitos: 11.1, 11.2
"""

from __future__ import annotations

import threading


class MessageCounter:
    """Contador atômico de mensagens entre agentes.

    Thread-safe via ``threading.Lock``. Cada interação entre agentes
    (chamada de método) incrementa o contador por *n* (default 2: ida + volta).

    Exemplo de uso::

        counter = MessageCounter()
        counter.increment()      # +2 (ida + volta)
        counter.increment(1)     # +1 (apenas ida)
        assert counter.count == 3
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._lock: threading.Lock = threading.Lock()

    def increment(self, n: int = 2) -> None:
        """Incrementa o contador por *n* (default 2: ida + volta).

        Args:
            n: Número de mensagens a incrementar. Default é 2,
               representando uma chamada de ida e uma de volta.

        Raises:
            ValueError: Se *n* for negativo.
        """
        if n < 0:
            raise ValueError(f"Increment value must be non-negative, got {n}")
        with self._lock:
            self._count += n

    @property
    def count(self) -> int:
        """Retorna o total de mensagens contabilizadas (thread-safe)."""
        with self._lock:
            return self._count
