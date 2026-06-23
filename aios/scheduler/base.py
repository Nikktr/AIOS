from abc import ABC, abstractmethod
from threading import Thread
from typing import List, Callable, Dict, Any
import logging
import time

from aios.hooks.types.llm import LLMRequestQueueGetMessage
from aios.hooks.types.memory import MemoryRequestQueueGetMessage
from aios.hooks.types.tool import ToolRequestQueueGetMessage
from aios.hooks.types.storage import StorageRequestQueueGetMessage
from aios.utils.logger import SchedulerLogger
from aios.memory.manager import MemoryManager
from aios.storage.storage import StorageManager
from aios.llm_core.adapter import LLMAdapter
from aios.tool.manager import ToolManager

logger = logging.getLogger(__name__)

class BaseScheduler(ABC):
    """
    Abstract base class for all schedulers in the system.
    
    This class defines the common interface and functionality that all schedulers
    must implement, including request processing and thread management.
    
    Example:
        ```python
        class MyScheduler(BaseScheduler):
            def _process_llm_requests(self):
                # Implementation
                pass
                
            def _process_memory_requests(self):
                # Implementation
                pass
                
            # ... other required methods
        ```
    """
    
    def __init__(
        self,
        llm: LLMAdapter,
        memory_manager: MemoryManager,
        storage_manager: StorageManager,
        tool_manager: ToolManager,
        log_mode: str,
        get_llm_syscall: LLMRequestQueueGetMessage,
        get_memory_syscall: MemoryRequestQueueGetMessage,
        get_storage_syscall: StorageRequestQueueGetMessage,
        get_tool_syscall: ToolRequestQueueGetMessage,
    ):
        """
        Initialize the base scheduler.

        Args:
            llm: LLM adapter instance
            memory_manager: Memory management instance
            storage_manager: Storage management instance
            tool_manager: Tool management instance
            log_mode: Logging mode configuration
            get_llm_syscall: Function to get LLM syscalls
            get_memory_syscall: Function to get Memory syscalls
            get_storage_syscall: Function to get Storage syscalls
            get_tool_syscall: Function to get Tool syscalls
        """
        self.llm = llm
        self.memory_manager = memory_manager
        self.storage_manager = storage_manager
        self.tool_manager = tool_manager
        
        self.get_llm_syscall = get_llm_syscall
        self.get_memory_syscall = get_memory_syscall
        self.get_storage_syscall = get_storage_syscall
        self.get_tool_syscall = get_tool_syscall
        
        self.active = False
        self.log_mode = log_mode
        self.logger = self._setup_logger()

        self.processing_threads: Dict[str, Thread] = {}
        self._processors: Dict[str, Callable] = {}
        self._supervisor_thread: Thread | None = None
        self._restart_counts: Dict[str, int] = {}

    SUPERVISOR_INTERVAL = 5  # seconds between health checks
    MAX_RESTARTS = 10  # per thread before giving up

    def _setup_logger(self) -> SchedulerLogger:
        return SchedulerLogger(self.__class__.__name__, self.log_mode)

    def _guarded_run(self, name: str, fn: Callable) -> None:
        try:
            fn()
        except Exception:
            logger.exception("Scheduler thread '%s' crashed", name)

    def _start_worker(self, name: str, fn: Callable) -> None:
        thread = Thread(
            target=self._guarded_run, args=(name, fn),
            name=name, daemon=True,
        )
        self.processing_threads[name] = thread
        thread.start()

    def _supervisor(self) -> None:
        while self.active:
            time.sleep(self.SUPERVISOR_INTERVAL)
            for name, fn in list(self._processors.items()):
                thread = self.processing_threads.get(name)
                if thread and thread.is_alive():
                    continue
                count = self._restart_counts.get(name, 0)
                if count >= self.MAX_RESTARTS:
                    logger.error(
                        "Scheduler thread '%s' exceeded %d restarts, giving up",
                        name, self.MAX_RESTARTS,
                    )
                    continue
                self._restart_counts[name] = count + 1
                logger.warning(
                    "Scheduler thread '%s' is dead, restarting (%d/%d)",
                    name, count + 1, self.MAX_RESTARTS,
                )
                self._start_worker(name, fn)

    def start_processing_threads(self, processors: List[Callable]) -> None:
        for processor in processors:
            name = processor.__name__
            self._processors[name] = processor
            self._restart_counts[name] = 0
            self._start_worker(name, processor)

        self._supervisor_thread = Thread(
            target=self._supervisor, name="scheduler_supervisor", daemon=True,
        )
        self._supervisor_thread.start()
        logger.info(
            "Scheduler supervisor started, watching %d threads",
            len(self._processors),
        )

    def stop_processing_threads(self) -> None:
        self.active = False
        if self._supervisor_thread:
            self._supervisor_thread.join(timeout=10)
        for thread in self.processing_threads.values():
            thread.join(timeout=5)
        self.processing_threads.clear()
        self._processors.clear()

    @abstractmethod
    def process_llm_requests(self) -> None:
        """Process LLM requests from the queue."""
        pass
    
    @abstractmethod
    def process_memory_requests(self) -> None:
        """Process Memory requests from the queue."""
        pass
    
    @abstractmethod
    def process_storage_requests(self) -> None:
        """Process Storage requests from the queue."""
        pass

    @abstractmethod
    def process_tool_requests(self) -> None:
        """Process Tool requests from the queue."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the scheduler."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the scheduler."""
        pass
