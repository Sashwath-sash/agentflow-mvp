from .models import AgentInstance, ExecutionResult, TaskGraph, TaskResult, TaskStatus
from .registry import ToolRegistry


class RuntimeExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def _order(self, graph: TaskGraph) -> list[str]:
        tasks = {task.id: task for task in graph.tasks}
        order: list[str] = []
        while tasks:
            ready = [task_id for task_id, task in tasks.items() if all(dep in order for dep in task.depends_on)]
            if not ready:
                raise ValueError("task graph contains a cycle or unknown dependency")
            order.extend(ready)
            for task_id in ready:
                del tasks[task_id]
        return order

    def execute(self, graph: TaskGraph) -> ExecutionResult:
        by_id = {task.id: task for task in graph.tasks}
        order = self._order(graph)
        results: list[TaskResult] = []
        for task_id in order:
            task = by_id[task_id]
            agent = AgentInstance(role=task.role, objective=task.objective, approved_tools=[tool.name for tool in self.registry.tools_for(task.required_capabilities)])
            results.append(TaskResult(task_id=task.id, status=TaskStatus.COMPLETED, agent=agent, output={"message": "Baseline task accepted for adapter execution.", "dependencies": task.depends_on}))
        return ExecutionResult(objective=graph.objective, order=order, results=results)
