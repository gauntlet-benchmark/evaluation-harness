#!/usr/bin/env python3
"""
Core harness for running agents on browsergym tasks.
Provides a clean, simple interface for running both built-in and custom agents.
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

# Rich logging support
from agisdk.REAL.logging import logger as rich_logger

# Ray imports for distributed execution
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Import the necessary browsergym components
from agisdk.REAL.browsergym.experiments import Agent, AbstractAgentArgs, EnvArgs, ExpArgs, get_exp_result
from agisdk.REAL.demo_agent.basic_agent import DemoAgentArgs

# Ray remote actor for distributed task execution
if RAY_AVAILABLE:
    @ray.remote(resources={"memory_gb": 3})
    def run_task_ray(
        task_name: str,
        agent_args: "AbstractAgentArgs",
        env_args_dict: Dict[str, Any],
        results_dir: str,
        registration_paths: Optional[List[str]] = None,
        save_step_screenshots: bool = False,
        save_step_info: bool = False,
        show_task_completion_summary: bool = True,
        post_run_js_snippet: Optional[str] = None,
        post_run_js_snippet_path: Optional[str] = None,
        post_run_url: Optional[str] = None,
        initial_delay: float = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run a single task."""
        # Import required modules inside the function for Ray workers
        import os
        import time
        import json
        from pathlib import Path
        from agisdk.REAL.browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
        from agisdk.REAL.logging import logger as rich_logger
        from eval.register import register_evaluation_tasks
        
        rich_logger.info(f"Running task: {task_name}")

        if registration_paths:
            register_evaluation_tasks([Path(path) for path in registration_paths])
        
        # Set task name in env args
        env_args_dict["task_name"] = task_name
        
        # Create EnvArgs from dictionary
        env_args = EnvArgs(**env_args_dict)
        
        # Set up experiment
        exp_args = ExpArgs(
            env_args=env_args,
            agent_args=agent_args,
            save_screenshot=save_step_screenshots,
            save_step_info_pkl=save_step_info,
            post_run_js_snippet=post_run_js_snippet,
            post_run_js_snippet_path=post_run_js_snippet_path,
            post_run_url=post_run_url,
            initial_delay=initial_delay,
        )
        
        # Start timing
        start_time = time.time()
        
        # Run experiment
        exp_args.prepare(results_dir)
        
        # Write the run metadata to summary_info.json before running, so a run
        # that crashes still identifies itself.
        summary_info_path = Path(exp_args.exp_dir) / "summary_info.json"

        initial_summary = {
            "task_name": task_name,
            "agent_type": agent_args.agent_name if hasattr(agent_args, "agent_name") else type(agent_args).__name__,
            "model_name": getattr(agent_args, "model_name", "unknown"),
            "max_steps": env_args.max_steps,
            "experiment_status": "started",
        }

        # Write initial summary info
        with open(summary_info_path, "w") as f:
            json.dump(initial_summary, f, indent=4)
        
        # Run the experiment
        exp_args.run()
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get results
        exp_result = get_exp_result(exp_args.exp_dir)
        exp_record = exp_result.get_exp_record()
        
        # Add timing information to the record
        exp_record['elapsed_time'] = elapsed_time
        
        # Add experiment directory to the record
        exp_record['exp_dir'] = str(exp_args.exp_dir)
        
        if show_task_completion_summary:
            # Extract task_id from task_name (e.g. "tc_circuit_001" from "eval.tc_circuit_001")
            task_id = task_name.split('.', 1)[1] if '.' in task_name else task_name
            rich_logger.task_complete(
                errored=bool(exp_record.get("err_msg")),
                time_taken=elapsed_time,
                task_id=task_id,
            )

        return task_name, exp_record

logger = logging.getLogger(__name__)

class harness:
    """
    A simplified harness for running browsergym tasks with various agents.
    """
    
    def __init__(
        self,
        model: str = None,
        agentargs: AbstractAgentArgs = None,
        task_name: str = None,
        headless: bool = True,
        max_steps: int = 25,
        use_html: bool = False,
        use_axtree: bool = True,
        use_screenshot: bool = True,
        browser_dimensions: tuple = (1280, 720),
        golden_user_data_dir: str = None,
        extensions_dir: str = None,
        viewport: dict = None,
        results_dir: str = "./results",
        num_workers: int = 1,
        system_message_handling: str = None,
        system_prompt_append: str = None,
        prefix_prompt: str = None,
        save_step_screenshots: bool = False,
        save_step_info: bool = False,
        show_task_completion_summary: bool = True,
        post_run_js_snippet: str = None,
        post_run_js_snippet_path: str = None,
        post_run_url: str = None,
        initial_delay: float = 0,
        registration_paths: Optional[List[str]] = None,
        seed: Optional[int] = None,
        reasoning: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        thinking_type: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize the harness with the provided configuration.
        
        Args:
            model: Name of the AI model to use (e.g., "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano")
            agentargs: Arguments for a custom agent (if not using a built-in model)
            task_name: Specific task name to run (e.g., "eval.tc_circuit_001")
            headless: Whether to run the browser in headless mode
            max_steps: Maximum number of steps per task
            use_html: Whether to include HTML in observations
            use_axtree: Whether to include accessibility tree in observations
            use_screenshot: Whether to include screenshots in observations
            browser_dimensions: Tuple of (width, height) for browser viewport
            golden_user_data_dir: Path to browser user data directory
            extensions_dir: Path to Chrome extensions directory
            viewport: Dictionary with width and height for browser viewport
            results_dir: Directory to store results
            num_workers: Number of parallel workers (if > 1, uses Ray for distributed execution)
            system_message_handling: How to handle system messages - "separate" (default) or "combined" (no system prompt).
                                   Only applies when using the model parameter. For o1-mini, defaults to "combined".
            system_prompt_append: Optional extra instructions appended to the built-in system prompt.
            prefix_prompt: Optional text prepended before the task goal prompt.
            save_step_screenshots: Whether to save a screenshot for each step in the experiment directory.
            save_step_info: Whether to save per-step state and agent output payloads to the experiment directory.
            show_task_completion_summary: Whether to print a per-task completion line after each task.
            post_run_js_snippet: JavaScript source to execute against the final live page before teardown.
            post_run_js_snippet_path: Original file path for the post-run JavaScript snippet.
            post_run_url: Optional URL to visit after task completion before capturing extra page data and running post-run JS.
            initial_delay: Seconds to wait after page load before the agent takes its first action.
        """
        self.results_dir = results_dir
        self.num_workers = num_workers
        self.save_step_screenshots = save_step_screenshots
        self.save_step_info = save_step_info
        self.show_task_completion_summary = show_task_completion_summary
        self.post_run_js_snippet = post_run_js_snippet
        self.post_run_js_snippet_path = post_run_js_snippet_path
        self.post_run_url = post_run_url
        self.initial_delay = initial_delay
        self.registration_paths = registration_paths or []
        
        logger.info(f"Harness initialized with model={model or 'custom'}, task={task_name}")
        # Initialize agent arguments
        if agentargs is not None:
            if system_message_handling is not None:
                logger.warning("system_message_handling parameter is ignored when using custom agentargs")
            self.agent_args = agentargs
        elif model is not None:
            # Validate system_message_handling parameter if provided
            if system_message_handling is not None:
                valid_values = ["separate", "combined"]
                if system_message_handling not in valid_values:
                    raise ValueError(f"system_message_handling must be one of {valid_values}, got: {system_message_handling}")
            
            # Set system message handling based on parameter or model default
            if system_message_handling is not None:
                use_system_message_handling = system_message_handling
            else:
                # Default to "separate" unless using o1-mini which requires "combined"
                use_system_message_handling = "combined" if "o1-mini" in model.lower() else "separate"
            
            self.agent_args = DemoAgentArgs(
                model_name=model,
                chat_mode=False,
                demo_mode="default",
                use_html=use_html,
                use_axtree=use_axtree,
                use_screenshot=use_screenshot,
                system_message_handling=use_system_message_handling,
                system_prompt_append=system_prompt_append,
                prefix_prompt=prefix_prompt,
                seed=seed,
                reasoning=reasoning,
                reasoning_effort=reasoning_effort,
                thinking_type=thinking_type,
                provider=provider,
            )
        else:
            raise ValueError("Either model or agentargs must be provided")
        
        # Initialize environment arguments
        if viewport is None:
            viewport = {"width": browser_dimensions[0], "height": browser_dimensions[1]}
            
        self.env_args = {
            "task_seed": None,
            "max_steps": max_steps,
            "headless": headless,
            "golden_user_data_dir": golden_user_data_dir,
            "extensions_dir": extensions_dir,
            "viewport": viewport,
        }
        
        self.task_name = task_name

        # Create default results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def run(self, tasks: List[str] = None) -> Dict[str, Any]:
        """
        Run the tasks with the configured agent and environment.

        Args:
            tasks: Optional list of specific task names to run. Defaults to the
                  single `task_name` the harness was configured with.

        Returns:
            Dictionary of results indexed by task name
        """
        if tasks is None and self.task_name:
            tasks = [self.task_name]

        if not tasks:
            raise ValueError("No tasks found to run")

        logger.info(f"Running {len(tasks)} tasks")

        return self._run_tasks(
            tasks=tasks,
            agent_args=self.agent_args,
            env_args_dict=self.env_args,
            results_dir=self.results_dir,
            num_workers=self.num_workers,
            save_step_screenshots=self.save_step_screenshots,
            save_step_info=self.save_step_info,
            show_task_completion_summary=self.show_task_completion_summary,
            post_run_js_snippet=self.post_run_js_snippet,
            post_run_js_snippet_path=self.post_run_js_snippet_path,
            post_run_url=self.post_run_url,
            initial_delay=self.initial_delay,
            registration_paths=self.registration_paths,
        )

    def _run_tasks(
        self,
        tasks: List[str],
        agent_args: AbstractAgentArgs,
        env_args_dict: Dict[str, Any],
        results_dir: str = "./results",
        num_workers: int = 1,
        save_step_screenshots: bool = False,
        save_step_info: bool = False,
        show_task_completion_summary: bool = True,
        post_run_js_snippet: Optional[str] = None,
        post_run_js_snippet_path: Optional[str] = None,
        post_run_url: Optional[str] = None,
        initial_delay: float = 0,
        registration_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run tasks with the provided agent and environment configuration.

        Args:
            tasks: List of task names to run
            agent_args: Arguments for the agent
            env_args_dict: Dictionary of arguments for the environment
            results_dir: Directory to store results
            num_workers: Number of workers (if > 1, uses Ray for distributed execution)

        Returns:
            Dictionary of results indexed by task name
        """
        results = {}

        rich_logger.info(f"🏃 Running {len(tasks)} tasks...")
        rich_logger.info(f"💻 Number of workers configured: {num_workers}")

        if num_workers > 1:
            if not RAY_AVAILABLE:
                raise RuntimeError("Ray is required for parallel execution but not available. Please install Ray with: pip install ray")

            if not ray.is_initialized():
                # Initialize Ray with memory tokens as concurrency limit
                ray.init(resources={"memory_gb": num_workers})

            # Submit all tasks as futures - Ray will queue them based on memory_gb availability
            ray_futures = [
                run_task_ray.remote(
                    task_name=task_name,
                    agent_args=agent_args,
                    env_args_dict=env_args_dict,
                    results_dir=results_dir,
                    registration_paths=registration_paths,
                    save_step_screenshots=save_step_screenshots,
                    save_step_info=save_step_info,
                    show_task_completion_summary=show_task_completion_summary,
                    post_run_js_snippet=post_run_js_snippet,
                    post_run_js_snippet_path=post_run_js_snippet_path,
                    post_run_url=post_run_url,
                    initial_delay=initial_delay,
                )
                for task_name in tasks
            ]
            results.update(dict(ray.get(ray_futures)))
        else:
            for task_name in tasks:
                task_name, exp_record = self._run_single_task(
                    task_name=task_name,
                    agent_args=agent_args,
                    env_args_dict=env_args_dict,
                    results_dir=results_dir,
                    save_step_screenshots=save_step_screenshots,
                    save_step_info=save_step_info,
                    show_task_completion_summary=show_task_completion_summary,
                    post_run_js_snippet=post_run_js_snippet,
                    post_run_js_snippet_path=post_run_js_snippet_path,
                    post_run_url=post_run_url,
                    initial_delay=initial_delay,
                    registration_paths=registration_paths,
                )
                results[task_name] = exp_record

        return results

    def _run_single_task(
        self,
        task_name: str,
        agent_args: AbstractAgentArgs,
        env_args_dict: Dict[str, Any],
        results_dir: str,
        save_step_screenshots: bool = False,
        save_step_info: bool = False,
        show_task_completion_summary: bool = True,
        post_run_js_snippet: Optional[str] = None,
        post_run_js_snippet_path: Optional[str] = None,
        post_run_url: Optional[str] = None,
        initial_delay: float = 0,
        registration_paths: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run a single task with the provided agent and environment configuration.
        
        Args:
            task_name: Name of the task to run
            agent_args: Arguments for the agent
            env_args_dict: Dictionary of arguments for the environment
            results_dir: Directory to store results

        Returns:
            Tuple of (task_name, results_dict)
        """
        print(f"Running task: {task_name}")

        if registration_paths:
            from eval.register import register_evaluation_tasks
            register_evaluation_tasks([Path(path) for path in registration_paths])
        
        # Set task name in env args
        env_args_dict["task_name"] = task_name
        
        # Create EnvArgs from dictionary
        env_args = EnvArgs(**env_args_dict)
        
        # Set up experiment
        model_name = getattr(agent_args, "model_name", "unknown")
        exp_args = ExpArgs(
            env_args=env_args,
            agent_args=agent_args,
            model_name=model_name,
            save_screenshot=save_step_screenshots,
            save_step_info_pkl=save_step_info,
            post_run_js_snippet=post_run_js_snippet,
            post_run_js_snippet_path=post_run_js_snippet_path,
            post_run_url=post_run_url,
            initial_delay=initial_delay,
        )
        
        # Start timing
        start_time = time.time()
        
        # Run experiment
        exp_args.prepare(results_dir)
        
        # Write the run metadata to summary_info.json before running, so a run
        # that crashes still identifies itself.
        summary_info_path = Path(exp_args.exp_dir) / "summary_info.json"

        initial_summary = {
            "task_name": task_name,
            "agent_type": agent_args.agent_name if hasattr(agent_args, "agent_name") else type(agent_args).__name__,
            "model_name": model_name,
            "max_steps": env_args.max_steps,
            "experiment_status": "started",
        }

        # Write initial summary info
        with open(summary_info_path, "w") as f:
            json.dump(initial_summary, f, indent=4)
        
        # Run the experiment
        exp_args.run()
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get results
        exp_result = get_exp_result(exp_args.exp_dir)
        exp_record = exp_result.get_exp_record()
        
        # Add timing information to the record
        exp_record['elapsed_time'] = elapsed_time
        
        # Add experiment directory to the record
        exp_record['exp_dir'] = str(exp_args.exp_dir)
        
        if show_task_completion_summary:
            # Extract task_id from task_name (e.g. "tc_circuit_001" from "eval.tc_circuit_001")
            task_id = task_name.split('.', 1)[1] if '.' in task_name else task_name
            rich_logger.task_complete(
                errored=bool(exp_record.get("err_msg")),
                time_taken=elapsed_time,
                task_id=task_id,
            )

        return task_name, exp_record


# Make AbstractAgentArgs and Agent classes available at the top level for convenient imports
AbstractAgentArgs = AbstractAgentArgs
Agent = Agent
