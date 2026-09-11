<p align="center">
  <h1 align="center">🚀 AGI SDK</h1>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2504.11543">📄 Paper</a> • 
  <a href="https://www.theagi.company/blog/introducing-real-bench">📝 Blog</a> • 
  <a href="https://www.theagi.company">🏢 AGI Inc</a> • 
  <a href="https://www.realevals.xyz">🏆 Leaderboard</a>
</p>


<p align="center">
  <a href="https://pypi.org/project/agisdk"><img src="https://img.shields.io/pypi/v/agisdk?color=brightgreen" alt="PyPI version"></a>
  <a href="https://pypi.org/project/agisdk"><img src="https://img.shields.io/pypi/pyversions/agisdk" alt="Python versions"></a>
  <a href="https://static.pepy.tech/badge/agisdk"><img src="https://static.pepy.tech/badge/agisdk" alt="Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/agi-inc/agisdk" alt="License"></a>
</p>

<p align="center">
  <b>Build, evaluate, and level up your AI agents for the real web.</b>
</p>




# ✨ What is AGI SDK?

**AGI SDK** is a toolkit for **building** and **evaluating** AI browser agents in real-world environments.

It powers [REAL Bench](https://realevals.xyz): the first high-fidelity benchmark for AI agents navigating modern websites like Amazon, DoorDash, Airbnb, and more.

🔹 **Train agents** to browse and interact with real apps  
🔹 **Benchmark agents** with robust, standardized tasks  
🔹 **Submit to the leaderboard** and see how your agents stack up!

> **TL;DR**: Go from “idea” to “benchmarked agent” in <60 seconds

## 🛠️ Installation (30 s)

```bash
# Install the SDK
pip install agisdk

# Install Playwright browser dependencies
playwright install --force

# Set your LLM API key (for evaluation)
export OPENAI_API_KEY="your-api-key"   # any supported provider key works
```

✅ Supports OpenAI, Anthropic, OpenRouter, and custom models! <br>

On Apple Silicon run `brew install --cask playwright` first.


## ⏱️ 60-second Quick-Start

Here's a minimal example to get you started for benchmarking an AI agent on the REAL Bench environment:

```python
from agisdk import REAL

harness = REAL.harness(
    model="gpt-4o",       # any LLM tag
    task_type="omnizon",  # Amazon-like store
    headless=False        # watch it click in real-time!
)

print(harness.run())      # 🎉
```
Need more control? [See full examples ›](/example)

## 🔥 Features

- Full-stack **web replicas** of top real-world apps (Amazon, Uber, Gmail, Airbnb, etc.)
- **Robust agent API**: Observations, Actions, Memory, Errors
- **Leaderboard integration** (REAL Bench)
- **Customizable harness**: plug your own agents
- **Multi-model support**: OpenAI, Anthropic, OpenRouter, or your own model
- **Parallel evaluation** for faster experiments



## Running Custom Agents

> **Note.** The upstream `example/` folder described here is not part of this repo — it
> demonstrated the REAL leaderboard and the `webclones.*` task suite, neither of which this
> benchmark uses. Pass a custom agent to the harness via `agentargs=`; see
> [`src/agisdk/REAL/harness.md`](src/agisdk/REAL/harness.md).

## Local Development

Only if you want to develop locally, you can install from source:

```bash
# Clone the repository
git clone https://github.com/agi-inc/agisdk.git
cd agisdk

# Install in development mode
pip install -e .
```

## 🌐 Available Tasks

The AGI SDK includes high-fidelity, fully-deterministic websites for agents to explore. These are modern web stack sites (React + Next.js) with rich functionality for core user flows, realistic mock data, and consistent behavior for testing and evaluation.

The benchmark includes these environments:

| App Clone | Task Prefix | Example Use Case |
| :--- | :--- | :--- |
| 🛒 Amazon → Omnizon | `webclones.omnizon-*` | Buy a laptop, find a gift |
| 🍔 DoorDash → DashDish | `webclones.dashdish-*` | Order dinner |
| ✈️ United → FlyUnified | `webclones.fly-unified-*` | Book a flight |
| 🏡 Airbnb → Staynb | `webclones.staynb-*` | Reserve accommodation |
| 📅 Google Calendar → GoCalendar | `webclones.gocalendar-*` | Schedule a meeting |
| 📬 Gmail → GoMail | `webclones.gomail-*` | Compose an email |
| 🍽️ OpenTable → OpenDining | `webclones.opendining-*` | Book a restaurant |
| 👔 LinkedIn → NetworkIn | `webclones.networkin-*` | Accept a connection |
| 🚗 Uber → Udriver | `webclones.udriver-*` | Book a ride |
| 💼 UpWork → TopWork | `webclones.topwork-*` | Find a freelance gig |
| 🏠 Zillow → Zilloft | `webclones.zilloft-*` | Browse houses |

Each task comes with **human-written goals** designed to stress-test agent capabilities.

## 🔑 API Keys

To use models from other providers, set their respective API keys:

```bash
# For Anthropic models (like sonnet-3.7)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## 👁️ Observation Structure

Your agent gets access to the following observation structure:

```python
{
    'chat_messages': [...],          # History of chat messages
    'goal': "...",                   # Text description of the goal
    'goal_object': [...],            # Structured goal object with text and images
    'open_pages_urls': [...],        # List of open page URLs
    'active_page_index': 0,          # Index of the active page
    'url': "...",                    # Current URL
    'screenshot': np.array(...),     # Screenshot as numpy array
    'dom_object': {...},             # DOM structure
    'axtree_object': {...},          # Accessibility tree
    'extra_element_properties': {...}, # Additional element properties
    'focused_element_bid': "...",    # ID of the focused element
    'last_action': "...",            # Last action performed
    'last_action_error': "...",      # Error from last action (if any)
    'elapsed_time': 0.0,             # Time elapsed in the episode
    'browser': {...}                 # Playwright browser object (for direct control)
}
```

## 🎯 Actions

Actions are specified as strings in the format of function calls. Here are some commonly used actions:

```python
# Navigation
"goto('https://www.google.com')"
"go_back()"
"go_forward()"

# Interaction
"click('element_id')"
"fill('input_id', 'text to enter')"
"press('Enter')"

# Communication
"send_msg_to_user('I found the answer: $42.99')"

# Reporting infeasible tasks
"report_infeasible('The requested item is out of stock')"
```

## ⚙️ Harness Configuration

The harness function accepts the following parameters:

```python
REAL.harness(
    # Agent configuration (provide one of these)
    model="gpt-4o",                                # OpenAI models
    model="sonnet-3.7",                            # Anthropic models
    model="openrouter/deepseek/deepseek-chat-v3-0324", # OpenRouter models (with openrouter/ prefix)
    agentargs=MyAgentArgs(),                       # Or provide your own agent arguments

    # Task selection (provide one of these or don't provide any to run all tasks)
    task_name="webclones.omnizon-1",  # Specific task to run
    task_type="omnizon",              # Run all tasks of this type
    task_id=1,                        # Run specific task ID within a type

    # Browser configuration
    headless=False,                   # Whether to show the browser
    max_steps=25,                     # Maximum number of steps
    browser_dimensions=(1280, 720),   # Browser window dimensions

    # Observation options
    use_html=False,                   # Include HTML in observations
    use_axtree=True,                  # Include accessibility tree
    use_screenshot=True,              # Include screenshots

    # Leaderboard submission
    leaderboard=False,                # Whether to submit to leaderboard
    run_id="my_unique_id",            # Unique ID for the submission

    # Execution options
    num_workers=4,                    # Number of parallel workers
    use_cache=True,                   # Use cached results when available
    cache_only=False,                 # Only use cached results
    force_refresh=False,              # Force re-running tasks

    # Output options
    results_dir="./results"           # Where to store results
)
```


## 🤝 Contributing

We welcome contributions of all kinds:
- 📢 Feature requests? [Open an Issue](https://github.com/agi-inc/agisdk/issues)
- 🐛 Bug reports? [Create a ticket](https://github.com/agi-inc/agisdk/issues)
- 📈 Improve REAL tasks? Join our [Project Board](https://github.com/orgs/agi-inc/projects/2)
- 🛠️ Submit code? Fork + PR - we love clean commits!

Let's build the future of agents together. 🔥

## 💬 Community
- [Join our Discord](https://discord.gg/c95EJDfXzx) (_coming soon!_)
- [Follow AGI Inc. on LinkedIn](https://www.linkedin.com/company/the-agi-company/)

## ⭐️ Why AGI SDK?

Because **your agents deserve better** than toy environments. <br>
Because **the real web is messy** and that's where the magic happens. <br>
Because **the future is agentic** and it starts here.
