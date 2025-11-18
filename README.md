# Podcast creation with the Google Agent Development Kit (ADK)

This project creates a two-person podcast that talks about the latest Artificial Intelligence (AI) news.

It generates 3 files similar to these examples:<br/>
**`EXAMPLE_ai_news_report.md`**: An AI news report in Markdown format<br/>
**`EXAMPLE_ai_podcast_script.md`**: The script for the podcast in Markdown format<br/>
**`EXAMPLE_ai_today_podcast.wav`**: The podcast audio file that gets created from the script

To keep it simple, this project only generates an AI news podcast. To make it multi-functional, you would have to put a "router/general assistant" agent in front of the existing AI news (root) agent, and tell the Google ADK (via system instructions and tool/agent descriptions) to only route to the AI news part when the user actually asks for AI news. Everything else would be answered directly by the general assistant. Alternatively, you could also package the AI news agent as a function tool and give it to "router/general assistant" agent. The Google ADK allows for an agent to be used as a tool by another agent.

## Google ADK

Google's ADK is a powerful framework for building AI agents that can plan, reason, and use tools, with strong integration into Google's Gemini Large Language Models (LLMs) and cloud services. It is a strong choice for developers who want a polished, scalable, and Google-native way to build advanced agents.

### Callbacks

Callbacks in the Google ADK are Python functions that run at various checkpoints of an agent. Callbacks let you monitor and control what your agent does at key moments without changing its core code.

Callbacks unlock significant flexibility and enable advanced agent capabilities:<br/>
**Observe & Debug**: Log detailed information at critical steps for monitoring and troubleshooting<br/>
**Customise & Control**: Modify data flowing through the agent (for example LLM requests or tool results), or even bypass certain steps entirely based on your logic<br/>
**Implement Guardrails**: Enforce safety rules, validate inputs/outputs, or prevent disallowed operations<br/>
**Manage State**: Read or dynamically update the agent's session state during execution<br/>
**Integrate & Enhance**: Trigger external actions (API calls, notifications), or add features like caching

The Google ADK provides several callback points:<br/>
**`before_agent_callback`**: Runs before agent execution starts<br/>
**`after_agent_callback`**: Runs after agent execution completes<br/>
**`before_tool_callback`**: Runs before any tool is executed<br/>
**`after_tool_callback`**: Runs after any tool completes<br/>
**`before_model_callback`**: Runs before LLM calls<br/>
**`after_model_callback`**: Runs after LLM responses

![alt text](https://github.com/user-attachments/assets/c0850e7c-c14a-47dc-a58e-b4102e0c7f03 "Google ADK Callback Flow")

## Google Cloud setup

To access all Google Gemini models for this example, you must:

1. [Create a Google account.](https://accounts.google.com/signup) When you are done, you will have a Google account that you can use for all Google services, including Google Cloud.
2. [Sign up for Google Cloud.](https://cloud.google.com/free) You must enter your credit card details to get your free $300 trial credit for new customers. Without enabling billing through a credit card, your choice of models and token usage will be limited.
3. Create your first project in the Google Cloud.
4. Create an API key for your project.
5. Enter your Google API key in the `.env.example` file and rename this file to just `.env` (remove the ".example" ending).

## Running the agent through the Google ADK web interface

The Google ADK web interface is a convenient way to trace your agent and interact through live voice conversations and text.

1. In your terminal, start the Google ADK web interface with:

   ```
   adk web --host localhost --port 8003
   ```

2. Open your web browser and visit [http://localhost:8003](http://localhost:8003).

3. Choose one of the following two possible input modes: Voice (a) or Text (b):<br/>
   (a) **Voice input**: Make sure that you switch on the microphone with the `Use Microphone` button.<br/>
   Wait a few seconds before you say the following command in the microphone:

   ```
   Get the latest AI news and create a podcast
   ```

   (b) **Text input**: Type this message and press the `<ENTER>` key:

   ```
   Get the latest AI news and create a podcast
   ```

4. The agent automatically creates a podcast. This takes multiple minutes. The agent creates 3 files:<br/>
   (1) `ai_news_report.md` is a report in Markdown format<br/>
   (2) `ai_podcast_script.md` is the script for the podcast in Markdown format. The agent uses this script to generate the podcast audio file.<br/>
   (3) `ai_today_podcast.wav` is the podcast audio file
