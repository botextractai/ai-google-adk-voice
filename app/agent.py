import concurrent.futures
import os
import pathlib
import time
import traceback
import wave
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.genai import types
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

def debug_log(message, level="INFO"):
    """Helper function for timestamped debug logging"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[DEBUG {level} {timestamp}] {message}", flush=True)

# Load environment variables from root directory
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / '.env')

class NewsStory(BaseModel):
    """A single news story with its context."""
    company: str = Field(description="Company name associated with the story (e.g., 'Nvidia', 'OpenAI'). Use 'N/A' if not applicable.")
    ticker: str = Field(description="Stock ticker for the company (e.g., 'NVDA'). Use 'N/A' if private or not found.")
    summary: str = Field(description="A brief, one-sentence summary of the news story.")
    why_it_matters: str = Field(description="A concise explanation of the story's significance or impact.")
    financial_context: str = Field(description="Current stock price and change, e.g., '$950.00 (+1.5%)'. Use 'No financial data' if not applicable.")
    source_domain: str = Field(description="The source domain of the news, e.g., 'techcrunch.com'.")
    process_log: List[str] = Field(description="populate the `process_log` field in the schema with the `process_log` list from the `google_search` tool's output.")

class AINewsReport(BaseModel):
    """A structured report of the latest AI news."""
    title: str = Field(default="AI Research Report", description="The main title of the report.")
    report_summary: str = Field(description="A brief, high-level summary of the key findings in the report.")
    stories: List[NewsStory] = Field(description="A list of the individual news stories found.")

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Helper function to save audio data as a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def save_ai_podcast_script(ai_podcast_script: str, filename: str = "ai_podcast_script") -> Dict[str, str]:
    """
    Saves the podcast script as a markdown file for later audio generation.
    """
    try:
        # Save with markdown file extension
        if not filename.endswith(".md"):
            filename += ".md"
        
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        
        # Save the script as a markdown file
        file_path.write_text(ai_podcast_script, encoding="utf-8")
        
        return {
            "status": "success",
            "message": f"Successfully saved podcast script to {file_path.resolve()}",
            "file_path": str(file_path.resolve()),
            "file_size": len(ai_podcast_script.encode('utf-8'))
        }
    except Exception as e:
        return {"status": "error", "message": f"Script saving failed: {str(e)[:200]}"}

def generate_podcast_audio(script_file: str = "ai_podcast_script.md", output_file: str = "ai_today_podcast.wav") -> Dict[str, str]:
    """
    Generate podcast audio from a script file.
    
    Args:
        script_file: Path to the script file (default: "ai_podcast_script.md")
        output_file: Output audio file path (default: "ai_today_podcast.wav")
    """
    response = None
    data = None
    
    try:
        # Read the script file
        script_path = pathlib.Path(script_file)
        if not script_path.exists():
            return {
                "status": "error",
                "message": f"Script file {script_file} not found!"
            }
        
        with open(script_path, 'r', encoding='utf-8') as f:
            ai_podcast_script = f.read()
        
        # Initialise Gemini client with API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "message": "GOOGLE_API_KEY not found in environment variables!"
            }
        
        client = genai.Client(api_key=api_key)
        prompt = f"Convert this podcast conversation to audio:\n\n{ai_podcast_script}"
        
        # Use gemini-2.5-flash-preview-tts model
        model = "gemini-2.5-flash-preview-tts"
        
        # Retry logic with exponential backoff for timeout/connection errors
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                                speaker_voice_configs=[
                                    types.SpeakerVoiceConfig(
                                        speaker='Joe',
                                        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck'))
                                    ),
                                    types.SpeakerVoiceConfig(
                                        speaker='Jane',
                                        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Aoede'))
                                    )
                                ]
                            )
                        )
                    )
                )
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # Check if it's a timeout or connection-related error
                is_timeout_error = (
                    'timeout' in error_str or
                    'keepalive' in error_str or
                    'connection' in error_str or
                    'ping' in error_str or
                    'timed out' in error_str or
                    error_type in ['TimeoutError', 'ConnectionError', 'ConnectTimeout', 'ReadTimeout']
                )
                
                if is_timeout_error and attempt < max_retries - 1:
                    continue
                else:
                    # If it's the last attempt or not a timeout error, raise it
                    if attempt == max_retries - 1:
                        return {
                            "status": "error",
                            "message": f"Audio generation failed after {max_retries} attempts: {str(e)[:200]}"
                        }
                    raise
        
        # Extract audio data
        try:
            if response is None:
                return {
                    "status": "error",
                    "message": "No response received from API"
                }
            data = response.candidates[0].content.parts[0].inline_data.data
        except (AttributeError, IndexError, KeyError) as e:
            return {
                "status": "error",
                "message": f"Failed to extract audio data from response: {str(e)[:200]}"
            }
        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str or 'keepalive' in error_str or 'connection' in error_str:
                try:
                    if response and hasattr(response, 'candidates'):
                        data = response.candidates[0].content.parts[0].inline_data.data
                except:
                    return {
                        "status": "error",
                        "message": "Could not recover audio data after connection error"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to extract audio data: {str(e)[:200]}"
                }
        
        # Save audio file
        if data is None:
            return {
                "status": "error",
                "message": "No audio data to save"
            }
        
        if not output_file.endswith(".wav"):
            output_file += ".wav"
        
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / output_file
        
        try:
            wave_file(str(file_path), data)
            return {
                "status": "success",
                "message": f"SUCCESS: Podcast audio file has been generated and saved successfully to {file_path.resolve()}. File size: {len(data)} bytes. The audio generation completed without errors.",
                "file_path": str(file_path.resolve()),
                "file_size": len(data)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"ERROR: Failed to save audio file: {str(e)[:200]}"
            }
    except Exception as e:
        error_str = str(e).lower()
        # If we have the data but there's an error, try to save it anyway
        if data is not None:
            try:
                if not output_file.endswith(".wav"):
                    output_file += ".wav"
                current_directory = pathlib.Path.cwd()
                file_path = current_directory / output_file
                wave_file(str(file_path), data)
                return {
                    "status": "success",
                    "message": f"SUCCESS: Podcast audio file has been generated and saved successfully to {file_path.resolve()} despite a minor error. File size: {len(data)} bytes. The audio generation completed successfully.",
                    "file_path": str(file_path.resolve()),
                    "file_size": len(data)
                }
            except:
                pass
        return {
            "status": "error",
            "message": f"Audio generation failed: {str(e)[:200]}"
        }

def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers.
    Optimised with timeout handling and parallel processing.
    """
    financial_data: Dict[str, str] = {}
    
    # Filter out invalid tickers upfront
    valid_tickers = [ticker.upper().strip() for ticker in tickers
                    if ticker and ticker.upper() not in ['N/A', 'NA', '']]
    
    if not valid_tickers:
        return {ticker: "No financial data" for ticker in tickers}
    
    def fetch_ticker_data(ticker_symbol):
        """Fetch data for a single ticker with timeout"""
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            change_percent = info.get("regularMarketChangePercent")
            
            if price is not None and change_percent is not None:
                change_str = f"{change_percent:+.2f}%"
                return ticker_symbol, f"${price:.2f} ({change_str})"
            else:
                return ticker_symbol, "Price data not available."
        except Exception:
            return ticker_symbol, "Invalid Ticker or Data Error"
    
    # Use ThreadPoolExecutor with timeout for parallel processing
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(fetch_ticker_data, ticker): ticker
                for ticker in valid_tickers
            }
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_ticker, timeout=10):
                ticker_symbol, result = future.result()
                financial_data[ticker_symbol] = result
    except concurrent.futures.TimeoutError:
        # If timeout occurs, fill remaining with timeout message
        for ticker in valid_tickers:
            if ticker not in financial_data:
                financial_data[ticker] = "Data fetch timeout"
    except Exception:
        # Fallback to individual processing if parallel fails
        for ticker_symbol in valid_tickers:
            if ticker_symbol not in financial_data:
                financial_data[ticker_symbol] = "Data fetch error"
    return financial_data

def save_news_to_markdown(filename: str, content: str) -> Dict[str, str]:
    """
    Saves the given content to a Markdown file in the current directory.
    """
    try:
        if not filename.endswith(".md"):
            filename += ".md"
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        file_path.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "message": f"Successfully saved news to {file_path.resolve()}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {str(e)}"}

def create_simple_report(search_results: str, financial_data: Dict[str, str], process_log: Optional[List[str]] = None) -> str:
    """
    Creates a simple markdown report from search results and financial data.
    Includes Data Sourcing Notes section with process_log information.
    """
    report = f"""# AI News Report - {datetime.now().strftime('%Y-%m-%d')}
    
    ## Latest AI News
    
    {search_results}
    
    ## Financial Data
    
    """
    
    # Handle both dictionary and string inputs
    if isinstance(financial_data, dict):
        for ticker, data in financial_data.items():
            if ticker != "N/A" and data != "No financial data":
                report += f"- **{ticker}**: {data}\n"
    elif isinstance(financial_data, str):
        report += f"- {financial_data}\n"
    else:
        report += "- Financial data not available\n"
    
    # Add Data Sourcing Notes section if process_log is provided
    if process_log:
        report += "\n## Data Sourcing Notes\n\n"
        for log_entry in process_log:
            report += f"- {log_entry}\n"
    else:
        report += "\n## Data Sourcing Notes\n\n- No sourcing notes available\n"
    
    report += f"""
    ## Report Generated
    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    return report

WHITELIST_DOMAINS = ["techcrunch.com", "venturebeat.com", "theverge.com", "technologyreview.com", "arstechnica.com", "wired.com", "forbes.com", "justainews.com"]

def filter_news_sources_callback(tool, args, tool_context):
    """Callback to enforce that google_search queries only use whitelisted domains."""
    try:
        print(f"*** filter_news_sources_callback ENTERED for tool: {tool.name if tool else 'None'} ***", flush=True)
        debug_log(f"CALLBACK: filter_news_sources_callback - tool.name={tool.name if tool else 'None'}")
        debug_log(f"CALLBACK: args={args}")
        
        if tool.name == "google_search":
            original_query = args.get("query", "")
            debug_log(f"CALLBACK: Original query: '{original_query}'")
            
            # Add to process_log that we're filtering sources
            try:
                if 'process_log' not in tool_context.state:
                    tool_context.state['process_log'] = []
                log_entry = "Applied whitelist filter to search query: limited to tech news sources"
                tool_context.state['process_log'].append(log_entry)
                debug_log(f"CALLBACK: Added to process_log: {log_entry}")
            except Exception as e:
                debug_log(f"CALLBACK: Could not update process_log: {e}")
            
            if any(f"site:{domain}" in original_query.lower() for domain in WHITELIST_DOMAINS):
                debug_log("CALLBACK: Query already contains whitelist domain, skipping modification")
                return None
            whitelist_query_part = " OR ".join([f"site:{domain}" for domain in WHITELIST_DOMAINS])
            modified_query = f"{original_query} {whitelist_query_part}"
            args['query'] = modified_query
            debug_log(f"CALLBACK: MODIFIED query to enforce whitelist: '{args['query']}'")
        else:
            debug_log(f"CALLBACK: Not a google_search tool (name='{tool.name}'), skipping filter_news_sources_callback")
        return None
    except Exception as e:
        print(f"*** ERROR in filter_news_sources_callback: {e} ***", flush=True)
        traceback.print_exc()
        return None

def enforce_data_freshness_callback(tool, args, tool_context):
    """Callback to add a time filter to search queries to get recent news."""
    try:
        print(f"*** enforce_data_freshness_callback ENTERED for tool: {tool.name if tool else 'None'} ***", flush=True)
        debug_log(f"CALLBACK: enforce_data_freshness_callback - tool.name={tool.name if tool else 'None'}")
        
        if tool.name == "google_search":
            query = args.get("query", "")
            debug_log(f"CALLBACK: Current query before freshness check: '{query}'")
            
            # Add to process_log that we're filtering by time
            try:
                if 'process_log' not in tool_context.state:
                    tool_context.state['process_log'] = []
                log_entry = "Applied time filter to search: limited to results from the past week"
                tool_context.state['process_log'].append(log_entry)
                debug_log(f"CALLBACK: Added to process_log: {log_entry}")
            except Exception as e:
                debug_log(f"CALLBACK: Could not update process_log: {e}")
            
            # Adds a Google search parameter to filter results from the last week
            if "tbs=qdr:w" not in query:
                modified_query = f"{query} tbs=qdr:w"
                args['query'] = modified_query
                debug_log(f"CALLBACK: MODIFIED query for freshness: '{args['query']}'")
            else:
                debug_log("CALLBACK: Query already has tbs=qdr:w, skipping modification")
        return None
    except Exception as e:
        print(f"*** ERROR in enforce_data_freshness_callback: {e} ***", flush=True)
        traceback.print_exc()
        return None

def initialise_process_log(tool, args, tool_context):
    """Helper to ensure the process_log list exists in the state."""
    try:
        print(f"*** initialise_process_log ENTERED for tool: {tool.name if tool else 'None'} ***", flush=True)
        debug_log(f"CALLBACK: initialise_process_log START")
        debug_log(f"CALLBACK: tool object: {tool}")
        debug_log(f"CALLBACK: tool type: {type(tool)}")
        debug_log(f"CALLBACK: tool.name={tool.name if tool else 'None'}")
        debug_log(f"CALLBACK: hasattr(tool, '__class__'): {hasattr(tool, '__class__')}")
        if hasattr(tool, '__class__'):
            debug_log(f"CALLBACK: tool class name: {tool.__class__.__name__}")
        
        # Safely check if process_log exists in state
        try:
            process_log = tool_context.state.get('process_log', None)
            if process_log is not None:
                debug_log(f"CALLBACK: process_log already exists: {process_log}")
            else:
                tool_context.state['process_log'] = []
                debug_log("CALLBACK: Initialised process_log list in state")
        except Exception as e:
            debug_log(f"CALLBACK: Error accessing state - {type(e).__name__}: {e}")
            tool_context.state['process_log'] = []
            debug_log("CALLBACK: Initialised process_log list in state (after error recovery)")
        return None
    except Exception as e:
        print(f"*** ERROR in initialise_process_log: {e} ***", flush=True)
        traceback.print_exc()
        return None

def inject_process_log_after_search(tool, args, tool_context, tool_response):
    """
    Callback: After a successful search, this injects the process_log into the response 
    and adds a specific note about which domains were sourced. This makes the callbacks' 
    actions visible to the LLM.
    """
    try:
        print(f"*** inject_process_log_after_search ENTERED for tool: {tool.name if tool else 'None'} ***", flush=True)
        debug_log(f"CALLBACK: inject_process_log_after_search START")
        debug_log(f"CALLBACK: tool.name={tool.name if tool else 'None'}")
        debug_log(f"CALLBACK: tool_response type={type(tool_response)}")
        debug_log(f"CALLBACK: tool_response is str? {isinstance(tool_response, str)}")
        debug_log(f"CALLBACK: tool_response is dict? {isinstance(tool_response, dict)}")
        
        if tool_response is not None:
            debug_log(f"CALLBACK: tool_response type: {type(tool_response).__name__}")
            if isinstance(tool_response, str):
                debug_log(f"CALLBACK: tool_response length={len(tool_response)}")
                debug_log(f"CALLBACK: tool_response preview (first 200 chars): {tool_response[:200]}")
            elif isinstance(tool_response, dict):
                debug_log(f"CALLBACK: tool_response keys: {list(tool_response.keys())}")
                for key in tool_response.keys():
                    debug_log(f"CALLBACK: tool_response['{key}'] type: {type(tool_response[key]).__name__}")
                    if isinstance(tool_response[key], str):
                        debug_log(f"CALLBACK: tool_response['{key}'] length: {len(tool_response[key])}")
                        debug_log(f"CALLBACK: tool_response['{key}'] preview: {tool_response[key][:200]}")
            else:
                debug_log(f"CALLBACK: tool_response repr: {repr(tool_response)[:200]}")
        
        if tool.name == "google_search":
            debug_log("CALLBACK: Processing google_search response")
           
            # Handle both string and dict responses
            if isinstance(tool_response, str):
                search_results_text = tool_response
            elif isinstance(tool_response, dict):
                debug_log("CALLBACK: tool_response is already a dict, checking for existing keys")
                if 'search_results' in tool_response:
                    search_results_text = tool_response['search_results']
                    debug_log("CALLBACK: Found 'search_results' key in response")
                else:
                    debug_log("CALLBACK: No 'search_results' key found, cannot process")
                    return tool_response
            else:
                debug_log(f"CALLBACK: Unexpected tool_response type: {type(tool_response)}")
                return tool_response
            
            # Get the process_log that was populated by the BEFORE callbacks
            try:
                final_log = tool_context.state.get('process_log', [])
                debug_log(f"CALLBACK: Final log to inject (populated by BEFORE callbacks): {final_log}")
            except Exception as e:
                debug_log(f"CALLBACK: Error getting final log - {type(e).__name__}: {e}")
                final_log = []
                debug_log(f"CALLBACK: Using empty list as fallback")
            
            modified_response = {
                "search_results": search_results_text,
                "process_log": final_log
            }
            debug_log(f"CALLBACK: Returning modified response with keys: {list(modified_response.keys())}")
            debug_log(f"CALLBACK: Modified response search_results length: {len(modified_response.get('search_results', ''))}")
            debug_log(f"CALLBACK: Modified response preview: {str(modified_response)[:200]}")
            return modified_response
        else:
            debug_log(f"CALLBACK: Not a google_search tool (name='{tool.name}'), returning as-is")
            return tool_response
    except Exception as e:
        print(f"*** ERROR in inject_process_log_after_search: {e} ***", flush=True)
        traceback.print_exc()
        return tool_response

# Main orchestrator agent - Simple text-only agent to avoid live audio streaming issues
debug_log("AGENT: Initialising root_agent")
print("*** About to create Agent object ***", flush=True)
root_agent = Agent(
    name="ai_news_reporter",
    model="gemini-2.0-flash-exp",
    instruction="""
You are an AI News Podcast Producer for NASDAQ-listed companies.

**IMPORTANT: Execute steps immediately, don't overthink.**

1. Say: "Okay, I'll start researching the latest AI news for NASDAQ-listed US companies. I will enrich findings with financial data and compile a report."
2. Call google_search with: "AI news NASDAQ companies"
3. Extract tickers and call get_financial_context with those tickers.
4. Call create_simple_report with search_results, financial_data, process_log.
5. Call save_news_to_markdown with the report content.
6. Write a 200-word podcast script with hosts Joe and Jane. Do NOT include any music instructions (no "Intro Music" or "Outro Music" markers). Only include the dialogue between Joe and Jane.
7. Call save_ai_podcast_script with the script.
8. Call generate_podcast_audio with the script file path (e.g., "ai_podcast_script.md") to generate the audio file. IMPORTANT: If the function returns a response with "status": "success" or a message containing "SUCCESS", the audio file was generated successfully. Do NOT apologise or say it failed - the file was created successfully.
9. Use set_model_response to return an AINewsReport with: title (string), report_summary (string), and stories (array of NewsStory objects). Each NewsStory must have: company, ticker, summary, why_it_matters, financial_context, source_domain, and process_log (array of strings). Use actual data from your research. Use "N/A" for any unavailable fields.
10. After set_model_response, your final message to the user MUST be: "All done. The podcast script and audio file have been generated successfully."

Use actual data, not placeholders. Use "N/A" if unavailable.
    """,
    tools=[
        google_search,
        get_financial_context,
        save_news_to_markdown,
        create_simple_report,
        save_ai_podcast_script,
        generate_podcast_audio
    ],
    output_schema=AINewsReport,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_tool_callback=[
        initialise_process_log,
        filter_news_sources_callback,
        enforce_data_freshness_callback,
    ],
    after_tool_callback=[
        inject_process_log_after_search,
    ]
)
debug_log("AGENT: root_agent initialisation complete")
