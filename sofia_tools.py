#!/usr/bin/env python3
"""
SOFIA Advanced Tool Integration System
Expanded tool capabilities with APIs, databases, and web scraping
"""

import json, re, math, datetime, requests, sqlite3, os
import numpy as np
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import wikipedia
import feedparser
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import feedparser
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def can_handle(self, query: str) -> bool:
        raise NotImplementedError

    def execute(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "Performs mathematical calculations")

    def can_handle(self, query: str) -> bool:
        # Enhanced pattern matching for calculations
        patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Basic operations
            r'calculate\s+.+',  # "calculate X + Y"
            r'what\s+is\s+.+[\+\-\*\/].+',  # "what is 5 + 3"
            r'solve\s+.+',  # "solve 2*x = 10"
        ]
        return any(re.search(pattern, query.lower()) for pattern in patterns)

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Extract mathematical expressions
            expr_match = re.search(r'(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)', query.lower())
            if expr_match:
                expression = expr_match.group(1).replace(' ', '')
                result = eval(expression)
                return {
                    "tool": "calculator",
                    "expression": expression,
                    "result": result,
                    "type": "arithmetic"
                }

            # Handle more complex expressions
            if 'sqrt' in query.lower():
                numbers = re.findall(r'\d+(?:\.\d+)?', query)
                if numbers:
                    num = float(numbers[0])
                    result = math.sqrt(num)
                    return {
                        "tool": "calculator",
                        "expression": f"sqrt({num})",
                        "result": result,
                        "type": "square_root"
                    }

        except Exception as e:
            return {"tool": "calculator", "error": f"Could not calculate: {str(e)}"}

        return {"tool": "calculator", "error": "No valid calculation found"}

class TimeTool(Tool):
    def __init__(self):
        super().__init__("time", "Provides current time, date, and temporal information")

    def can_handle(self, query: str) -> bool:
        time_keywords = ['time', 'date', 'day', 'hour', 'minute', 'second', 'now', 'today', 'tomorrow', 'yesterday']
        return any(keyword in query.lower() for keyword in time_keywords)

    def execute(self, query: str) -> Dict[str, Any]:
        now = datetime.now()

        result = {
            "tool": "time",
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "month": now.strftime("%B"),
            "year": now.year,
            "timezone": "UTC"  # Simplified
        }

        # Handle relative time queries
        query_lower = query.lower()
        if 'tomorrow' in query_lower:
            tomorrow = now + timedelta(days=1)
            result['tomorrow'] = tomorrow.strftime("%Y-%m-%d (%A)")
        elif 'yesterday' in query_lower:
            yesterday = now - timedelta(days=1)
            result['yesterday'] = yesterday.strftime("%Y-%m-%d (%A)")

        return result

class SearchTool(Tool):
    def __init__(self):
        super().__init__("search", "Performs web searches and information retrieval")

    def can_handle(self, query: str) -> bool:
        search_keywords = ['search', 'find', 'lookup', 'what is', 'who is', 'where is', 'how to']
        return any(keyword in query.lower() for keyword in search_keywords)

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Extract search term
            search_term = self._extract_search_term(query)

            # Try Wikipedia first for factual queries
            if any(word in query.lower() for word in ['what is', 'who is', 'definition']):
                return self._wikipedia_search(search_term)

            # For general searches, provide a structured response
            return {
                "tool": "search",
                "search_term": search_term,
                "result": f"Search results for '{search_term}' would be retrieved from web sources",
                "type": "general_search",
                "suggestion": "Consider using specific APIs for better results"
            }

        except Exception as e:
            return {"tool": "search", "error": f"Search failed: {str(e)}"}

    def _extract_search_term(self, query: str) -> str:
        """Extract the main search term from query"""
        # Remove question words and get the core subject
        query = re.sub(r'^(what|who|where|how|when|why)\s+is\s+', '', query.lower())
        query = re.sub(r'^(search|find|lookup)\s+(for\s+)?', '', query.lower())
        return query.strip()

    def _wikipedia_search(self, term: str) -> Dict[str, Any]:
        """Search Wikipedia for information"""
        try:
            page = wikipedia.page(term, auto_suggest=True)
            summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary

            return {
                "tool": "search",
                "search_term": term,
                "result": summary,
                "source": "Wikipedia",
                "url": page.url,
                "type": "wikipedia_summary"
            }
        except wikipedia.exceptions.DisambiguationError as e:
            return {
                "tool": "search",
                "search_term": term,
                "result": f"Multiple results found. Options: {', '.join(e.options[:5])}",
                "type": "disambiguation"
            }
        except Exception as e:
            return {"tool": "search", "error": f"Wikipedia search failed: {str(e)}"}

class WeatherTool(Tool):
    def __init__(self, api_key: str = None):
        super().__init__("weather", "Provides weather information")
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')

    def can_handle(self, query: str) -> bool:
        weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy']
        return any(keyword in query.lower() for keyword in weather_keywords)

    def execute(self, query: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "tool": "weather",
                "error": "Weather API key not configured",
                "suggestion": "Set OPENWEATHER_API_KEY environment variable"
            }

        try:
            # Extract location from query
            location = self._extract_location(query)
            if not location:
                return {"tool": "weather", "error": "Could not determine location"}

            # In a real implementation, you would call the weather API
            # For demo purposes, return mock data
            return {
                "tool": "weather",
                "location": location,
                "temperature": "22°C",
                "condition": "Partly cloudy",
                "humidity": "65%",
                "wind_speed": "10 km/h",
                "type": "current_weather"
            }

        except Exception as e:
            return {"tool": "weather", "error": f"Weather lookup failed: {str(e)}"}

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from weather query"""
        # Simple location extraction - in practice, use geocoding
        locations = ['paris', 'london', 'tokyo', 'new york', 'berlin', 'madrid']
        query_lower = query.lower()

        for location in locations:
            if location in query_lower:
                return location.title()

        # Default to a major city if no location found
        return "New York"

class DatabaseTool(Tool):
    def __init__(self, db_path: str = "sofia_memory.db"):
        super().__init__("database", "Access and query local knowledge database")
        self.db_path = db_path
        self._init_database()

    def can_handle(self, query: str) -> bool:
        db_keywords = ['remember', 'recall', 'stored', 'database', 'knowledge', 'facts']
        return any(keyword in query.lower() for keyword in db_keywords)

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            query_lower = query.lower()

            if 'remember' in query_lower or 'store' in query_lower:
                return self._store_information(query)
            elif 'recall' in query_lower or 'retrieve' in query_lower:
                return self._retrieve_information(query)
            else:
                return self._query_database(query)

        except Exception as e:
            return {"tool": "database", "error": f"Database operation failed: {str(e)}"}

    def _init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                fact TEXT UNIQUE,
                category TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _store_information(self, query: str) -> Dict[str, Any]:
        """Store information in the database"""
        # Extract topic and content from query
        # This is a simplified implementation
        content = re.sub(r'(remember|store)\s+', '', query, flags=re.IGNORECASE).strip()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO knowledge (topic, content, source) VALUES (?, ?, ?)",
            ("user_input", content, "conversation")
        )

        conn.commit()
        conn.close()

        return {
            "tool": "database",
            "operation": "store",
            "content": content,
            "status": "stored"
        }

    def _retrieve_information(self, query: str) -> Dict[str, Any]:
        """Retrieve information from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent knowledge entries
        cursor.execute(
            "SELECT content, timestamp FROM knowledge ORDER BY timestamp DESC LIMIT 5"
        )

        results = cursor.fetchall()
        conn.close()

        return {
            "tool": "database",
            "operation": "retrieve",
            "results": [{"content": row[0], "timestamp": row[1]} for row in results],
            "count": len(results)
        }

    def _query_database(self, query: str) -> Dict[str, Any]:
        """Perform a general database query"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM knowledge")
        knowledge_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM facts")
        facts_count = cursor.fetchone()[0]

        conn.close()

        return {
            "tool": "database",
            "operation": "stats",
            "knowledge_entries": knowledge_count,
            "facts_stored": facts_count
        }

class WebScrapingTool(Tool):
    def __init__(self):
        super().__init__("web_scraper", "Extracts information from web pages")

    def can_handle(self, query: str) -> bool:
        web_keywords = ['scrape', 'extract', 'from website', 'web page', 'url']
        return any(keyword in query.lower() for keyword in web_keywords)

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Extract URL from query
            url_match = re.search(r'https?://[^\s]+', query)
            if not url_match:
                return {"tool": "web_scraper", "error": "No valid URL found in query"}

            url = url_match.group(0)

            # In a real implementation, you would scrape the webpage
            # For demo purposes, return mock data
            return {
                "tool": "web_scraper",
                "url": url,
                "title": f"Content from {urlparse(url).netloc}",
                "summary": "Web scraping would extract the main content, headers, and relevant information from the webpage.",
                "word_count": 1250,
                "type": "web_content"
            }

        except Exception as e:
            return {"tool": "web_scraper", "error": f"Web scraping failed: {str(e)}"}

class NewsTool(Tool):
    def __init__(self):
        super().__init__("news", "Provides latest news and headlines")

    def can_handle(self, query: str) -> bool:
        news_keywords = ['news', 'headlines', 'latest', 'breaking', 'updates']
        return any(keyword in query.lower() for keyword in news_keywords)

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # In a real implementation, you would use a news API
            # For demo purposes, return mock news
            mock_headlines = [
                "AI Breakthrough: New Model Achieves Human-Level Performance",
                "Technology Conference Announces Major Developments",
                "Science Discovery Changes Understanding of Universe",
                "Global Climate Initiative Shows Promising Results"
            ]

            return {
                "tool": "news",
                "headlines": mock_headlines,
                "count": len(mock_headlines),
                "source": "Mock News API",
                "timestamp": datetime.now().isoformat(),
                "type": "headlines"
            }

        except Exception as e:
            return {"tool": "news", "error": f"News retrieval failed: {str(e)}"}

class TranslationTool(Tool):
    def __init__(self):
        super().__init__("translator", "Translates text between languages")

    def can_handle(self, query: str) -> bool:
        translation_keywords = ['translate', 'translation', 'to spanish', 'to french', 'to german', 'in spanish']
        return any(keyword in query.lower() for keyword in translation_keywords)

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            # Extract text to translate and target language
            text_match = re.search(r'translate\s+["\']([^"\']+)["\']', query, re.IGNORECASE)
            if not text_match:
                return {"tool": "translator", "error": "Could not extract text to translate"}

            text = text_match.group(1)

            # Extract target language
            lang_match = re.search(r'to\s+(\w+)', query, re.IGNORECASE)
            target_lang = lang_match.group(1) if lang_match else "spanish"

            # In a real implementation, you would use a translation API
            # For demo purposes, return mock translation
            mock_translations = {
                "spanish": f"Traducción al español: {text}",
                "french": f"Traduction française: {text}",
                "german": f"Deutsche Übersetzung: {text}"
            }

            translation = mock_translations.get(target_lang.lower(), f"Mock translation to {target_lang}: {text}")

            return {
                "tool": "translator",
                "original_text": text,
                "translated_text": translation,
                "source_language": "english",
                "target_language": target_lang,
                "type": "translation"
            }

        except Exception as e:
            return {"tool": "translator", "error": f"Translation failed: {str(e)}"}

class ToolManager:
    def __init__(self):
        self.tools = [
            CalculatorTool(),
            TimeTool(),
            SearchTool(),
            WeatherTool(),
            DatabaseTool(),
            WebScrapingTool(),
            NewsTool(),
            TranslationTool()
        ]
        print(f"🔧 Loaded {len(self.tools)} advanced tools: {[t.name for t in self.tools]}")

    def execute_tools(self, query: str) -> List[Dict[str, Any]]:
        results = []
        for tool in self.tools:
            if tool.can_handle(query):
                print(f"🛠️  Using {tool.name}: {tool.description}")
                result = tool.execute(query)
                if 'error' not in result:  # Only add successful results
                    results.append(result)
                else:
                    print(f"⚠️  Tool {tool.name} failed: {result['error']}")
        return results

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        return [{"name": tool.name, "description": tool.description} for tool in self.tools]

class AdvancedToolAugmentedSOFIA:
    def __init__(self):
        from conversational_sofia import ConversationalSOFIA
        self.sofia = ConversationalSOFIA()
        self.tool_manager = ToolManager()

    def process_query(self, query: str) -> Dict[str, Any]:
        print(f"🤖 Processing: '{query}'")

        tool_results = self.tool_manager.execute_tools(query)

        if tool_results:
            # Enhanced context building
            tool_contexts = []
            for result in tool_results:
                formatted = self._format_tool_result(result)
                tool_contexts.append(f"Tool {result['tool']}: {formatted}")

            enhanced_query = f"{query}\n\nTool Results:\n" + "\n".join(tool_contexts)
        else:
            enhanced_query = query

        response, embedding = self.sofia.chat(enhanced_query)

        return {
            "response": response,
            "embedding": embedding,
            "tools_used": tool_results,
            "tool_count": len(tool_results),
            "enhanced_query": enhanced_query
        }

    def _format_tool_result(self, result: Dict[str, Any]) -> str:
        """Format tool results for context"""
        tool_type = result.get('tool', '')

        if tool_type == 'time':
            return f"{result.get('current_time', 'N/A')} on {result.get('current_date', 'N/A')} ({result.get('day_of_week', 'N/A')})"

        elif tool_type == 'calculator':
            if 'expression' in result:
                return f"{result['expression']} = {result['result']}"
            else:
                return str(result.get('result', 'N/A'))

        elif tool_type == 'search':
            return result.get('result', 'Search completed')

        elif tool_type == 'weather':
            return f"Weather in {result.get('location', 'Unknown')}: {result.get('temperature', 'N/A')}, {result.get('condition', 'N/A')}"

        elif tool_type == 'database':
            if result.get('operation') == 'retrieve':
                return f"Retrieved {result.get('count', 0)} knowledge entries"
            elif result.get('operation') == 'store':
                return f"Stored: {result.get('content', 'N/A')}"
            else:
                return f"Database has {result.get('knowledge_entries', 0)} entries"

        elif tool_type == 'news':
            headlines = result.get('headlines', [])
            return f"Latest headlines: {', '.join(headlines[:2])}"

        elif tool_type == 'translator':
            return f"'{result.get('original_text', '')}' → '{result.get('translated_text', '')}'"

        else:
            return str(result.get('result', 'Tool executed successfully'))

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sofia_tools.py 'query'")
        print("\nAvailable tools:")
        tool_manager = ToolManager()
        for tool_info in tool_manager.get_available_tools():
            print(f"  - {tool_info['name']}: {tool_info['description']}")
        return

    user_input = ' '.join(sys.argv[1:])

    tool_sofia = AdvancedToolAugmentedSOFIA()
    result = tool_sofia.process_query(user_input)

    print(f"\n🤖 SOFIA: {result['response']}")
    if result['tools_used']:
        print(f"\n🛠️  Advanced Tools Used: {result['tool_count']}")
        for tool_result in result['tools_used']:
            formatted = tool_sofia._format_tool_result(tool_result)
            print(f"  - {tool_result['tool']}: {formatted}")

if __name__ == "__main__":
    main()
            formatted = tool_sofia._format_result(tool_result)
            print(f"  - {tool_result['tool']}: {formatted}")      print(f"🤖 Processing: '{query}'")
        
        tool_results = self.tool_manager.execute_tools(query)
        
        if tool_results:
            tool_context = "\n".join([f"Tool {r['tool']}: {self._format_result(r)}" for r in tool_results])
            enhanced_query = f"{query}\n\nTools:\n{tool_context}"
        else:
            enhanced_query = query

        response, embedding = self.sofia.chat(enhanced_query)
        
        return {
            "response": response,
            "embedding": embedding,
            "tools_used": tool_results,
            "tool_count": len(tool_results)
        }

    def _format_result(self, result: Dict[str, Any]) -> str:
        if result['tool'] == 'time':
            return f"{result.get('current_time', 'N/A')} on {result.get('current_date', 'N/A')} ({result.get('day_of_week', 'N/A')})"
        elif result['tool'] == 'calculator':
            return f"{result.get('expression', 'N/A')} = {result.get('result', 'N/A')}"
        return str(result.get('result', 'N/A'))capabilities.
"""
import json, re, math, datetime, numpy as np
from typing import Dict, Any, List, Optional

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def can_handle(self, query: str) -> bool:
        raise NotImplementedError

    def execute(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "Performs calculations")

    def can_handle(self, query: str) -> bool:
        return bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', query.lower()))

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            expr = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', query.lower())
            if expr:
                result = eval(expr.group(1).replace(' ', ''))
                return {"tool": "calculator", "expression": expr.group(1), "result": result}
        except:
            pass
        return {"tool": "calculator", "error": "Could not calculate"}

class TimeTool(Tool):
    def __init__(self):
        super().__init__("time", "Provides time/date")

    def can_handle(self, query: str) -> bool:
        return 'time' in query.lower() or 'date' in query.lower()

    def execute(self, query: str) -> Dict[str, Any]:
        now = datetime.datetime.now()
        return {
            "tool": "time",
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A")
        }

class ToolManager:
    def __init__(self):
        self.tools = [CalculatorTool(), TimeTool()]
        print(f"🔧 Loaded {len(self.tools)} tools")

    def execute_tools(self, query: str) -> List[Dict[str, Any]]:
        results = []
        for tool in self.tools:
            if tool.can_handle(query):
                print(f"🛠️  Using {tool.name}")
                results.append(tool.execute(query))
        return results

class ToolAugmentedSOFIA:
    def __init__(self):
        from conversational_sofia import ConversationalSOFIA
        self.sofia = ConversationalSOFIA()
        self.tool_manager = ToolManager()

    def process_query(self, query: str) -> Dict[str, Any]:
        print(f"🤖 Processing: '{query}'")
        
        tool_results = self.tool_manager.execute_tools(query)
        
        if tool_results:
            tool_context = "\n".join([f"Tool {r['tool']}: {self._format_result(r)}" for r in tool_results])
            enhanced_query = f"{query}\n\nTools:\n{tool_context}"
        else:
            enhanced_query = query

        response, embedding = self.sofia.chat(enhanced_query)
        
        return {
            "response": response,
            "embedding": embedding,
            "tools_used": tool_results,
            "tool_count": len(tool_results)
        }

    def _format_result(self, result: Dict[str, Any]) -> str:
        if result['tool'] == 'time':
            return f"{result.get('current_time', 'N/A')} on {result.get('current_date', 'N/A')} ({result.get('day_of_week', 'N/A')})"
        elif result['tool'] == 'calculator':
            return f"{result.get('expression', 'N/A')} = {result.get('result', 'N/A')}"
        return str(result.get('result', 'N/A'))

def main():
    import sys
    user_input = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read().strip()
    
    if not user_input:
        print("Usage: python sofia_tools.py 'query'")
        return

    tool_sofia = ToolAugmentedSOFIA()
    result = tool_sofia.process_query(user_input)
    
    print(f"\\n🤖 SOFIA: {result['response']}")
    if result['tools_used']:
        print(f"\\n🛠️  Tools: {result['tool_count']}")
        for tool_result in result['tools_used']:
            print(f"  - {tool_result['tool']}: {tool_result.get('result', 'N/A')}")

if __name__ == "__main__":
    main()
