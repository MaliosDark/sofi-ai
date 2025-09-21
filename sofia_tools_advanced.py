#!/usr/bin/env python3
"""
SOFIA Advanced Tool Integration System
Expanded tool capabilities with APIs, databases, and web scraping
"""

import json, re, math, datetime, requests, sqlite3, os
import numpy as np
from typing import Dict, Any, List, Optional, Union
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
        patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+',
            r'calculate\s+.+',
            r'what\s+is\s+.+[\+\-\*\/].+',
            r'solve\s+.+',
        ]
        return any(re.search(pattern, query.lower()) for pattern in patterns)

    def execute(self, query: str) -> Dict[str, Any]:
        try:
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
        now = datetime.datetime.now()

        result = {
            "tool": "time",
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "month": now.strftime("%B"),
            "year": now.year,
            "timezone": "UTC"
        }

        query_lower = query.lower()
        if 'tomorrow' in query_lower:
            tomorrow = now + datetime.timedelta(days=1)
            result['tomorrow'] = tomorrow.strftime("%Y-%m-%d (%A)")
        elif 'yesterday' in query_lower:
            yesterday = now - datetime.timedelta(days=1)
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
            search_term = self._extract_search_term(query)

            if any(word in query.lower() for word in ['what is', 'who is', 'definition']):
                return self._wikipedia_search(search_term)

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
        query = re.sub(r'^(what|who|where|how|when|why)\s+is\s+', '', query.lower())
        query = re.sub(r'^(search|find|lookup)\s+(for\s+)?', '', query.lower())
        return query.strip()

    def _wikipedia_search(self, term: str) -> Dict[str, Any]:
        try:
            import wikipedia
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
        except Exception as e:
            return {"tool": "search", "error": f"Wikipedia search failed: {str(e)}"}

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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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

class ToolManager:
    def __init__(self):
        self.tools = [
            CalculatorTool(),
            TimeTool(),
            SearchTool(),
            DatabaseTool()
        ]
        print(f"🔧 Loaded {len(self.tools)} advanced tools: {[t.name for t in self.tools]}")

    def execute_tools(self, query: str) -> List[Dict[str, Any]]:
        results = []
        for tool in self.tools:
            if tool.can_handle(query):
                print(f"🛠️  Using {tool.name}: {tool.description}")
                result = tool.execute(query)
                if 'error' not in result:
                    results.append(result)
                else:
                    print(f"⚠️  Tool {tool.name} failed: {result['error']}")
        return results

    def get_available_tools(self) -> List[Dict[str, str]]:
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

        elif tool_type == 'database':
            if result.get('operation') == 'retrieve':
                return f"Retrieved {result.get('count', 0)} knowledge entries"
            elif result.get('operation') == 'store':
                return f"Stored: {result.get('content', 'N/A')}"
            else:
                return f"Database has {result.get('knowledge_entries', 0)} entries"

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
