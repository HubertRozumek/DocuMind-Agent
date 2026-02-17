import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool, Tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Tool input schemas
class TicketCheckInput(BaseModel):
    """
    Input schema for ticket checking tool
    """

    ticket_id: str = Field(description="Ticket ID to check (e.g., TICKET-001)")


class UserTicketsInput(BaseModel):
    """
    Input schema for user tickets search
    """

    user: str = Field(description="Username to search tickets for")


class DocumentSearchInput(BaseModel):
    """
    Input schema for document search
    """

    query: str = Field(description="Search query for documents")
    top_k: int = Field(default=5, description="Number of results to return")


class TicketCreationInput(BaseModel):
    """
    Input schema for creating tickets
    """

    title: str = Field(description="Ticket title")
    description: str = Field(description="Detailed description of the issue")
    priority: str = Field(default="medium", description="Priority: low, medium, high, critical")
    category: str = Field(default="General", description="Ticket category")


def create_ticket_tools() -> List[Tool]:
    """
    Create LangChain tools for ticket operations.

    Returns:
        List of ticket-related tools
    """
    from src.tools.ticket_checker import check_ticket_status, get_open_tickets_summary, get_ticket_api, search_my_tickets

    tools = [
        StructuredTool(
            name="check_ticket_status",
            description=(
                "Check the status of a support ticket. "
                "Use this when the user asks about a specific ticket ID. "
                "Input should be the ticket ID (e.g., TICKET-001)."
            ),
            func=check_ticket_status,
            args_schema=TicketCheckInput,
        ),
        StructuredTool(
            name="search_user_tickets",
            description=(
                "Search for all tickets assigned to a specific user. "
                "Use this when the user asks about their tickets or someone else's tickets. "
                "Input should be the username."
            ),
            func=search_my_tickets,
            args_schema=UserTicketsInput,
        ),
        Tool(
            name="get_open_tickets",
            description=(
                "Get a summary of all currently open tickets in the system. " "Use this when the user asks about open tickets or ticket queue."
            ),
            func=get_open_tickets_summary,
        ),
        Tool(
            name="get_ticket_statistics",
            description=(
                "Get overall statistics about tickets in the system "
                "(total count, by status, by priority, by category). "
                "Use this for general ticket system overview questions."
            ),
            func=lambda: str(get_ticket_api().get_statistics()),
        ),
    ]

    logger.info(f"Created {len(tools)} ticket tools")
    return tools


def create_document_tools(vector_store=None) -> List[Tool]:
    """
    Create LangChain tools for document operations.

    Args:
        vector_store: Vector store instance for document search

    Returns:
        List of document-related tools
    """

    def search_documents(query: str, top_k: int = 5) -> str:
        """
        Search for relevant documents
        """
        if vector_store is None:
            return "Document search is not available (vector store not initialized)."

        try:
            results = vector_store.search(query=query, n_results=top_k)

            if not results.get("documents"):
                return f"No documents found for query: {query}"

            output = f"Found {len(results['documents'])} relevant document(s):\n\n"

            for i, (doc, similarity) in enumerate(zip(results["documents"], results.get("similarities", []))):
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                output += f"{i+1}. (Similarity: {similarity:.2f})\n{preview}\n\n"

            return output.strip()

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return f"Error searching documents: {str(e)}"

    def get_document_count() -> str:
        """
        Get total number of documents in the store
        """
        if vector_store is None:
            return "Document store not available."

        try:
            stats = vector_store.get_collection_stats()
            return f"Total documents in store: {stats.get('total_documents', 0)}"
        except Exception as e:
            return f"Error getting document count: {str(e)}"

    tools = [
        StructuredTool(
            name="search_documents",
            description=(
                "Search for information in company documents. "
                "Use this when the user asks about policies, procedures, or regulations. "
                "Input should be a search query describing what to look for."
            ),
            func=search_documents,
            args_schema=DocumentSearchInput,
        ),
        Tool(
            name="get_document_count",
            description=("Get the total number of documents in the knowledge base. " "Use this when asked about how many documents are available."),
            func=get_document_count,
        ),
    ]

    logger.info(f"Created {len(tools)} document tools")
    return tools


class ToolRouter:
    """
    Routes queries to appropriate tools or document search.
    """

    def __init__(self, tools: List[Tool], vector_store=None):
        """
        Initialize tool router.

        Args:
            tools: List of available LangChain tools
            vector_store: Vector store for document search
        """
        self.tools = {tool.name: tool for tool in tools}
        self.vector_store = vector_store
        self.tool_usage_stats: Dict = {}

        # Keywords that suggest tool usage
        self.ticket_keywords = [
            "ticket",
            "support",
            "issue",
            "request",
            "incident",
            "helpdesk",
            "status",
        ]

        logger.info(f"ToolRouter initialized with {len(self.tools)} tools")

    def should_use_tool(self, query: str) -> Optional[tuple]:
        """
        Determine if a tool should be used for the query.

        Returns:
            Tuple of (tool_name, kwargs) or None for document search
        """
        query_lower = query.lower()

        # Check for ticket-related queries
        if any(keyword in query_lower for keyword in self.ticket_keywords):
            # Extract ticket ID if present
            import re

            ticket_match = re.search(r"ticket[- ]?(\d+)", query_lower)

            # "My tickets" query
            if "my tickets" in query_lower or "my ticket" in query_lower:
                return ("search_user_tickets", {"user": "current_user"})

            # Specific ticket ID with status check
            if ticket_match:
                ticket_id = f"TICKET-{ticket_match.group(1).zfill(3)}"
                if "status" in query_lower or "check" in query_lower:
                    return ("check_ticket_status", {"ticket_id": ticket_id})

            # Open tickets query
            if "open" in query_lower and "ticket" in query_lower:
                return ("get_open_tickets", {})

            # Statistics query
            if "statistics" in query_lower or "summary" in query_lower:
                return ("get_ticket_statistics", {})

        # Document count query
        if any(phrase in query_lower for phrase in ["how many documents", "number of documents", "document count"]):
            return ("get_document_count", {})

        # Default to document search
        return None

    def route_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Route query to appropriate tool or document search.

        Args:
            query: User query
            **kwargs: Additional arguments for tools

        Returns:
            Result dictionary with routing information
        """
        result = self.should_use_tool(query)

        try:
            if result:
                tool_name, tool_kwargs = result

                if tool_name in self.tools:
                    logger.info(f"Routing to tool: {tool_name}")
                    tool = self.tools[tool_name]

                    # Track usage
                    self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + 1

                    # Execute tool
                    try:
                        if hasattr(tool, "args_schema") and tool.args_schema:
                            final_kwargs = {**tool_kwargs, **kwargs}
                            if hasattr(tool, "invoke"):
                                tool_result = tool.invoke(final_kwargs)
                            else:
                                tool_result = tool.func(**final_kwargs)
                        else:
                            if hasattr(tool, "invoke"):
                                tool_result = tool.invoke({})
                            else:
                                tool_result = tool.func()

                        return {
                            "type": "tool",
                            "tool_name": tool_name,
                            "result": tool_result,
                            "success": True,
                        }

                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        return {
                            "type": "error",
                            "tool_name": tool_name,
                            "result": None,
                            "success": False,
                            "error": str(e),
                        }

            # Use document search
            logger.info("Routing to document search")

            if self.vector_store:
                results = self.vector_store.search(query=query, n_results=kwargs.get("top_k", 5))

                return {
                    "type": "document_search",
                    "tool_name": None,
                    "result": results,
                    "success": True,
                }
            else:
                return {
                    "type": "document_search",
                    "tool_name": None,
                    "result": None,
                    "success": False,
                    "error": "Vector store not available",
                }

        except Exception as e:
            logger.error(f"Routing error: {e}")
            return {
                "type": "error",
                "tool_name": None,
                "result": None,
                "success": False,
                "error": str(e),
            }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get tool usage statistics
        """
        return {
            "tool_usage": self.tool_usage_stats.copy(),
            "total_tool_calls": sum(self.tool_usage_stats.values()),
            "available_tools": list(self.tools.keys()),
        }


class ToolErrorHandler:
    """
    Handles errors from tool execution.
    """

    def __init__(self):
        self.error_log = []
        self.error_counts = {}

    def handle_tool_error(
        self,
        error: Exception,
        tool_name: str,
        query: str,
        fallback_to_search: bool = True,
    ) -> Dict[str, Any]:
        """
        Handle errors from tool execution.

        Args:
            error: Exception that occurred
            tool_name: Name of the tool that failed
            query: Original query
            fallback_to_search: Whether to fall back to document search

        Returns:
            Error handling result
        """
        error_entry = {
            "tool_name": tool_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "query": query,
            "timestamp": logging.Formatter().formatTime(logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None)),
        }

        self.error_log.append(error_entry)
        self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1

        logger.error(f"Tool '{tool_name}' failed with {type(error).__name__}: {str(error)}")

        response = {
            "success": False,
            "error": str(error),
            "tool_name": tool_name,
            "fallback_used": fallback_to_search,
        }

        if fallback_to_search:
            logger.info("Falling back to document search")
            response["fallback_message"] = f"The {tool_name} tool encountered an error. " "Searching documents instead..."

        return response

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics
        """
        return {
            "total_errors": len(self.error_log),
            "errors_by_tool": self.error_counts.copy(),
            "recent_errors": self.error_log[-10:] if self.error_log else [],
        }

    def reset_stats(self):
        """Reset error statistics"""
        self.error_log.clear()
        self.error_counts.clear()
        logger.info("Error statistics reset")


def create_all_tools(vector_store=None) -> List[Tool]:
    """
    Create all available tools (tickets + documents).

    Args:
        vector_store: Vector store for document operations

    Returns:
        List of all tools
    """
    tools = []

    # Add ticket tools
    tools.extend(create_ticket_tools())

    # Add document tools
    tools.extend(create_document_tools(vector_store))

    logger.info(f"Created total of {len(tools)} tools")
    return tools


# Convenience function to get tool by name
def get_tool_by_name(tools: List[Tool], name: str) -> Optional[Tool]:
    """
    Get a specific tool by name
    """
    for tool in tools:
        if tool.name == name:
            return tool
    return None
