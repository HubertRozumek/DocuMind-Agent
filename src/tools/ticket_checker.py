import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random
from enum import Enum

logger = logging.getLogger(__name__)


class TicketStatus(Enum):
    """
    Possible ticket statuses
    """
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(Enum):
    """
    Ticket priority levels
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MockTicketAPI:
    """
    Mock API for ticket management system.
    Simulates database with pre-populated tickets.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize mock API with sample data.

        Args:
            seed: Random seed for reproducible results
        """
        random.seed(seed)
        self.tickets_db = self._generate_sample_tickets()
        logger.info(f"MockTicketAPI initialized with {len(self.tickets_db)} tickets")

    def _generate_sample_tickets(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate sample tickets for testing
        """
        base_date = datetime.now()

        tickets = {
            "TICKET-001": {
                "id": "TICKET-001",
                "title": "Password reset request",
                "status": TicketStatus.RESOLVED.value,
                "priority": TicketPriority.MEDIUM.value,
                "assigned_to": "Jan Kowalski",
                "created_at": (base_date - timedelta(days=5)).isoformat(),
                "updated_at": (base_date - timedelta(days=2)).isoformat(),
                "resolved_at": (base_date - timedelta(days=2)).isoformat(),
                "description": "User unable to log in, needs password reset",
                "resolution": "Password reset link sent, issue resolved",
                "category": "Access Management",
            },
            "TICKET-002": {
                "id": "TICKET-002",
                "title": "VPN connection issues",
                "status": TicketStatus.IN_PROGRESS.value,
                "priority": TicketPriority.HIGH.value,
                "assigned_to": "Anna Nowak",
                "created_at": (base_date - timedelta(days=1)).isoformat(),
                "updated_at": base_date.isoformat(),
                "resolved_at": None,
                "description": "Employee cannot connect to company VPN",
                "resolution": None,
                "category": "Network",
            },
            "TICKET-003": {
                "id": "TICKET-003",
                "title": "Software installation request",
                "status": TicketStatus.OPEN.value,
                "priority": TicketPriority.LOW.value,
                "assigned_to": None,
                "created_at": base_date.isoformat(),
                "updated_at": base_date.isoformat(),
                "resolved_at": None,
                "description": "Need Adobe Acrobat Pro installed",
                "resolution": None,
                "category": "Software",
            },
            "TICKET-004": {
                "id": "TICKET-004",
                "title": "Security incident report",
                "status": TicketStatus.CLOSED.value,
                "priority": TicketPriority.CRITICAL.value,
                "assigned_to": "Piotr Wiśniewski",
                "created_at": (base_date - timedelta(days=10)).isoformat(),
                "updated_at": (base_date - timedelta(days=3)).isoformat(),
                "resolved_at": (base_date - timedelta(days=3)).isoformat(),
                "description": "Suspicious email attachment reported",
                "resolution": "Investigated and confirmed as phishing attempt. User trained.",
                "category": "Security",
            },
            "TICKET-005": {
                "id": "TICKET-005",
                "title": "Hardware malfunction",
                "status": TicketStatus.PENDING.value,
                "priority": TicketPriority.MEDIUM.value,
                "assigned_to": "Michał Zieliński",
                "created_at": (base_date - timedelta(days=3)).isoformat(),
                "updated_at": (base_date - timedelta(hours=6)).isoformat(),
                "resolved_at": None,
                "description": "Laptop keyboard keys not working",
                "resolution": None,
                "category": "Hardware",
            },
        }

        return tickets

    def get_ticket(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve ticket by ID.

        Args:
            ticket_id: Ticket identifier (e.g., "TICKET-001")

        Returns:
            Ticket data or None if not found
        """
        ticket = self.tickets_db.get(ticket_id.upper())

        if ticket:
            logger.info(f"Retrieved ticket: {ticket_id}")
            return ticket.copy()
        else:
            logger.warning(f"Ticket not found: {ticket_id}")
            return None

    def get_ticket_status(self, ticket_id: str) -> Dict[str, Any]:
        """
        Get ticket status information.

        Args:
            ticket_id: Ticket identifier

        Returns:
            Status information or error
        """
        ticket = self.get_ticket(ticket_id)

        if not ticket:
            return {
                "error": "Ticket not found",
                "ticket_id": ticket_id,
                "found": False,
            }

        return {
            "ticket_id": ticket["id"],
            "status": ticket["status"],
            "priority": ticket["priority"],
            "assigned_to": ticket["assigned_to"],
            "last_update": ticket["updated_at"],
            "found": True,
        }

    def search_tickets(
            self,
            status: Optional[str] = None,
            priority: Optional[str] = None,
            assigned_to: Optional[str] = None,
            category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search tickets by criteria.

        Args:
            status: Filter by status
            priority: Filter by priority
            assigned_to: Filter by assignee
            category: Filter by category

        Returns:
            List of matching tickets
        """
        results = []

        for ticket in self.tickets_db.values():
            match = True

            if status and ticket["status"] != status:
                match = False
            if priority and ticket["priority"] != priority:
                match = False
            if assigned_to and ticket["assigned_to"] != assigned_to:
                match = False
            if category and ticket["category"] != category:
                match = False

            if match:
                results.append(ticket.copy())

        logger.info(f"Search found {len(results)} tickets")
        return results

    def get_my_tickets(self, user: str) -> List[Dict[str, Any]]:
        """
        Get all tickets assigned to a specific user.

        Args:
            user: Username

        Returns:
            List of user's tickets
        """
        return self.search_tickets(assigned_to=user)

    def get_open_tickets(self) -> List[Dict[str, Any]]:
        """
        Get all open tickets
        """
        return self.search_tickets(status=TicketStatus.OPEN.value)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ticket system statistics.

        Returns:
            Statistics dictionary
        """
        total = len(self.tickets_db)
        by_status = {}
        by_priority = {}
        by_category = {}

        for ticket in self.tickets_db.values():
            # Count by status
            status = ticket["status"]
            by_status[status] = by_status.get(status, 0) + 1

            # Count by priority
            priority = ticket["priority"]
            by_priority[priority] = by_priority.get(priority, 0) + 1

            # Count by category
            category = ticket["category"]
            by_category[category] = by_category.get(category, 0) + 1

        return {
            "total_tickets": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "by_category": by_category,
            "timestamp": datetime.now().isoformat(),
        }

    def create_ticket(
            self,
            title: str,
            description: str,
            priority: str = "medium",
            category: str = "General",
    ) -> Dict[str, Any]:
        """
        Create a new ticket (simulated).

        Args:
            title: Ticket title
            description: Ticket description
            priority: Ticket priority
            category: Ticket category

        Returns:
            Created ticket data
        """
        # Validation
        if not title or not title.strip():
            raise ValueError("Title cannot be empty")

        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        valid_priorities = ["low", "medium", "high", "critical"]
        if priority.lower() not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")

        # Generate new ticket ID
        ticket_count = len(self.tickets_db) + 1
        ticket_id = f"TICKET-{ticket_count:03d}"

        new_ticket = {
            "id": ticket_id,
            "title": title,
            "status": TicketStatus.OPEN.value,
            "priority": priority,
            "assigned_to": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "resolved_at": None,
            "description": description,
            "resolution": None,
            "category": category,
        }

        self.tickets_db[ticket_id] = new_ticket
        logger.info(f"Created new ticket: {ticket_id}")

        return new_ticket.copy()


# Global instance for easy access
_global_api = None


def get_ticket_api() -> MockTicketAPI:
    """
    Get global ticket API instance
    """
    global _global_api
    if _global_api is None:
        _global_api = MockTicketAPI()
    return _global_api


# Convenience functions for LangChain Tool integration
def check_ticket_status(ticket_id: str) -> str:
    """
    Check status of a ticket.

    Args:
        ticket_id: Ticket ID to check (e.g., "TICKET-001")

    Returns:
        Human-readable status information
    """
    api = get_ticket_api()
    result = api.get_ticket_status(ticket_id)

    if not result.get("found"):
        return f"Ticket {ticket_id} not found in the system."

    status_text = f"""
Ticket: {result['ticket_id']}
Status: {result['status'].upper()}
Priority: {result['priority'].upper()}
Assigned to: {result['assigned_to'] or 'Unassigned'}
Last updated: {result['last_update']}
"""
    return status_text.strip()


def search_my_tickets(user: str) -> str:
    """
    Get all tickets assigned to a user.

    Args:
        user: Username to search for

    Returns:
        List of user's tickets
    """
    api = get_ticket_api()
    tickets = api.get_my_tickets(user)

    if not tickets:
        return f"No tickets found for user: {user}"

    result = f"Found {len(tickets)} ticket(s) for {user}:\n\n"

    for ticket in tickets:
        result += f"- {ticket['id']}: {ticket['title']} ({ticket['status']})\n"

    return result.strip()


def get_open_tickets_summary() -> str:
    """
    Get summary of all open tickets.

    Returns:
        Summary of open tickets
    """
    api = get_ticket_api()
    tickets = api.get_open_tickets()

    if not tickets:
        return "No open tickets in the system."

    result = f"Found {len(tickets)} open ticket(s):\n\n"

    for ticket in tickets:
        result += f"- {ticket['id']}: {ticket['title']} (Priority: {ticket['priority']})\n"

    return result.strip()