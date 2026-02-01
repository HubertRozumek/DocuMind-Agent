"""
Optimized prompts and parameters for the DocuMind Agent.
Task 10.3: Prompt optimization and parameter tuning.
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class PromptProfile(Enum):
    """Different prompt profiles for various use cases"""

    FAST = "fast"  # Quick responses, lower quality
    BALANCED = "balanced"  # Default, good balance
    ACCURATE = "accurate"  # High quality, slower
    CREATIVE = "creative"  # More creative responses


@dataclass
class OptimizedPromptConfig:
    """Configuration for optimized prompts"""

    # Generator prompts
    generator_system_prompt: str
    generator_user_template: str
    generator_temperature: float
    generator_max_tokens: int

    # Grader prompts
    grader_system_prompt: str
    grader_confidence_threshold: float
    grader_temperature: float

    # Rewriter prompts
    rewriter_system_prompt: str
    rewriter_user_template: str
    rewriter_temperature: float

    # Search parameters
    search_top_k: int
    search_threshold: float

    # Agent behavior
    max_iterations: int
    confidence_boost_factor: float


# OPTIMIZED PROMPTS FOR GENERATOR
GENERATOR_SYSTEM_PROMPTS = {
    PromptProfile.FAST: """You are a helpful assistant. Answer questions concisely based on the provided documents.""",
    PromptProfile.BALANCED: """You are an expert assistant helping employees find information in company documents.

Your role:
- Answer questions accurately based on provided documents
- Be professional and clear
- Cite specific parts of documents when possible
- If information is missing, say so clearly

Remember: Only use information from the provided documents.""",
    PromptProfile.ACCURATE: """You are a senior information specialist for a large organization.

Your expertise:
1. Precise document analysis and interpretation
2. Clear communication of complex policies and procedures
3. Careful citation of source materials
4. Recognition of information gaps

Guidelines:
- Read all provided documents carefully
- Extract relevant information accurately
- Cite specific sections that support your answer
- Distinguish between certain and uncertain information
- If documents don't contain the answer, clearly state this
- Use professional business language
- Provide actionable information when possible

Quality standards:
- Accuracy > Speed
- Completeness matters
- Professional tone required""",
    PromptProfile.CREATIVE: """You are a creative, engaging assistant helping users understand company information.

Your approach:
- Make complex information accessible
- Use examples and analogies when helpful
- Be conversational yet professional
- Focus on what matters to the user

Balance creativity with accuracy - always base answers on the documents.""",
}

GENERATOR_USER_TEMPLATES = {
    PromptProfile.FAST: """Question: {question}

Documents:
{documents}

Answer:""",
    PromptProfile.BALANCED: """DOCUMENTS:
{documents}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer based ONLY on the provided documents
2. If documents don't contain the answer, say "I couldn't find this information in the available documents"
3. Cite relevant parts of documents
4. Be clear and professional

ANSWER:""",
    PromptProfile.ACCURATE: """=== PROVIDED DOCUMENTS ===
{documents}

=== USER QUESTION ===
{question}

=== TASK ===
Analyze the documents carefully and provide a comprehensive answer to the user's question.

Requirements:
✓ Use ONLY information from the provided documents
✓ Cite specific passages that support your answer
✓ If multiple documents are relevant, integrate information from all of them
✓ If documents don't fully answer the question, explain what's covered and what's missing
✓ Use clear, professional language appropriate for a business setting
✓ Structure your answer logically (e.g., overview → details → summary)

=== YOUR ANSWER ===
""",
    PromptProfile.CREATIVE: """Here's what I found in the documents:
{documents}

Your question:
{question}

Let me help you understand this! I'll explain based on what's in our documents.

My answer:""",
}

# OPTIMIZED PROMPTS FOR GRADER
GRADER_SYSTEM_PROMPTS = {
    PromptProfile.FAST: "Determine if the document is relevant to the question. Answer YES or NO.",
    PromptProfile.BALANCED: """You are an expert at evaluating document relevance.

Task: Determine if a document is relevant to a user's question.

Criteria:
- RELEVANT: Document directly or partially answers the question
- NOT RELEVANT: Document doesn't contain information related to the question

Be strict but fair in your evaluation.""",
    PromptProfile.ACCURATE: """You are a senior information analyst specializing in document relevance assessment.

Evaluation criteria:

HIGHLY RELEVANT (confidence 0.8-1.0):
- Document directly answers the question
- Contains specific information requested
- Answers are clear and unambiguous

MODERATELY RELEVANT (confidence 0.5-0.7):
- Document partially addresses the question
- Contains related but not exact information
- May require interpretation

NOT RELEVANT (confidence 0.0-0.4):
- No connection to the question
- Wrong topic or context
- Generic information not applicable

Output requirements:
- Provide a confidence score (0-1)
- Give a brief reason for your assessment
- Be conservative - if unsure, mark as not relevant""",
}

# OPTIMIZED PROMPTS FOR QUERY REWRITER
REWRITER_SYSTEM_PROMPTS = {
    PromptProfile.FAST: "Rewrite the question to make it clearer for document search.",
    PromptProfile.BALANCED: """You are an expert at improving search queries for corporate document systems.

Task: Rewrite user questions to improve document retrieval.

Guidelines:
1. Make vague questions more specific
2. Add relevant corporate terminology
3. Preserve the original intent
4. Create 2-3 alternative versions
5. Focus on searchable keywords

Format: One rewritten question per line.""",
    PromptProfile.ACCURATE: """You are a corporate information specialist with expertise in query optimization.

Your task: Transform user questions into optimized search queries for a corporate document repository.

Analysis steps:
1. Identify the core information need
2. Recognize implicit assumptions or context
3. Add domain-specific terminology
4. Consider multiple search angles
5. Ensure questions are answerable by documents

Optimization techniques:
- Make ambiguous terms specific (e.g., "time off" → "annual leave policy")
- Add corporate context (e.g., "password rules" → "IT security password requirements")
- Break complex questions into focused queries
- Use formal business language
- Include relevant department/process names

Output: Provide 2-3 optimized versions of the question, each on a new line.
Quality over quantity - each version should explore a different search angle.""",
}

REWRITER_USER_TEMPLATES = {
    PromptProfile.FAST: """Original question: {question}

Rewritten versions:""",
    PromptProfile.BALANCED: """Original question: {question}

Previous attempts: {search_history}

Provide 2-3 improved versions that:
- Are more specific
- Use professional terminology
- Could find better document matches

Rewritten questions:""",
    PromptProfile.ACCURATE: """=== ORIGINAL QUESTION ===
{question}

=== SEARCH HISTORY ===
{search_history}

=== YOUR TASK ===
Analyze the question and provide 2-3 optimized versions for document search.

Consider:
- What information is the user really seeking?
- What corporate terminology applies?
- What documents would contain this information?
- How can we make this more specific?

=== OPTIMIZED VERSIONS ===
(one per line, no numbering)""",
}


# CONFIGURATION PRESETS
def get_prompt_config(profile: PromptProfile) -> OptimizedPromptConfig:
    """
    Get optimized configuration for a specific profile.

    Args:
        profile: Prompt profile to use

    Returns:
        Complete prompt configuration
    """
    configs = {
        PromptProfile.FAST: OptimizedPromptConfig(
            generator_system_prompt=GENERATOR_SYSTEM_PROMPTS[PromptProfile.FAST],
            generator_user_template=GENERATOR_USER_TEMPLATES[PromptProfile.FAST],
            generator_temperature=0.3,
            generator_max_tokens=500,
            grader_system_prompt=GRADER_SYSTEM_PROMPTS[PromptProfile.FAST],
            grader_confidence_threshold=0.5,
            grader_temperature=0.0,
            rewriter_system_prompt=REWRITER_SYSTEM_PROMPTS[PromptProfile.FAST],
            rewriter_user_template=REWRITER_USER_TEMPLATES[PromptProfile.FAST],
            rewriter_temperature=0.5,
            search_top_k=3,
            search_threshold=0.6,
            max_iterations=2,
            confidence_boost_factor=1.0,
        ),
        PromptProfile.BALANCED: OptimizedPromptConfig(
            generator_system_prompt=GENERATOR_SYSTEM_PROMPTS[PromptProfile.BALANCED],
            generator_user_template=GENERATOR_USER_TEMPLATES[PromptProfile.BALANCED],
            generator_temperature=0.1,
            generator_max_tokens=1000,
            grader_system_prompt=GRADER_SYSTEM_PROMPTS[PromptProfile.BALANCED],
            grader_confidence_threshold=0.6,
            grader_temperature=0.0,
            rewriter_system_prompt=REWRITER_SYSTEM_PROMPTS[PromptProfile.BALANCED],
            rewriter_user_template=REWRITER_USER_TEMPLATES[PromptProfile.BALANCED],
            rewriter_temperature=0.3,
            search_top_k=5,
            search_threshold=0.7,
            max_iterations=3,
            confidence_boost_factor=1.1,
        ),
        PromptProfile.ACCURATE: OptimizedPromptConfig(
            generator_system_prompt=GENERATOR_SYSTEM_PROMPTS[PromptProfile.ACCURATE],
            generator_user_template=GENERATOR_USER_TEMPLATES[PromptProfile.ACCURATE],
            generator_temperature=0.05,
            generator_max_tokens=1500,
            grader_system_prompt=GRADER_SYSTEM_PROMPTS[PromptProfile.ACCURATE],
            grader_confidence_threshold=0.7,
            grader_temperature=0.0,
            rewriter_system_prompt=REWRITER_SYSTEM_PROMPTS[PromptProfile.ACCURATE],
            rewriter_user_template=REWRITER_USER_TEMPLATES[PromptProfile.ACCURATE],
            rewriter_temperature=0.2,
            search_top_k=7,
            search_threshold=0.75,
            max_iterations=4,
            confidence_boost_factor=1.2,
        ),
        PromptProfile.CREATIVE: OptimizedPromptConfig(
            generator_system_prompt=GENERATOR_SYSTEM_PROMPTS[PromptProfile.CREATIVE],
            generator_user_template=GENERATOR_USER_TEMPLATES[PromptProfile.CREATIVE],
            generator_temperature=0.7,
            generator_max_tokens=1200,
            grader_system_prompt=GRADER_SYSTEM_PROMPTS[PromptProfile.BALANCED],
            grader_confidence_threshold=0.6,
            grader_temperature=0.1,
            rewriter_system_prompt=REWRITER_SYSTEM_PROMPTS[PromptProfile.BALANCED],
            rewriter_user_template=REWRITER_USER_TEMPLATES[PromptProfile.BALANCED],
            rewriter_temperature=0.6,
            search_top_k=5,
            search_threshold=0.65,
            max_iterations=3,
            confidence_boost_factor=1.0,
        ),
    }

    return configs[profile]


# PERFORMANCE TUNING RECOMMENDATIONS
PERFORMANCE_RECOMMENDATIONS = {
    "embeddings": {
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 32,
        "normalize": True,
        "device": "auto",  # Will use GPU if available
    },
    "vector_store": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "similarity_metric": "cosine",
        "index_type": "hnsw",  # For ChromaDB
    },
    "llm": {
        "model_recommendations": {
            "fast": "phi3:mini",
            "balanced": "llama3.2:3b",
            "accurate": "llama3.1:8b",
        },
        "context_window": 4096,
        "max_retries": 3,
        "timeout": 30,
    },
    "agent": {
        "retrieval_batch_size": 5,
        "max_parallel_grading": 3,
        "cache_results": True,
        "log_level": "INFO",
    },
}


def get_optimal_config_for_hardware() -> Dict[str, Any]:
    """
    Get optimal configuration based on available hardware.

    Returns:
        Optimized configuration dictionary
    """
    import torch

    config = {
        "profile": PromptProfile.BALANCED,
        "device": "cpu",
        "batch_size": 16,
        "num_threads": 4,
    }

    # Check for GPU
    if torch.cuda.is_available():
        config["device"] = "cuda"
        config["batch_size"] = 32
        config["profile"] = PromptProfile.ACCURATE

    # Check for Apple Silicon
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        config["device"] = "mps"
        config["batch_size"] = 24
        config["profile"] = PromptProfile.BALANCED

    # CPU optimization
    else:
        import multiprocessing

        config["num_threads"] = max(2, multiprocessing.cpu_count() - 1)
        config["batch_size"] = 16
        config["profile"] = PromptProfile.FAST

    return config


# Quick access function
def get_default_config() -> OptimizedPromptConfig:
    """Get default balanced configuration"""
    return get_prompt_config(PromptProfile.BALANCED)


if __name__ == "__main__":
    # Example usage
    print("=== Optimized Prompt Configurations ===\n")

    for profile in PromptProfile:
        config = get_prompt_config(profile)
        print(f"{profile.value.upper()} Profile:")
        print(f"  Generator temperature: {config.generator_temperature}")
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Search threshold: {config.search_threshold}")
        print(f"  Top-K results: {config.search_top_k}")
        print()

    print("\n=== Hardware-Optimized Config ===")
    hw_config = get_optimal_config_for_hardware()
    print(f"Device: {hw_config['device']}")
    print(f"Profile: {hw_config['profile'].value}")
    print(f"Batch size: {hw_config['batch_size']}")