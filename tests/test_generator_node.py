from src.agent.nodes.generator_node import GeneratorNode

def test_generator_handles_no_documents():


    generator = GeneratorNode()
    result = generator.generate_answer(
        question="Test question",
        documents=[]
    )

    assert result["answer"] is not None
    assert result["confidence"] == 0.0
    assert result["sources_used"] == 0