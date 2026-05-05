def test_core_dependencies_importable() -> None:
    import anthropic
    import fastapi
    import langgraph
    import streamlit

    assert anthropic is not None
    assert fastapi is not None
    assert langgraph is not None
    assert streamlit is not None
