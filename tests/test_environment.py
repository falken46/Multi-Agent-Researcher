def test_core_dependencies_importable() -> None:
    import fastapi
    import langgraph
    import openai
    import streamlit

    assert fastapi is not None
    assert langgraph is not None
    assert openai is not None
    assert streamlit is not None
