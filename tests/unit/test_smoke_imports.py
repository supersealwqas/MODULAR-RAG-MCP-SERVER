"""Smoke tests: verify all top-level packages are importable."""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """Ensure project directory structure is correct by importing top-level packages."""

    def test_import_src(self):
        import src
        assert src is not None

    def test_import_mcp_server(self):
        from src import mcp_server
        assert mcp_server is not None

    def test_import_core(self):
        from src import core
        assert core is not None

    def test_import_ingestion(self):
        from src import ingestion
        assert ingestion is not None

    def test_import_libs(self):
        from src import libs
        assert libs is not None

    def test_import_observability(self):
        from src import observability
        assert observability is not None

    def test_import_core_submodules(self):
        from src.core import query_engine
        from src.core import response
        from src.core import trace
        assert query_engine is not None
        assert response is not None
        assert trace is not None

    def test_import_libs_submodules(self):
        from src.libs import loader
        from src.libs import llm
        from src.libs import embedding
        from src.libs import splitter
        from src.libs import vector_store
        from src.libs import reranker
        from src.libs import evaluator
        assert all([loader, llm, embedding, splitter, vector_store, reranker, evaluator])

    def test_import_ingestion_submodules(self):
        from src.ingestion import chunking
        from src.ingestion import transform
        from src.ingestion import embedding
        from src.ingestion import storage
        assert all([chunking, transform, embedding, storage])

    def test_config_directory_exists(self, tmp_path):
        from pathlib import Path
        config_dir = Path("config")
        assert config_dir.exists()
        assert (config_dir / "settings.yaml").exists()
        assert (config_dir / "prompts").exists()

    def test_prompt_files_readable(self):
        from pathlib import Path
        prompts_dir = Path("config/prompts")
        for prompt_file in prompts_dir.glob("*.txt"):
            content = prompt_file.read_text(encoding="utf-8")
            assert len(content) > 0, f"{prompt_file.name} is empty"
