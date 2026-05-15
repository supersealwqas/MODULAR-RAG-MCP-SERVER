"""冒烟测试：验证所有顶层包可正常导入。"""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """确保项目目录结构正确，通过导入顶层包验证。"""

    def test_import_src(self):
        """验证 src 包可导入。"""
        import src
        assert src is not None

    def test_import_mcp_server(self):
        """验证 mcp_server 包可导入。"""
        from src import mcp_server
        assert mcp_server is not None

    def test_import_core(self):
        """验证 core 包可导入。"""
        from src import core
        assert core is not None

    def test_import_ingestion(self):
        """验证 ingestion 包可导入。"""
        from src import ingestion
        assert ingestion is not None

    def test_import_libs(self):
        """验证 libs 包可导入。"""
        from src import libs
        assert libs is not None

    def test_import_observability(self):
        """验证 observability 包可导入。"""
        from src import observability
        assert observability is not None

    def test_import_core_submodules(self):
        """验证 core 子模块可导入。"""
        from src.core import query_engine
        from src.core import response
        from src.core import trace
        assert query_engine is not None
        assert response is not None
        assert trace is not None

    def test_import_libs_submodules(self):
        """验证 libs 子模块可导入。"""
        from src.libs import loader
        from src.libs import llm
        from src.libs import embedding
        from src.libs import splitter
        from src.libs import vector_store
        from src.libs import reranker
        from src.libs import evaluator
        assert all([loader, llm, embedding, splitter, vector_store, reranker, evaluator])

    def test_import_ingestion_submodules(self):
        """验证 ingestion 子模块可导入。"""
        from src.ingestion import chunking
        from src.ingestion import transform
        from src.ingestion import embedding
        from src.ingestion import storage
        assert all([chunking, transform, embedding, storage])

    def test_config_directory_exists(self):
        """验证配置目录存在。"""
        from pathlib import Path
        config_dir = Path("config")
        assert config_dir.exists()
        assert (config_dir / "settings.yaml").exists()
        assert (config_dir / "prompts").exists()

    def test_prompt_files_readable(self):
        """验证提示词文件可读取且非空。"""
        from pathlib import Path
        prompts_dir = Path("config/prompts")
        for prompt_file in prompts_dir.glob("*.txt"):
            content = prompt_file.read_text(encoding="utf-8")
            assert len(content) > 0, f"{prompt_file.name} 内容为空"
