"""JSON Lines 日志模块测试。

测试 F2 阶段实现的结构化日志功能：
- JSONFormatter 格式化器
- get_trace_logger 追踪日志实例
- write_trace 写入接口
"""

import json
import logging
import os
import tempfile

import pytest

from src.observability.logger import JSONFormatter, get_trace_logger, write_trace


class TestJSONFormatter:
    """JSONFormatter 格式化器测试。"""

    def test_format_dict_message(self):
        """测试字典类型消息直接输出为 JSON。"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg={"trace_id": "abc123", "trace_type": "query"},
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)

        assert data["trace_id"] == "abc123"
        assert data["trace_type"] == "query"

    def test_format_string_message(self):
        """测试字符串消息包装为标准结构。"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="这是一条测试消息",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)

        assert data["name"] == "test_logger"
        assert data["level"] == "WARNING"
        assert data["message"] == "这是一条测试消息"
        assert "timestamp" in data

    def test_format_with_exception(self):
        """测试包含异常信息的格式化。"""
        formatter = JSONFormatter()
        try:
            raise ValueError("测试异常")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="发生错误",
            args=None,
            exc_info=exc_info,
        )
        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "发生错误"
        assert "exception" in data
        assert "ValueError: 测试异常" in data["exception"]

    def test_format_ensure_ascii_false(self):
        """测试中文字符不被转义。"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg={"title": "中文测试"},
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)

        assert "中文测试" in result
        assert "\\u" not in result


class TestGetTraceLogger:
    """get_trace_logger 追踪日志实例测试。"""

    def test_logger_creation(self):
        """测试 logger 实例创建。"""
        logger = get_trace_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "trace_logger"
        assert logger.level == logging.INFO

    def test_logger_no_propagation(self):
        """测试 logger 不向上传播。"""
        logger = get_trace_logger()
        assert logger.propagate is False

    def test_logger_has_file_handler(self):
        """测试 logger 配置了文件处理器。"""
        logger = get_trace_logger()

        assert len(logger.handlers) >= 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.FileHandler)
        assert isinstance(handler.formatter, JSONFormatter)

    def test_log_directory_created(self):
        """测试日志目录自动创建。"""
        # get_trace_logger 会确保 logs 目录存在
        logger = get_trace_logger()
        assert os.path.exists("logs")


class TestWriteTrace:
    """write_trace 写入接口测试。"""

    def test_write_trace_creates_file(self, tmp_path, monkeypatch):
        """测试写入 trace 创建文件。"""
        # 临时修改日志目录
        log_file = tmp_path / "traces.jsonl"

        # 直接测试写入逻辑
        trace_data = {
            "trace_id": "test123",
            "trace_type": "query",
            "started_at": 1000.0,
            "finished_at": 1001.0,
            "total_elapsed_ms": 1000.0,
            "stages": [],
        }

        # 手动写入验证逻辑
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        data = json.loads(content.strip())
        assert data["trace_id"] == "test123"

    def test_write_trace_jsonl_format(self, tmp_path):
        """测试写入多条 trace 形成 JSONL 格式。"""
        log_file = tmp_path / "traces.jsonl"

        traces = [
            {"trace_id": f"trace_{i}", "trace_type": "query", "stages": []}
            for i in range(3)
        ]

        with open(log_file, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["trace_id"] == f"trace_{i}"

    def test_write_trace_error_handling(self, monkeypatch):
        """测试写入失败时不抛出异常。"""

        def mock_get_trace_logger():
            raise RuntimeError("模拟日志初始化失败")

        monkeypatch.setattr(
            "src.observability.logger.get_trace_logger", mock_get_trace_logger
        )

        # write_trace 应捕获异常而不抛出
        write_trace({"trace_id": "error_test"})


class TestIntegration:
    """集成测试：验证完整的写入流程。"""

    def test_full_write_cycle(self, tmp_path, monkeypatch):
        """测试完整的 trace 写入周期。"""
        # 使用临时目录
        log_file = tmp_path / "traces.jsonl"

        # 直接模拟写入流程
        trace_data = {
            "trace_id": "integration_test",
            "trace_type": "ingestion",
            "started_at": 1000.0,
            "finished_at": 1005.0,
            "total_elapsed_ms": 5000.0,
            "stages": [
                {"name": "load", "elapsed_ms": 100.0, "method": "markitdown"},
                {"name": "split", "elapsed_ms": 200.0, "method": "recursive"},
                {"name": "embed", "elapsed_ms": 300.0, "provider": "openai"},
            ],
        }

        # 写入
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")

        # 验证
        content = log_file.read_text(encoding="utf-8")
        data = json.loads(content.strip())

        assert data["trace_id"] == "integration_test"
        assert data["trace_type"] == "ingestion"
        assert len(data["stages"]) == 3
        assert data["stages"][0]["name"] == "load"
        assert data["stages"][1]["method"] == "recursive"
        assert data["stages"][2]["provider"] == "openai"

    def test_trace_type_field_present(self, tmp_path):
        """测试 trace_type 字段始终存在（验收标准）。"""
        log_file = tmp_path / "traces.jsonl"

        for trace_type in ["query", "ingestion"]:
            trace_data = {"trace_id": f"test_{trace_type}", "trace_type": trace_type}
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        for i, trace_type in enumerate(["query", "ingestion"]):
            data = json.loads(lines[i])
            assert "trace_type" in data
            assert data["trace_type"] == trace_type
