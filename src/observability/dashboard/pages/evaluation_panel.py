"""评估面板页面。

运行评估、查看指标、历史对比。
支持选择评估后端与 golden test set。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def _get_eval_runner():
    """获取 EvalRunner 实例（延迟创建）。

    返回:
        EvalRunner 实例或 None（创建失败时）
    """
    try:
        from src.core.settings import load_settings
        from src.observability.evaluation.eval_runner import EvalRunner

        settings = load_settings()
        return EvalRunner(settings)
    except Exception as e:
        st.error(f"创建 EvalRunner 失败: {e}")
        return None


def _load_golden_test_sets() -> List[str]:
    """扫描可用的 golden test set 文件。

    返回:
        文件路径列表
    """
    fixtures_dir = Path("tests/fixtures")
    if not fixtures_dir.exists():
        return []

    test_sets = []
    for f in fixtures_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                # 验证格式
                if isinstance(data, dict) and "test_cases" in data:
                    test_sets.append(str(f))
                elif isinstance(data, list):
                    test_sets.append(str(f))
        except (json.JSONDecodeError, Exception):
            continue

    return test_sets


def _load_evaluation_history() -> List[Dict[str, Any]]:
    """加载历史评估结果。

    返回:
        历史评估记录列表
    """
    history_file = Path("logs/evaluation_history.jsonl")
    if not history_file.exists():
        return []

    records = []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    return records


def _save_evaluation_result(result: Dict[str, Any]) -> None:
    """保存评估结果到历史记录。

    参数:
        result: 评估结果字典
    """
    history_file = Path("logs/evaluation_history.jsonl")
    history_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        st.warning(f"保存评估历史失败: {e}")


def _render_metrics_chart(summary: Dict[str, float]) -> None:
    """渲染指标柱状图。

    参数:
        summary: 汇总指标字典
    """
    if not summary:
        st.info("暂无指标数据")
        return

    # 准备图表数据
    metrics = list(summary.keys())
    values = list(summary.values())

    # 使用 Streamlit 原生柱状图
    chart_data = {
        "指标": metrics,
        "值": values,
    }
    st.bar_chart(chart_data, x="指标", y="值", height=300)


def _render_query_details(results: List[Dict[str, Any]]) -> None:
    """渲染各 query 的详细结果。

    参数:
        results: 评估结果列表
    """
    if not results:
        st.info("暂无详细结果")
        return

    for i, result in enumerate(results):
        query = result.get("query", "?")
        metrics = result.get("metrics", {})
        error = result.get("metadata", {}).get("error")

        with st.expander(f"[{i + 1}] {query[:60]}..."):
            if error:
                st.error(f"评估错误: {error}")
            else:
                cols = st.columns(len(metrics) if metrics else 1)
                for j, (metric, value) in enumerate(metrics.items()):
                    with cols[j]:
                        st.metric(label=metric, value=f"{value:.4f}")


def render_evaluation_panel(config_service: ConfigService) -> None:
    """渲染评估面板页面。

    参数:
        config_service: 配置服务实例
    """
    st.title("📈 评估面板")
    st.markdown("---")

    # === 评估配置 ===
    st.subheader("⚙️ 评估配置")

    col1, col2 = st.columns(2)

    with col1:
        # 选择评估后端
        settings = config_service.get_settings()
        available_backends = ["custom"]
        try:
            from src.libs.evaluator.evaluator_factory import EvaluatorFactory
            available_backends = EvaluatorFactory.list_providers()
        except Exception:
            pass

        selected_backend = st.selectbox(
            "评估后端",
            options=available_backends,
            index=0,
            help="选择评估器后端（custom 为内置轻量指标，ragas 需安装）",
        )

    with col2:
        # 选择 golden test set
        test_sets = _load_golden_test_sets()
        if not test_sets:
            st.warning("未找到 golden test set 文件，请先在 tests/fixtures/ 下创建 JSON 文件")
            selected_test_set = None
        else:
            selected_test_set = st.selectbox(
                "Golden Test Set",
                options=test_sets,
                format_func=lambda x: Path(x).name,
                help="选择黄金测试集文件",
            )

    # 高级选项
    with st.expander("🔧 高级选项"):
        col3, col4 = st.columns(2)
        with col3:
            top_k = st.number_input("Top-K", min_value=1, max_value=100, value=10)
        with col4:
            collection = st.text_input("限定集合（可选）", value="")

    st.markdown("---")

    # === 运行评估 ===
    st.subheader("▶️ 运行评估")

    if st.button("🚀 开始评估", type="primary", disabled=selected_test_set is None):
        if selected_test_set is None:
            st.error("请选择 golden test set")
            return

        with st.spinner("正在执行评估..."):
            try:
                runner = _get_eval_runner()
                if runner is None:
                    return

                report = runner.run(
                    test_set_path=selected_test_set,
                    top_k=top_k,
                    collection=collection or None,
                    verbose=True,
                )

                # 保存结果到历史
                result_record = {
                    "timestamp": datetime.now().isoformat(),
                    "backend": selected_backend,
                    "test_set": Path(selected_test_set).name,
                    "top_k": top_k,
                    "collection": collection or None,
                    "total_cases": report.total_cases,
                    "summary": report.summary,
                }
                _save_evaluation_result(result_record)

                # 缓存当前结果
                st.session_state["last_eval_report"] = {
                    "total_cases": report.total_cases,
                    "summary": report.summary,
                    "results": [
                        {
                            "query": r.query,
                            "metrics": r.metrics,
                            "metadata": r.metadata,
                        }
                        for r in report.results
                    ],
                }

                st.success(f"评估完成！共 {report.total_cases} 条用例")

            except Exception as e:
                st.error(f"评估执行失败: {e}")
                return

    st.markdown("---")

    # === 评估结果 ===
    st.subheader("📊 评估结果")

    # 显示最近一次评估结果
    last_report = st.session_state.get("last_eval_report")
    if last_report:
        # 汇总指标
        st.markdown("**汇总指标:**")
        _render_metrics_chart(last_report.get("summary", {}))

        # 指标卡片
        summary = last_report.get("summary", {})
        if summary:
            cols = st.columns(len(summary))
            for i, (metric, value) in enumerate(summary.items()):
                with cols[i]:
                    st.metric(label=metric, value=f"{value:.4f}")

        st.markdown("---")

        # 详细结果
        st.markdown("**各 Query 详细结果:**")
        _render_query_details(last_report.get("results", []))
    else:
        st.info("暂无评估结果，请先运行评估")

    st.markdown("---")

    # === 历史对比 ===
    st.subheader("📜 历史评估记录")

    history = _load_evaluation_history()
    if history:
        # 显示最近 10 条记录
        recent_history = history[-10:]
        for record in reversed(recent_history):
            timestamp = record.get("timestamp", "?")
            backend = record.get("backend", "?")
            test_set = record.get("test_set", "?")
            total_cases = record.get("total_cases", 0)
            summary = record.get("summary", {})

            with st.expander(f"{timestamp} | {backend} | {test_set} ({total_cases} 条)"):
                if summary:
                    cols = st.columns(len(summary))
                    for i, (metric, value) in enumerate(summary.items()):
                        with cols[i]:
                            st.metric(label=metric, value=f"{value:.4f}")
                else:
                    st.info("无汇总指标")
    else:
        st.info("暂无历史评估记录")
