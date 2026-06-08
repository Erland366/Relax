# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Integration tests for data_source.py and eager Dataset global-slice
semantics.

Extracted from test_streaming_dataset.py during tests/ directory
restructuring.

Run with: pytest tests/engine/rollout/test_data_source.py -v
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest


class TestEagerDataset:
    """Tests for eager Dataset global-slice semantics."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        return tokenizer

    def test_dataset_multi_file_global_slice(self, mock_tokenizer):
        from relax.utils.data.data import Dataset

        data1 = [{"text": f"A{i}", "label": f"a{i}"} for i in range(3)]
        data2 = [{"text": f"B{i}", "label": f"b{i}"} for i in range(3)]
        files = []
        try:
            for data in (data1, data2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                    files.append(f.name)

            path = f"[{files[0]},{files[1]}]@[1:5]"
            dataset = Dataset(
                path=path,
                tokenizer=mock_tokenizer,
                processor=None,
                max_length=None,
                prompt_key="text",
                label_key="label",
            )

            assert len(dataset) == 4
            prompts = [dataset[i].prompt for i in range(len(dataset))]
            assert prompts == ["A1", "A2", "B0", "B1"]
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)


class TestDataSourceIntegration:
    """Integration tests for data_source.py with StreamingDataset."""

    def test_data_source_setup_loads_inputs_directly(self, monkeypatch):
        from relax.engine.rollout import data_source as data_source_module

        events = []

        def fake_load_tokenizer(*args, **kwargs):
            events.append("tokenizer")
            return MagicMock()

        def fake_load_processor(*args, **kwargs):
            events.append("processor")
            return None

        def fake_create_dataset(*args, **kwargs):
            events.append("dataset")
            return MagicMock()

        monkeypatch.setattr(data_source_module, "load_tokenizer", fake_load_tokenizer)
        monkeypatch.setattr(data_source_module, "load_processor", fake_load_processor)
        monkeypatch.setattr(data_source_module, "_create_dataset", fake_create_dataset)

        args = MagicMock()
        args.sglang_model_impl = "transformers"
        args.rollout_global_dataset = True
        args.hf_checkpoint = "/tmp/checkpoint"
        args.dump_details = None
        args.rollout_shuffle = False

        data_source_module.RolloutDataSource(args)

        assert events == ["tokenizer", "processor", "dataset"]

    def test_build_data_source_config_keeps_only_rollout_fields(self):
        from argparse import Namespace

        from relax.engine.rollout import data_source as data_source_module

        args = Namespace(
            sglang_model_impl="transformers",
            rollout_global_dataset=True,
            hf_checkpoint="/tmp/checkpoint",
            dump_details=None,
            rollout_shuffle=True,
            use_streaming_dataset=True,
            streaming_buffer_size=10000,
            prompt_data="/tmp/prompts.jsonl",
            rollout_max_prompt_len=4096,
            input_key="prompt",
            multimodal_keys=None,
            label_key="label",
            tool_key="tools",
            metadata_key="metadata",
            system_prompt=None,
            apply_chat_template=True,
            apply_chat_template_kwargs={},
            use_audio_in_video=False,
            rollout_seed=42,
            n_samples_per_prompt=8,
            buffer_filter_path=None,
            image_max_token_num=None,
            image_min_token_num=None,
            video_min_token_num=None,
            video_max_token_num=None,
            video_fps=None,
            video_fps_min_frames=None,
            video_fps_max_frames=None,
            frame_factor=None,
            audio_sample_rate=None,
            save="/tmp/save",
            load="/tmp/load",
            tq_config=object(),
            megatron_enum=object(),
        )

        config = data_source_module.build_data_source_config(args)

        assert isinstance(config, Namespace)
        assert config.prompt_data == "/tmp/prompts.jsonl"
        assert config.save == "/tmp/save"
        assert config.load == "/tmp/load"
        assert not hasattr(config, "tq_config")
        assert not hasattr(config, "megatron_enum")

    def test_rollout_data_source_loads_tokenizer_before_dataset(self, monkeypatch):
        from relax.engine.rollout import data_source as data_source_module

        call_order = []

        def fake_load_tokenizer(*args, **kwargs):
            call_order.append("tokenizer")
            return MagicMock()

        def fake_load_processor(*args, **kwargs):
            call_order.append("processor")
            return None

        def fake_create_dataset(*args, **kwargs):
            call_order.append("dataset")
            return MagicMock()

        monkeypatch.setattr(data_source_module, "load_tokenizer", fake_load_tokenizer)
        monkeypatch.setattr(data_source_module, "load_processor", fake_load_processor)
        monkeypatch.setattr(data_source_module, "_create_dataset", fake_create_dataset)

        args = MagicMock()
        args.sglang_model_impl = "transformers"
        args.rollout_global_dataset = True
        args.hf_checkpoint = "/tmp/checkpoint"
        args.dump_details = None
        args.rollout_shuffle = False

        data_source_module.RolloutDataSource(args)

        assert call_order == ["tokenizer", "processor", "dataset"]

    @pytest.fixture
    def jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        data = [{"text": f"Sample {i}", "label": f"label_{i}"} for i in range(10)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            filepath = f.name

        yield filepath, data
        os.unlink(filepath)

    def test_factory_function_streaming(self, jsonl_file):
        """Test _create_dataset factory with streaming enabled."""
        from relax.engine.rollout.data_source import _create_dataset
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file

        args = MagicMock()
        args.use_streaming_dataset = True
        args.streaming_buffer_size = 100
        args.prompt_data = filepath
        args.rollout_max_prompt_len = None
        args.input_key = "text"
        args.multimodal_keys = None
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.system_prompt = None
        args.tool_key = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = None
        args.rollout_seed = 42
        args.custom_prompt_path = None

        tokenizer = MagicMock()

        dataset = _create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, StreamingDataset)
        assert len(dataset) == len(data)

    def test_factory_function_traditional(self, jsonl_file):
        """Test _create_dataset factory with streaming disabled."""
        from relax.engine.rollout.data_source import _create_dataset
        from relax.utils.data.data import Dataset

        filepath, data = jsonl_file

        args = MagicMock()
        args.use_streaming_dataset = False
        args.prompt_data = filepath
        args.rollout_max_prompt_len = None
        args.input_key = "text"
        args.multimodal_keys = None
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.system_prompt = None
        args.tool_key = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = None
        args.rollout_seed = 42
        args.custom_prompt_path = None

        tokenizer = MagicMock()

        dataset = _create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, Dataset)

    def test_factory_function_streaming_multi_file_slice(self):
        """Test _create_dataset factory with streaming dataset over multiple
        files and outer slice."""
        from relax.engine.rollout.data_source import _create_dataset
        from relax.utils.data.streaming_dataset import StreamingDataset

        data1 = [{"text": f"A{i}", "label": f"a{i}"} for i in range(3)]
        data2 = [{"text": f"B{i}", "label": f"b{i}"} for i in range(3)]
        files = []
        try:
            for data in (data1, data2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                    files.append(f.name)

            args = MagicMock()
            args.use_streaming_dataset = True
            args.streaming_buffer_size = 100
            args.prompt_data = f"[{files[0]},{files[1]}]@[2:6]"
            args.rollout_max_prompt_len = None
            args.input_key = "text"
            args.multimodal_keys = None
            args.label_key = "label"
            args.metadata_key = "metadata"
            args.system_prompt = None
            args.tool_key = None
            args.apply_chat_template = False
            args.apply_chat_template_kwargs = None
            args.rollout_seed = 42
            args.custom_prompt_path = None

            tokenizer = MagicMock()
            dataset = _create_dataset(args, tokenizer, processor=None)

            assert isinstance(dataset, StreamingDataset)
            assert len(dataset) == 4
            prompts = [dataset[i].prompt for i in range(len(dataset))]
            assert prompts == ["A2", "B0", "B1", "B2"]
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
