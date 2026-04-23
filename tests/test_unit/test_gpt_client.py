# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
from unittest.mock import patch

import pytest
import yaml
from PIL import Image
from asset_gen.utils.gpt_clients import CONFIG_FILE, GPTclient


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def env_vars(monkeypatch, config):
    agent_type = config["agent_type"]
    agent_config = config.get(agent_type, {})
    monkeypatch.setenv(
        "ENDPOINT", agent_config.get("endpoint", "fake_endpoint")
    )
    monkeypatch.setenv("API_KEY", agent_config.get("api_key", "fake_api_key"))
    monkeypatch.setenv("API_VERSION", agent_config.get("api_version", "v1"))
    monkeypatch.setenv(
        "MODEL_NAME", agent_config.get("model_name", "test_model")
    )
    yield


@pytest.fixture
def gpt_client(env_vars):
    client = GPTclient(
        endpoint=os.environ.get("ENDPOINT"),
        api_key=os.environ.get("API_KEY"),
        api_version=os.environ.get("API_VERSION"),
        model_name=os.environ.get("MODEL_NAME"),
        check_connection=False,
    )
    return client


@pytest.mark.parametrize(
    "text_prompt, image_base64",
    [
        ("What is the capital of China?", None),
        (
            "What is the content in each image?",
            "apps/assets/example_image/sample_02.jpg",
        ),
        (
            "What is the content in each image?",
            [
                "apps/assets/example_image/sample_02.jpg",
                "apps/assets/example_image/sample_03.jpg",
            ],
        ),
        (
            "What is the content in each image?",
            [
                Image.new("RGB", (64, 64), "red"),
                Image.new("RGB", (64, 64), "blue"),
            ],
        ),
    ],
)
def test_gptclient_query(gpt_client, text_prompt, image_base64):
    # mock GPTclient.query
    with patch.object(
        GPTclient, "query", return_value="mocked response"
    ) as mock_query:
        response = gpt_client.query(
            text_prompt=text_prompt, image_base64=image_base64
        )
        assert response == "mocked response"
        mock_query.assert_called_once_with(
            text_prompt=text_prompt, image_base64=image_base64
        )
