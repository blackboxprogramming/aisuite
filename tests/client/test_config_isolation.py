"""Provider configuration ownership checks; no provider calls or credentials needed."""

import unittest
from unittest.mock import Mock, patch

from aisuite import Client


class ClientConfigurationIsolationTests(unittest.TestCase):
    def test_default_clients_do_not_inherit_another_clients_configuration(self):
        first = Client()
        second = Client()
        first.configure({"ollama": {"base_url": "http://127.0.0.1:11434"}})
        self.assertEqual(second.provider_configs, {})
        self.assertEqual(Client().provider_configs, {})

    def test_explicit_none_keeps_an_empty_client(self):
        self.assertEqual(Client(provider_configs=None).provider_configs, {})

    def test_configure_does_not_mutate_the_constructor_input(self):
        original = {"ollama": {"base_url": "http://127.0.0.1:11434"}}
        client = Client(original)
        client.configure({"other": {"option": "new"}})
        self.assertNotIn("other", original)

    def test_clients_with_the_same_input_own_separate_provider_mappings(self):
        original = {"ollama": {"base_url": "http://127.0.0.1:11434"}}
        first = Client(original)
        second = Client(original)
        first.provider_configs["ollama"]["base_url"] = "http://127.0.0.1:11435"
        self.assertEqual(second.provider_configs["ollama"], original["ollama"])
        self.assertEqual(original["ollama"]["base_url"], "http://127.0.0.1:11434")

    def test_constructor_snapshots_the_supplied_provider_mapping(self):
        original = {"ollama": {"base_url": "http://127.0.0.1:11434"}}
        client = Client(original)
        original["ollama"]["base_url"] = "http://127.0.0.1:11435"
        original["other"] = {}
        self.assertEqual(
            client.provider_configs, {"ollama": {"base_url": "http://127.0.0.1:11434"}}
        )

    def test_configure_snapshots_the_supplied_provider_mapping(self):
        client = Client()
        config = {"ollama": {"base_url": "http://127.0.0.1:11434"}}
        client.configure(config)
        config["ollama"]["base_url"] = "http://127.0.0.1:11435"
        config.clear()
        self.assertEqual(
            client.provider_configs, {"ollama": {"base_url": "http://127.0.0.1:11434"}}
        )

    def test_opaque_provider_values_are_preserved_without_deepcopy(self):
        class OpaqueClient:
            def __deepcopy__(self, memo):
                raise AssertionError("SDK values must not be deep-copied")

        transport = OpaqueClient()
        client = Client({"ollama": {"http_client": transport}})
        self.assertIs(client.provider_configs["ollama"]["http_client"], transport)
        client.configure({"other": {"http_client": transport}})
        self.assertIs(client.provider_configs["other"]["http_client"], transport)

    def test_configure_none_is_a_noop_and_updates_preserve_other_providers(self):
        client = Client({"ollama": {"option": "original"}})
        client.configure(None)
        client.configure({"other": {"option": "added"}})
        self.assertEqual(
            client.provider_configs,
            {"ollama": {"option": "original"}, "other": {"option": "added"}},
        )

    def test_chat_provider_receives_only_its_own_clients_settings(self):
        first = Client()
        first.configure({"ollama": {"base_url": "http://127.0.0.1:11435"}})
        second = Client()
        provider = Mock()
        expected_response = object()
        provider.chat_completions_create.return_value = expected_response
        messages = [{"role": "user", "content": "hello"}]
        with patch(
            "aisuite.client.ProviderFactory.get_supported_providers",
            return_value={"ollama"},
        ), patch(
            "aisuite.client.ProviderFactory.create_provider",
            return_value=provider,
        ) as create:
            result = second.chat.completions.create("ollama:test-model", messages)
        create.assert_called_once_with("ollama", {})
        provider.chat_completions_create.assert_called_once_with("test-model", messages)
        self.assertIs(result, expected_response)


if __name__ == "__main__":
    unittest.main()
