"""Regression checks for native UUIDs; every CUDA query below is mocked."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from large_qubit.gpu_identity import normalize_gpu_uuid


WORKSPACE = Path(__file__).resolve().parents[2]
BARE_UUID = '01234567-89ab-cdef-0123-456789abcdef'
GPU_UUID = 'GPU-' + BARE_UUID
OTHER_UUID = '11234567-89ab-cdef-0123-456789abcdef'
THIRD_UUID = '21234567-89ab-cdef-0123-456789abcdef'


class NativeUuidFixture:
    """The relevant PyTorch _CUuuid contract is its bare-UUID __str__."""

    def __init__(self, label=BARE_UUID):
        self.label = label

    def __str__(self):
        return self.label


class GpuUuidTests(unittest.TestCase):
    def test_bare_prefixed_and_native_object_have_one_canonical_identity(self):
        for value in (BARE_UUID, GPU_UUID, NativeUuidFixture()):
            with self.subTest(value=type(value).__name__):
                self.assertEqual(normalize_gpu_uuid(value), GPU_UUID)

    def test_ascii_case_is_normalized_without_changing_the_identity(self):
        for value in (BARE_UUID.upper(), GPU_UUID.upper(), GPU_UUID.lower(),
                      NativeUuidFixture(BARE_UUID.upper())):
            with self.subTest(value=str(value)):
                self.assertEqual(normalize_gpu_uuid(value), GPU_UUID)

    def test_partial_missing_malformed_zero_and_mig_identities_are_rejected(self):
        values = (None, '', 'unavailable', 'none', 'GPU-fixture', 'GPU-', 'GPU-01234567',
                  BARE_UUID.replace('-', ''), '{' + BARE_UUID + '}', 'urn:uuid:' + BARE_UUID,
                  ' ' + BARE_UUID, BARE_UUID + '\n', BARE_UUID + '0',
                  '01234567-89ab-cdef-0123-456789abcg', 'GPU-GPU-' + BARE_UUID,
                  'MIG-' + BARE_UUID, 'MIG-GPU-' + BARE_UUID + '/1/0',
                  '00000000-0000-0000-0000-000000000000',
                  'GPU-00000000-0000-0000-0000-000000000000', b'GPU-' + BARE_UUID.encode())
        for value in values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'complete nonzero GPU UUID'):
                normalize_gpu_uuid(value)

    def test_import_does_not_import_torch_or_touch_cuda(self):
        script = ('import sys; from large_qubit.gpu_identity import normalize_gpu_uuid; '
                  'assert not ({"torch", "transformers", "peft"} & set(sys.modules)); '
                  f'assert normalize_gpu_uuid({BARE_UUID!r}) == {GPU_UUID!r}')
        subprocess.run([sys.executable, '-B', '-c', script], cwd=WORKSPACE, check=True)


def _assignment_source():
    source = (WORKSPACE / 'large_qubit/scripts/run_section7_tmux.sh').read_text(encoding='utf-8')
    return source.split('        assignment_output=', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY\n)', 1)[0]


def _resolve_launcher(devices, ordinals=('0', '1', '2'), *, available=True):
    cuda = SimpleNamespace(is_available=Mock(return_value=available),
                           device_count=Mock(return_value=len(devices)),
                           get_device_properties=Mock(side_effect=lambda index: devices[index]))
    output = io.StringIO()
    with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}), \
         patch.object(sys, 'argv', ['-', *ordinals]), redirect_stdout(output):
        exec(compile(_assignment_source(), 'actual_section7_launcher_uuid_query', 'exec'), {})
    return output.getvalue().splitlines(), cuda


def _devices():
    return [SimpleNamespace(name='NVIDIA H200', uuid=NativeUuidFixture(BARE_UUID.upper())),
            SimpleNamespace(name='NVIDIA H200 NVL', uuid='GPU-' + OTHER_UUID),
            SimpleNamespace(name='NVIDIA H200', uuid=THIRD_UUID)]


class LauncherUuidTests(unittest.TestCase):
    def test_actual_launcher_resolves_native_and_prefixed_uuid_strings(self):
        result, cuda = _resolve_launcher(_devices())
        self.assertEqual(result, [GPU_UUID, 'GPU-' + OTHER_UUID, 'GPU-' + THIRD_UUID])
        self.assertEqual(cuda.get_device_properties.call_count, 3)

    def test_actual_launcher_rejects_same_device_after_normalization(self):
        devices = _devices()
        devices[1].uuid = GPU_UUID
        with self.assertRaisesRegex(ValueError, 'distinct GPUs'):
            _resolve_launcher(devices)

    def test_actual_launcher_model_error_identifies_the_selected_wrong_device(self):
        devices = _devices()
        devices[1].name = 'NVIDIA A100-SXM4-80GB'
        with self.assertRaisesRegex(ValueError, 'cuda:1.*H200.*A100-SXM4-80GB'):
            _resolve_launcher(devices)

    def test_actual_launcher_uuid_error_identifies_device_and_raw_value(self):
        for value in (None, 'unavailable', 'GPU-short', 'MIG-' + BARE_UUID):
            devices = _devices()
            devices[1].uuid = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'cuda:1.*invalid GPU UUID') as raised:
                _resolve_launcher(devices)
            self.assertIn(repr(str(value)), str(raised.exception))

    def test_actual_launcher_still_requires_cuda_and_allocation_ordinals(self):
        with self.assertRaisesRegex(ValueError, 'CUDA is unavailable'):
            _resolve_launcher(_devices(), available=False)
        with self.assertRaisesRegex(ValueError, 'outside the visible allocation'):
            _resolve_launcher(_devices(), ordinals=('0', '1', '3'))


class RuntimeUuidTests(unittest.TestCase):
    def setUp(self):
        import torch
        from large_qubit import training
        self.training = training
        self.properties = SimpleNamespace(name='NVIDIA H200', uuid=NativeUuidFixture(BARE_UUID.upper()),
                                          major=9, minor=0)
        output = SimpleNamespace(stdout=f'{os.getpid()}, {GPU_UUID}, 32\n43210, {GPU_UUID}, 64\n'
                                        f'999, GPU-{OTHER_UUID}, 12\n')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.count = self.stack.enter_context(patch.object(torch.cuda, 'device_count', return_value=1))
        self.stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
        self.stack.enter_context(patch.object(torch.cuda, 'device', return_value=nullcontext()))
        self.stack.enter_context(patch.object(torch.cuda, 'get_device_properties', return_value=self.properties))
        self.bf16 = self.stack.enter_context(patch.object(torch.cuda, 'is_bf16_supported', return_value=True))
        self.stack.enter_context(patch.object(torch.cuda, 'mem_get_info', return_value=(1000, 2000)))
        self.stack.enter_context(patch.object(torch.backends.cudnn, 'version', return_value=None))
        self.query = self.stack.enter_context(patch.object(training.subprocess, 'run', return_value=output))

    def test_actual_environment_canonicalizes_native_uuid_and_matches_sharing_records(self):
        record = self.training.environment('cuda:0')
        self.assertEqual(record['gpu_uuid'], GPU_UUID)
        self.assertEqual(record['visible_gpu_count'], 1)
        self.assertEqual(record['other_compute_processes_visible'], [
            {'pid': 43210, 'gpu_uuid': GPU_UUID, 'used_memory_mib': 64}])
        self.bf16.assert_called_once_with(including_emulation=False)
        self.assertNotIn('command', str(record))
        self.assertEqual(self.query.call_args.args[0],
                         ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                          '--format=csv,noheader,nounits'])

    def test_actual_environment_accepts_existing_prefixed_uuid(self):
        self.properties.uuid = GPU_UUID
        self.assertEqual(self.training.environment('cuda:0')['gpu_uuid'], GPU_UUID)

    def test_actual_environment_rejects_invalid_uuid_before_sharing_query(self):
        for value in (None, 'GPU-fixture', 'GPU-00000000-0000-0000-0000-000000000000'):
            self.properties.uuid = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'UUID.*cuda:0') as raised:
                self.training.environment('cuda:0')
            self.assertIn(repr(str(value)), str(raised.exception))
        self.query.assert_not_called()

    def test_actual_environment_preserves_single_h200_and_native_bf16_guards(self):
        self.count.return_value = 2
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.training.environment('cuda:0')
        self.count.return_value = 1
        self.properties.name = 'NVIDIA A100'
        with self.assertRaisesRegex(ValueError, 'native BF16 on an H200'):
            self.training.environment('cuda:0')
        self.properties.name = 'NVIDIA H200'
        self.bf16.return_value = False
        with self.assertRaisesRegex(ValueError, 'native BF16 on an H200'):
            self.training.environment('cuda:0')
        self.query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
