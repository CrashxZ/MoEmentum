#!/usr/bin/env python3
"""CPU-only regression: execute the real parity CMake block in a tiny target graph.

This deliberately does not configure the CUDA/HIP toolchains or build GPU code.
Pass --cmake and --work-dir to retain generated graph/CTest receipts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARGS = argparse.Namespace()


class ParityConfiguration(unittest.TestCase):
    def configure(self, name, backend, mmq, kquants=False, q6=False):
        source = (ROOT / 'CMakeLists.txt').read_text()
        start = source.index('    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)')
        end = source.index('    add_executable(native_expert_bench', start)
        block = source[start:end]
        directory = ARGS.work_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'dummy.cpp').write_text('int main() { return 0; }\n')
        # Use real CMake commands, preserving target name and all block conditions.
        (directory / 'src/kernels').mkdir(parents=True, exist_ok=True)
        (directory / 'src/kernels/native_expert_parity.cpp').write_text('int main() { return 0; }\n')
        prelude = '''cmake_minimum_required(VERSION 3.20)
project(parity_config LANGUAGES CXX)
enable_testing()
foreach(dep strata_engine strata_kernels strata_kernels_cpu strata_artifact runtime)
  add_library(${dep} INTERFACE)
endforeach()
add_executable(native_grouped_parity dummy.cpp)
set(_strata_gpu_runtime_target runtime)
'''
        for key, value in dict(STRATA_ENABLE_CUDA=backend == 'CUDA',
                               STRATA_ENABLE_HIP=backend == 'HIP', STRATA_BUILD_TESTS=True,
                               STRATA_MMQ_KQUANTS=kquants, STRATA_Q6K_EXPERTS=q6).items():
            prelude += f'set({key} {"ON" if value else "OFF"})\n'
        if mmq:
            prelude += 'add_library(strata_mmq INTERFACE)\n'
        postlude = '''
get_target_property(links native_expert_parity LINK_LIBRARIES)
get_target_property(defs native_expert_parity COMPILE_DEFINITIONS)
file(WRITE "${CMAKE_BINARY_DIR}/target-properties.txt" "${links}\\n${defs}\\n")
'''
        (directory / 'CMakeLists.txt').write_text(prelude + block + postlude)
        build = directory / 'build'
        self.run_command([ARGS.cmake, '-S', str(directory), '-B', str(build), '-G', 'Ninja'])
        self.run_command([ARGS.cmake, '--build', str(build), '--target', 'native_expert_parity', 'native_grouped_parity'])
        properties = (build / 'target-properties.txt').read_text().splitlines()
        ctest = str(Path(ARGS.cmake).with_name('ctest')) if '/' in ARGS.cmake else 'ctest'
        tests = json.loads(self.run_command([ctest, '--test-dir', str(build), '--show-only=json-v1']))['tests']
        return properties, tests

    def run_command(self, command):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_optional_mmq_dependency(self):
        for backend, mmq in [('HIP', False), ('HIP', True), ('CUDA', True)]:
            with self.subTest(backend=backend, mmq=mmq):
                props, _ = self.configure(f'link-{backend}-{mmq}', backend, mmq)
                self.assertEqual('strata_mmq' in props[0].split(';'), mmq, props)
                self.assertEqual('STRATA_PARITY_MMQ=1' in props[1].split(';'), mmq, props)
                self.assertIn('strata_engine', props[0].split(';'))
                self.assertIn('strata_kernels_cpu', props[0].split(';'))

    def test_q36_pairs_follow_compiled_support(self):
        for kquants in [False, True]:
            for q6 in [False, True]:
                with self.subTest(kquants=kquants, q6=q6):
                    _, tests = self.configure(f'pairs-{kquants}-{q6}', 'CUDA', True, kquants, q6)
                    by_name = {test['name']: test for test in tests}
                    pairs = by_name['native_expert_parity_q36']['command'][2:]
                    expected = {'iq2_s/iq3_s', 'iq2_s/iq2_s', 'iq3_s/iq3_s', 'q8_0/q8_0'}
                    if kquants:
                        expected.update(['iq3_s/q4_K', 'q4_K/q4_K'])
                        if q6:
                            expected.add('q4_K/q6_K')
                    self.assertEqual(set(pairs), expected)
                    self.assertEqual(len(pairs), len(expected))
                    self.assertEqual(by_name['native_grouped_parity_q36']['command'][1:], ['--q36'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--work-dir', type=Path)
    ARGS, remaining = parser.parse_known_args()
    if ARGS.work_dir is None:
        with tempfile.TemporaryDirectory(prefix='parity-config-') as temporary:
            ARGS.work_dir = Path(temporary)
            unittest.main(argv=[__file__] + remaining)
    else:
        unittest.main(argv=[__file__] + remaining)
