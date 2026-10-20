#!/usr/bin/env python3
"""Render self-contained README SVGs from committed benchmark samples.

Run from any directory: python tools/render_moementum_readme.py
No network, fonts, plotting packages or private captures required.
"""
from pathlib import Path
import json
import statistics
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/media'


def text(x, y, value, size=18, color='#d9e6f2', weight=400):
    return (f'<text x="{x}" y="{y}" font-family="Verdana,DejaVu Sans,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(str(value))}</text>')


def frame(height, title, description):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f'<rect width="1000" height="{height}" rx="20" fill="#15283b"/>']


def render():
    MEDIA.mkdir(parents=True, exist_ok=True)
    a = frame(300, 'MoEmentum', 'Experimental low-VRAM MoE inference. Built on Strata. Give your experts some momentum.')
    # Several expert lanes become one forward-moving M; no fabricated telemetry.
    for i in range(4):
        y = 73 + i * 43
        a.append(f'<path d="M50 {y} H112 L156 144 L200 {y} H242" fill="none" stroke="#5bc5d5" stroke-width="8" stroke-linecap="round" opacity="{0.4+i*0.2}"/>')
    a += [text(290, 108, 'MoEmentum', 60, '#f3f8fc', 700),
          text(294, 151, 'Give your experts some momentum.', 23, '#82d4df'),
          text(294, 208, 'LOW-VRAM MoE INFERENCE', 17, '#f3bf75', 700),
          text(294, 244, 'Experimental source build / built on Strata', 16, '#adbdce'), '</svg>']
    (MEDIA / 'moementum-hero.svg').write_text('\n'.join(a)+'\n')

    b = json.loads((ROOT / 'docs/moementum/benchmark.json').read_text())
    tasks = [('ledger_sum', 100), ('stable_priority', 190),
             ('long_record_lookup', 1767), ('long_filter', 2514)]
    a = frame(620, 'Controlled prefill throughput',
              'Baseline versus MoEmentum on RTX 3050 6GB, 32GiB RAM, six vCPUs. Three samples per arm for each of four tasks. All bars share a zero-based 0 to 120 TPS scale.')
    a += [text(42, 52, 'Same hardware. Less prefill time.', 29, '#f3f8fc', 700),
          text(42, 86, 'RTX 3050 6GB / 32GiB RAM / 6 vCPUs / IQ2_M', 17, '#adbdce'),
          '<rect x="44" y="108" width="14" height="14" fill="#899eb5"/>',
          text(66, 121, 'Pre-optimization baseline', 14),
          '<rect x="335" y="108" width="14" height="14" fill="#5bc5d5"/>',
          text(358, 121, 'MoEmentum', 14)]
    x0, scale = 230, 4.8
    for tick in range(0, 121, 30):
        x = x0 + tick * scale
        a.append(f'<path d="M{x} 151 V529" stroke="#31475c" stroke-width="1"/>')
        a.append(text(x-8, 552, tick, 13, '#adbdce'))
    for i, (key, tokens) in enumerate(tasks):
        med = b['v103']['timings_and_quality'][key]['prompt_ms']['values']
        baseline = tokens * 1000 / statistics.median(med['A'])
        candidate = tokens * 1000 / statistics.median(med['C'])
        reduction = 100 * (1 - baseline / candidate)
        y = 165 + i * 92
        a += [text(43, y+25, f'{tokens:,} tokens', 17, '#f3f8fc', 700),
              text(43, y+50, f'{reduction:.2f}% less time', 13, '#f3bf75')]
        for j, (rate, color) in enumerate([(baseline, '#899eb5'), (candidate, '#5bc5d5')]):
            yy = y + j * 30
            w = rate * scale
            a.append(f'<rect x="{x0}" y="{yy}" width="{w:.4f}" height="21" rx="3" fill="{color}"/>')
            a.append(text(x0+w+10, yy+16, f'{rate:.1f} TPS', 14, '#f3f8fc'))
    a += [text(828, 552, 'prefill TPS', 14, '#adbdce'),
          text(42, 587, 'Input tokens / median prefill seconds. Decode essentially unchanged; finite parity coverage.', 13, '#adbdce'), '</svg>']
    (MEDIA / 'moementum-prefill.svg').write_text('\n'.join(a)+'\n')
    print('Rendered docs/media/moementum-{hero,prefill}.svg')


if __name__ == '__main__':
    render()
