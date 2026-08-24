#!/usr/bin/env python3
"""Create the headline cost comparison from official expanded-run results."""
from pathlib import Path

VALUES = {
    "Logged\nrouting": 167.67,
    "Starter\nbaseline": 147.23,
    "Our\nrouter": 64.59,
}
COLORS = ("#17324d", "#55758f", "#e96b4b")
WIDTH = 720
HEIGHT = 430


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    chart = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="430" viewBox="0 0 720 430">',
        '<rect width="720" height="430" fill="#f7f3ee"/>',
        '<text x="52" y="54" fill="#17324d" font-family="Georgia,serif" font-size="27" font-weight="700">Estimated input cost per full export</text>',
        '<text x="52" y="82" fill="#55758f" font-family="Arial,sans-serif" font-size="14">1,883 reconstructed trajectories · official challenge pricing</text>',
        '<line x1="80" y1="345" x2="670" y2="345" stroke="#c9c0b7"/>',
        '<text x="52" y="350" fill="#55758f" font-family="Arial,sans-serif" font-size="12" text-anchor="end">$0</text>',
        '<text x="52" y="185" fill="#55758f" font-family="Arial,sans-serif" font-size="12" text-anchor="end">$100</text>',
        '<line x1="80" y1="180" x2="670" y2="180" stroke="#ded7cf" stroke-dasharray="4 6"/>',
    ]
    max_value = 180
    bar_width = 128
    positions = (130, 300, 470)
    for (label, value), color, x in zip(VALUES.items(), COLORS, positions):
        height = value / max_value * 270
        y = 345 - height
        chart.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{height:.1f}" rx="4" fill="{color}"/>')
        chart.append(f'<text x="{x + bar_width / 2}" y="{y - 12:.1f}" fill="#17324d" font-family="Arial,sans-serif" font-size="19" font-weight="700" text-anchor="middle">${value:.2f}</text>')
        for index, line in enumerate(label.split("\\n")):
            chart.append(f'<text x="{x + bar_width / 2}" y="{375 + index * 18}" fill="#17324d" font-family="Arial,sans-serif" font-size="14" text-anchor="middle">{esc(line)}</text>')
    chart.extend([
        '<text x="52" y="410" fill="#55758f" font-family="Arial,sans-serif" font-size="12">Costs are estimated from chars/4 tokens; output tokens are unavailable.</text>',
        '</svg>',
    ])
    output = Path(__file__).parent.parent / "docs" / "results_comparison.svg"
    output.write_text("\n".join(chart))
    print(output)


if __name__ == "__main__":
    main()
