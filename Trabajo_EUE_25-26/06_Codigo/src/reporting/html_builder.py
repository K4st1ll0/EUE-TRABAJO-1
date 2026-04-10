from __future__ import annotations

from html import escape
from typing import Any, Iterable


INLINE_CSS = """
body { font-family: Arial, Helvetica, sans-serif; background: #f4f6f8; color: #1e2933; margin: 0; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
.panel { background: #fff; border: 1px solid #d7dde3; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { border: 1px solid #d7dde3; padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: #eef2f6; }
code { background: #eef2f6; padding: 1px 4px; border-radius: 4px; }
a { color: #0057a3; }
.status-ok { color: #116329; }
.status-warn { color: #935f00; }
.status-error { color: #9b1c1c; }
.metric { font-size: 1.8rem; font-weight: 700; margin: 8px 0 0; }
.muted { color: #64748b; }
"""


def render_html_page(
    title: str,
    body: str,
    stylesheet_href: str | None = None,
    inline_css: str | None = None,
) -> str:
    if stylesheet_href:
        styles = f'<link rel="stylesheet" href="{escape(stylesheet_href)}">'
    else:
        styles = f"<style>{inline_css or INLINE_CSS}</style>"
    return f"""<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <title>{escape(title)}</title>
    {styles}
  </head>
  <body>
    <main class="container">
      {body}
    </main>
  </body>
</html>
"""


def paragraph(text: str, class_name: str | None = None) -> str:
    cls = f' class="{escape(class_name)}"' if class_name else ""
    return f"<p{cls}>{escape(text)}</p>"


def section(title: str, body: str) -> str:
    return f'<section class="panel"><h2>{escape(title)}</h2>{body}</section>'


def card_grid(items: Iterable[dict[str, Any]]) -> str:
    chunks = []
    for item in items:
        status_class = f"status-{escape(str(item.get('status', 'ok')))}"
        chunks.append(
            "<div class=\"panel\">"
            f"<h3>{escape(str(item.get('title', '')))}</h3>"
            f"<div class=\"metric {status_class}\">{escape(str(item.get('value', '')))}</div>"
            f"<p class=\"muted\">{escape(str(item.get('caption', '')))}</p>"
            "</div>"
        )
    return '<div class="grid">' + "".join(chunks) + "</div>"


def unordered_list(items: Iterable[str]) -> str:
    values = list(items)
    if not values:
        return "<p class=\"muted\">Sin elementos.</p>"
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in values) + "</ul>"


def key_value_table(data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        rows.append(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        )
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def data_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "<p class=\"muted\">No hay datos.</p>"

    columns = columns or list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        tds = "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr>{tds}</tr>")
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )
