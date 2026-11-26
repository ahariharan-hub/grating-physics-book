"""Render the generated lesson's optional explanation with supported widgets."""
import html as _book_html
import re as _book_re
import ipywidgets as _book_widgets
from wave_optics.interference_setup import MirrorTiltExplorer as _BookMirrorTilt
from wave_optics.grating_intro import _close_widget_tree as _book_close_tree


def _book_phase_html(source):
    # These two fixed explanations use paragraphs, bold text and TeX only.
    # Escape the source before adding markup; leave all mathematical content
    # intact for the book's locally bundled MathJax typesetter.
    parts = []
    for block in _book_re.split(r'(\$\$.*?\$\$)', source, flags=_book_re.S):
        if block.startswith('$$') and block.endswith('$$'):
            parts.append('<div class="book-phase-equation" style="overflow-x:auto;margin:1em 0">'
                         + r'\[' + _book_html.escape(block[2:-2].strip()) + r'\]' + '</div>')
            continue
        for paragraph in _book_re.split(r'\n\s*\n', block.strip()):
            if not paragraph.strip():
                continue
            text = _book_html.escape(' '.join(paragraph.split()))
            text = _book_re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            text = _book_re.sub(r'\$(.+?)\$', lambda match: r'\(' + match[1] + r'\)', text)
            parts.append('<p>' + text + '</p>')
    return '<div class="book-phase-explanation">' + ''.join(parts) + '</div>'


if not getattr(_BookMirrorTilt, '_book_phase_markdown', False):
    _book_original_phase_ui = _BookMirrorTilt.phase_geometry_ui

    def _book_phase_geometry_ui(self):
        accordion = _book_original_phase_ui.fget(self)
        if not getattr(self, '_book_phase_html_ready', False):
            body = accordion.children[0]
            children, replaced = [], []
            for child in body.children:
                sources = [output.get('data', {}).get('text/markdown')
                           for output in getattr(child, 'outputs', ())]
                sources = [source for source in sources if source is not None]
                if sources:
                    children.append(_book_widgets.HTMLMath(
                        value=_book_phase_html('\n\n'.join(sources)),
                        layout=_book_widgets.Layout(width='100%', max_width='100%')))
                    replaced.append(child)
                else:
                    children.append(child)
            if len(replaced) != 2:
                for child in children:
                    if child not in body.children:
                        _book_close_tree(child)
                raise RuntimeError('Review the changed phase-geometry explanation widgets.')
            body.children = tuple(children)
            for child in replaced:
                _book_close_tree(child)
            self._book_phase_html_ready = True
        return accordion

    _BookMirrorTilt.phase_geometry_ui = property(
        _book_phase_geometry_ui, doc=_book_original_phase_ui.__doc__)
    _BookMirrorTilt._book_phase_markdown = True
